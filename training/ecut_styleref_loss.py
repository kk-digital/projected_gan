# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
from functools import reduce
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
import kornia.augmentation as K
import kornia
from models.transformer import VisioniTransformer
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from torch import nn
from models import losses
from models.patchnce import PatchNCELoss
from models.fastae_networks import Encoder as Ev1, Generator as Gv1
from models.fastae_v2_networks import Encoder as Ev2, Generator as Gv2
from models.fastae_v3_networks import Encoder as Ev3, Generator as Gv3
from models.fastae_v5_networks import Encoder as Ev9, Generator as Gv9
from models.fastae_v6_networks import Encoder as Ev10, Generator as Gv10
from models.fastae_v7_networks import Encoder as Ev11, Generator as Gv11
from models.fastae_v8_networks import Encoder as Ev12, Generator as Gv12
from models.style_networks import Encoder as Ev4, Generator as Gv4
from models.style_v2_networks import Encoder as Ev5, Generator as Gv5
from models.style_v3_networks import Encoder as Ev7, Generator as Gv7
from models.style_v4_networks import Encoder as Ev8, Generator as Gv8
from thirdparty.GANsNRoses.model import Encoder as Ev_gnr, Generator as Gv_gnr, LatDiscriminator

valid_gen_encoder = [
    (Gv1, Ev1),
    (Gv2, Ev2),
    (Gv3, Ev3),
    (Gv4, Ev4),
    (Gv5, Ev5),
    (Gv7, Ev7),
    (Gv8, Ev8),
    (Gv9, Ev9),
    (Gv10, Ev10),
    (Gv11, Ev11),
    (Gv12, Ev12),
    (Gv_gnr, Ev_gnr)
]

class Loss:
    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ECUTStyleRefLoss(Loss):
    def __init__(self, device, G, D, F, resolution: int,
                 nce_layers: list, feature_net: str, nce_idt: bool, num_patches: int,
                 style_recon_nce: bool = False, style_recon_force_idt: bool = False, feature_attn_layers: int=0, patch_max_shape: Tuple[int,int]=(256,256),
                 normalize_transformer_out: bool = True, sim_pnorm: float = 0,
                 lambda_style_GAN: float=2.0, lambda_GAN: float=1.0, lambda_NCE: float=1.0, lambda_identity: float = 0,
                 lambda_style_consis: float=50.0, lambda_style_recon: float = 5,
                 output_cons: bool = False, latent_bank_size: int = 256,
                 blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        assert reduce(lambda a, b: a or b, map(lambda u: isinstance(G, u[0]), valid_gen_encoder))
        self.G: Gv1 = G
        self.D = D
        self.F = F
        self.resolution = resolution
        self.nce_idt = nce_idt
        self.num_patches = num_patches
        self.feature_attn_layers = feature_attn_layers
        self.patch_max_shape = patch_max_shape
        self.normalize_transformer_out = normalize_transformer_out
        self.lambda_GAN = lambda_GAN
        self.lambda_style_GAN = lambda_style_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_identity = lambda_identity
        self.lambda_style_consis = lambda_style_consis
        self.lambda_style_recon = lambda_style_recon
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.criterionIdt = torch.nn.MSELoss()
        self.criterionStyleRecon = losses.ContrastiveNCELoss2() if style_recon_nce or True else torch.nn.MSELoss()
        self.style_recon_force_idt = style_recon_force_idt
        self.latent_dim = self.G.latent_dim
        self.output_cons = output_cons
        self.latent_bank_size = latent_bank_size
        self.aug = nn.Sequential(
            K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
            kornia.geometry.transform.Resize(256+30),
            K.RandomCrop((256,256)),
            K.RandomHorizontalFlip(),
        )

        if feature_net == 'efficientnet_lite':
            self.netPre = losses.EfficientNetLite().to(self.device)
        elif feature_net == 'vgg16':
            self.netPre = losses.VGG16().to(self.device)
        elif feature_net == 'learned':
            self.netPre = self.G
            if isinstance(self.G, Gv3):
                nce_layers = [2,4,6,8]
            elif isinstance(self.G, Gv4) or isinstance(self.G, Gv7) or isinstance(self.G, Gv8):
                nce_layers = [2,6,9,12,14,18]
            elif isinstance(self.G, Gv5):
                nce_layers = [2,6,9,12,15,18]
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(feature_net)

        # define loss functions
        self.criterionNCE = []
        patchnce_opt = dnnlib.EasyDict(
            nce_includes_all_negatives_from_minibatch=False,
            batch_size=1,
            nce_T=0.07,
        )

        self.nce_layers = nce_layers
        for _ in nce_layers:
            self.criterionNCE.append(PatchNCELoss(patchnce_opt, pnormSim=sim_pnorm).to(self.device))

        self.setup_F()
        self.F.train().requires_grad_(False).to(self.device)

    def setup_nce_features_attn(self, img):
        if self.feature_attn_layers == 0:
            return

        feat = self.netPre(img, self.nce_layers, encode_only=True)
        if isinstance(feat, tuple):
            feat = feat[1]
        
        max_h, max_w = self.patch_max_shape
        vit_modules = nn.ModuleList()
        for ft in feat:
            _, c, h, w = ft.shape
            out_h, out_w = min(max_h, h), min(max_w, w)
            assert h % out_h == 0 and w % out_w == 0
            patch_size = (h // out_h, w // out_w)
            vit = VisioniTransformer(c, (h, w), patch_size, min(512, c * patch_size[0] * patch_size[1]), self.feature_attn_layers, 4, self.normalize_transformer_out)
            vit_modules.append(vit)
        self.F.vit_modules = vit_modules.to(self.device)
        latent_bank = torch.zeros((self.latent_bank_size, self.latent_dim), device=self.device)
        self.F.latent_bank = nn.parameter.Parameter(latent_bank, requires_grad=False)

    def get_nce_features(self, img):
        feats = self.netPre(img, self.nce_layers, encode_only=True)
        if isinstance(feats, tuple):
            feats = feats[1]
        
        if self.feature_attn_layers > 0:
            feats = list(map(lambda a: a[0](a[1]), zip(self.F.vit_modules, feats)))

        return feats

    def setup_F(self):
        fimg = torch.empty([1, 3, self.resolution, self.resolution], device=self.device)
        self.setup_nce_features_attn(fimg)
        feat = self.get_nce_features(fimg)

        self.F.create_mlp(feat)
        self.D.latent_dis = LatDiscriminator(self.latent_dim).to(self.device).requires_grad_(False)

    def calculate_NCE_loss(self, feat_k, feat_q):
        n_layers = len(self.nce_layers)

        feat_k_pool, sample_ids = self.F(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.F(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        weights = [ 1 / n_layers for i in range(0, n_layers) ]

        for f_q, f_k, crit, weight, _ in zip(feat_q_pool, feat_k_pool, self.criterionNCE, weights, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean() * weight
        
        return total_nce_loss

    def run_D(self, img, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img)
        if isinstance(logits, list):
            l = []
            for lg in logits:
                l.append(lg.mean())
            logits = torch.stack(l, dim=0)
        return logits

    def accumulate_gradients(self, phase, real_A, real_B, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG
        batch_size = real_A.size(0)
        device = real_A.device

        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
        n_iter = cur_nimg // real_A.size(0)

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                reverse_se = self.G.encoder
                A_content = self.G.content_encode(real_A)
                B_style = reverse_se.style_encode(real_B)
                fake_B = self.G.decode(A_content, B_style)
                latent_bank = self.F.latent_bank.data

                # Adversarial loss
                gen_logits = self.run_D(fake_B, blur_sigma=blur_sigma)
                loss_Gmain_GAN = (-gen_logits).mean()
                loss_Gmain = self.lambda_GAN * loss_Gmain_GAN
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/gan', loss_Gmain_GAN)

                if self.lambda_NCE > 0:
                    loss_Gmain_NCE = self.calculate_NCE_loss(self.get_nce_features(real_A), self.get_nce_features(fake_B))
                    training_stats.report('Loss/G/NCE', loss_Gmain_NCE)
                    loss_Gmain = loss_Gmain + loss_Gmain_NCE * self.lambda_NCE
                
                if self.lambda_style_consis > 0:
                    idx = np.random.randint(batch_size)
                    real_B_aug = self.aug(real_B[[idx]].expand_as(real_B))
                    fake_B_aug = self.aug(fake_B[[idx]].expand_as(fake_B))
                    same_style_B = torch.concat([real_B_aug, fake_B_aug], dim=0)
                    styles = reverse_se.style_encode(same_style_B)
                    loss_Gmain_stylecons = styles.var(0, unbiased=False).sum()
                    training_stats.report('Loss/G/StyleConsistency', loss_Gmain_stylecons)
                    loss_Gmain = loss_Gmain + (loss_Gmain_stylecons) * self.lambda_style_consis

                if self.lambda_style_recon > 0:
                    recon_style = reverse_se.style_encode(fake_B)
                    loss_Gmain_style_recon = self.criterionStyleRecon(recon_style, latent_bank, B_style)
                    latent_bank = torch.cat([latent_bank[recon_style.size(0):], recon_style.detach()], dim=0)
                    training_stats.report('Loss/G/StyleReconstruction', loss_Gmain_style_recon)
                    loss_Gmain = loss_Gmain + loss_Gmain_style_recon * self.lambda_style_recon
                    self.F.latent_bank = nn.parameter.Parameter(latent_bank, requires_grad=False).to(self.device)

                self.fake_B = fake_B.detach()
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.backward()

        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_logits = self.run_D(self.fake_B, blur_sigma=blur_sigma)
                loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_logits = self.run_D(real_B, blur_sigma=blur_sigma)
                loss_Dreal = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
            
            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.backward()
