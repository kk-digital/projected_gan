# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import io
import os
import pickle
import click
from tqdm import tqdm
import dnnlib
import PIL.Image
import torch
import legacy


def ffnn(arr, n):
    out = []
    bk = []
    for v in arr:
        bk.append(v)
        if len(bk) == n:
            out.append(bk)
            bk = []
    if len(bk) > 0:
        out.append(bk)
    return out

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dataroot', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num_per_image', help='generator multiple image for one input', metavar='INT', type=click.IntRange(min=1), default=5)
@click.option('--batch_size', help='batch size', type=int, default=10, show_default=True)
@click.option('--same_style', help='using same style for all images', is_flag=True)
def generate_images(
    network_pkl: str,
    dataroot: str,
    outdir: str,
    num_per_image: int,
    batch_size: int,
    same_style: bool
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if 'munit' in network_pkl:
        with io.open(network_pkl, 'rb') as pf:
            G = pickle.load(pf).requires_grad_(False).eval().to(device)
            style_channels = G.autoencoder_a.style_channels

            def new_forward(img, latens = None):
                latens = [ torch.randn([1, style_channels,1,1]).to(device) for _ in range(num_per_image) ] if latens is None else latens
                ans = []
                for lats in ffnn(latens, batch_size):
                    s = len(lats)
                    lats = torch.cat(lats, 0)
                    out, _ = G.inference({'images_a': img.expand(s,-1,-1,-1), 'key': { 'images_a': { 'filename': '' }}}, styles=lats)
                    for i in range(s):
                        ans.append(out[i])
                return ans, latens
            G.forward = new_forward
    elif 'GANsNRoses' in network_pkl or 'gansnroses' in network_pkl:
        with io.open(network_pkl, 'rb') as pf:
            G = pickle.load(pf, fix_imports=True).requires_grad_(False).eval().to(device)
            latent_dim: int = 8
            old_forward = G.forward
            def new_forward(img, latens = None):
                latens = [ torch.randn([1, latent_dim]).to(device) for _ in range(num_per_image) ] if latens is None else latens
                ans = []
                for lats in ffnn(latens, batch_size):
                    s = len(lats)
                    lats = torch.cat(lats, 0)
                    imm = img.expand(s, -1, -1, -1)
                    out, _, _ = old_forward(imm, lats)
                    for i in range(s):
                        ans.append(out[i])
                return ans, latens
            G.forward = new_forward
    else:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
            def new_forward(img, latens = None):
                latent_dim: int = G.latent_dim
                latens = [ torch.randn([1, latent_dim]).to(device) for _ in range(num_per_image) ] if latens is None else latens
                content, _ = G.encode(img)

                ans = []
                for lats in ffnn(latens, batch_size):
                    s = len(lats)
                    lats = torch.cat(lats, 0)
                    if type(content) == list:
                        c = []
                        for cc in content:
                            c.append(cc.expand(s, -1, -1, -1))
                    else:
                        c = content.expand(s, -1, -1, -1)
                    imgs: torch.Tensor = G.decode(c, lats)
                    for i in range(s):
                        ans.append(imgs[i])
                return ans, latens
            G.forward = new_forward

    os.makedirs(outdir, exist_ok=True)
    eval_set_kwargs = dnnlib.EasyDict()
    eval_set_kwargs.class_name = 'training.unaligned_dataset.createDataset'
    eval_set_kwargs.dataroot = dataroot
    eval_set_kwargs.dataname = os.path.basename(dataroot)
    eval_set_kwargs.phase = 'test'
    eval_set_kwargs.preprocess = 'resize'
    eval_set_kwargs.load_size = 256
    eval_set_kwargs.crop_size = 256
    eval_set_kwargs.flip = False
    eval_set_kwargs.serial_batches = False
    eval_set_kwargs.max_dataset_size = 10000
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs)

    # Generate images.
    latens = None
    for i, imgs in tqdm(enumerate(eval_set), total=len(eval_set)):
        if i > len(eval_set):
            break

        if not same_style:
            latens = None

        img = imgs['A'].to(device).unsqueeze(0)
        img_path = imgs['A_paths']

        out_imgs, latens = G.forward(img, latens)
        k = 1
        for out in out_imgs:
            out = (out.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            pp = os.path.basename(img_path)
            pps = os.path.splitext(pp)
            PIL.Image.fromarray(out.cpu().numpy(), 'RGB').save(f'{outdir}/{pps[0]}_{k}{pps[1]}', quality=100, subsampling=0)
            k = k + 1


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
