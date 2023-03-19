# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import math
import os
import click
from tqdm import tqdm
import dnnlib
import PIL.Image
import torch
import legacy
from visual_utils import image_grid


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dataroot', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num_images', help='number of generated images', type=click.IntRange(min=1, max=100), default=10, show_default=True)
@click.option('--fixed_style', help='fixed style for all images', is_flag=True)
@click.option('--full_quality', help='save image without loss data', is_flag=True)
def generate_images(
    network_pkl: str,
    dataroot: str,
    outdir: str,
    num_images: int,
    fixed_style: bool,
    full_quality: bool
):
    torch.random.seed()
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

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
    stylepairs = 32

    the_style_1 = torch.randn([stylepairs, G.latent_dim]).to(device)
    the_style_2 = torch.randn([stylepairs, G.latent_dim]).to(device)

    # Generate images.
    for i, imgs in tqdm(enumerate(eval_set), total=len(eval_set)):
        if i > len(eval_set):
            break

        img = imgs['A'].to(device).unsqueeze(0)
        img_path = imgs['A_paths']

        content, _ = G.encode(img)
        bbx = []
        for k in range(stylepairs):
            out_images = [ img ]
            if not fixed_style:
                the_style_1 = torch.randn([stylepairs, G.latent_dim]).to(device)
                the_style_2 = torch.randn([stylepairs, G.latent_dim]).to(device)
            for dl in range(num_images+1):
                style = torch.lerp(the_style_1, the_style_2, dl / num_images)
                out = G.decode(content, style[k:k+1])
                out_images.append(out)
            bbx.append(torch.cat(out_images))

        out = image_grid(bbx, len(out_images))
        out = (out.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        extra_args = {}
        if full_quality:
            extra_args['quality'] = 100
            extra_args['subsampling'] = 0
        PIL.Image.fromarray(out.cpu().numpy(), 'RGB').save(f'{outdir}/{os.path.basename(img_path)}', **extra_args)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
