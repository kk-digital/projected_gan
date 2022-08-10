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


"""Main training loop."""

import io
import itertools
import os
import time
import copy
import json
from typing import List, Union
import dill
import psutil
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
import pickle
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from tdlogger import TdLogger

import legacy
from i2imetrics.all_score import calculate_scores_given_paths

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(eval_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // eval_set.resolution, 7, 32)
    gh = np.clip(4320 // eval_set.resolution, 4, 32)
    gh //= 2

    if True:
        if len(eval_set) < gw * gh:
            distance = gw * gh
            cw, ch = gw, gh
            for h in range(4, 17):
                for w in range(max(7, h*2), min(33, h*4+1)):
                    if w * h >= len(eval_set) and (w * h - len(eval_set)) < distance:
                        distance = w * h - len(eval_set)
                        cw, ch = w, h
            gw, gh = cw, ch

        all_indices = list(range(len(eval_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    imagesB, imagesA = zip(*[(eval_set[i]['B'], eval_set[i]['A']) for i in grid_indices])
    return (gw, gh), np.stack(imagesB), np.stack(imagesA)

#----------------------------------------------------------------------------

def save_image_grid(imgA, imgB, fname, drange, grid_size):
    assert imgA.shape == imgB.shape

    lo, hi = drange
    imgA = np.asarray(imgA, dtype=np.float32)
    imgA = (imgA - lo) * (255 / (hi - lo))
    imgA = np.rint(imgA).clip(0, 255).astype(np.uint8)
    imgB = np.asarray(imgB, dtype=np.float32)
    imgB = (imgB - lo) * (255 / (hi - lo))
    imgB = np.rint(imgB).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    N, C, H, W = imgA.shape
    assert gw * gh >= N
    if N < gw * gh:
        imgA = np.concatenate([imgA, np.zeros((gw*gh-N, C, H, W), dtype=np.uint8)], axis=0)
        imgB = np.concatenate([imgB, np.zeros((gw*gh-N, C, H, W), dtype=np.uint8)], axis=0)

    imgA = imgA.reshape([gh, gw, C, H, W])
    imgA = imgA.transpose(0, 3, 1, 4, 2)
    imgA = imgA.reshape([gh * H, gw * W, C])
    imgB = imgB.reshape([gh, gw, C, H, W])
    imgB = imgB.transpose(0, 3, 1, 4, 2)
    imgB = imgB.reshape([gh * H, gw * W, C])

    img = np.zeros((2 * gh * H, gw * W, C), dtype=np.uint8)
    for i in range(gh):
        img[2*i*H:(2*i+1)*H] = imgA[i*H:(i+1)*H]
        img[(2*i+1)*H:(2*i+2)*H] = imgB[i*H:(i+1)*H]

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname, format='png')
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname, format='png')

#----------------------------------------------------------------------------

def save_image(img: torch.Tensor, path: str):
    img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(path)

def eval_metrics(generator, eval_set, cur_nimg: int, run_dir: str, device):
    result_dir = os.path.join(run_dir, 'results', f'{cur_nimg//1000:06d}')
    real_B_path = os.path.join(result_dir, 'real_B')
    fake_B_path = os.path.join(result_dir, 'fake_B')
    os.makedirs(real_B_path, exist_ok=True)
    os.makedirs(fake_B_path, exist_ok=True)
    for i, imgs in enumerate(eval_set):
        if i > len(eval_set):
            break

        A, B = imgs['A'].to(device).unsqueeze(0), imgs['B'].to(device).unsqueeze(0)
        A_basename = os.path.basename(imgs['A_paths'])
        B_basename = os.path.basename(imgs['B_paths'])
        fake_B = generator(A)
        save_image(B[0], os.path.join(real_B_path, B_basename))
        save_image(fake_B[0], os.path.join(fake_B_path, A_basename))
    
    result = calculate_scores_given_paths(
                        [fake_B_path, real_B_path], device=device, batch_size=50, dims=2048,
                        use_fid_inception=True, torch_svd=False)
    return result

#----------------------------------------------------------------------------

def set_requires_grad(module: Union[List[torch.nn.Module], torch.nn.Module], require: bool):
    if not isinstance(module, list):
        module = [module]
    for mod in module:
        mod.requires_grad_(require)


def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    F_kwargs                = {},       # Options for min mutual-info network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    restart_every           = -1,       # Time interval in seconds to exit code
    desc: str = '',
    logger: TdLogger = None,
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    __RESTART__ = torch.tensor(0., device=device)       # will be broadcasted to exit loop
    __CUR_NIMG__ = torch.tensor(resume_kimg * 1000, dtype=torch.long, device=device)
    __CUR_TICK__ = torch.tensor(0, dtype=torch.long, device=device)
    __BATCH_IDX__ = torch.tensor(0, dtype=torch.long, device=device)
    __PL_MEAN__ = torch.zeros([], device=device)
    best_fid = 9999

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    eval_set_kwargs = dnnlib.EasyDict(**training_set_kwargs)
    eval_set_kwargs.phase = 'test'
    eval_set_kwargs.serial_batches = True
    eval_set_kwargs.preprocess = 'resize'
    eval_set = dnnlib.util.construct_class_by_name(**eval_set_kwargs) # subclass of training.dataset.Dataset
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(img_resolution=training_set.resolution, img_channels=3)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    F = dnnlib.util.construct_class_by_name(**F_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Check for existing checkpoint
    ckpt_pkl = None
    if restart_every > 0 and os.path.isfile(misc.get_ckpt_path(run_dir)):
        ckpt_pkl = resume_pkl = misc.get_ckpt_path(run_dir)

    # Print network summary tables.
    if rank == 0:
        fimg = torch.empty([batch_gpu, 3, training_set.resolution, training_set.resolution], device=device)
        img = misc.print_module_summary(G, [fimg])
        misc.print_module_summary(D, [img])

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, F, G_ema]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, G_ema=G_ema, D=D, F=F, resolution=training_set.resolution, **loss_kwargs) # subclass of training.loss.Loss

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('F', F), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        if ckpt_pkl is not None:            # Load ticks
            __CUR_NIMG__ = resume_data['progress']['cur_nimg'].to(device)
            __CUR_TICK__ = resume_data['progress']['cur_tick'].to(device)
            __BATCH_IDX__ = resume_data['progress']['batch_idx'].to(device)
            __PL_MEAN__ = resume_data['progress'].get('pl_mean', torch.zeros([])).to(device)
            best_fid = resume_data['progress']['best_fid']       # only needed for rank == 0

        del resume_data

    phases = []
    opt_G = dnnlib.util.construct_class_by_name(params=itertools.chain(G.parameters(), F.parameters()), **G_opt_kwargs) # subclass of torch.optim.Optimizer
    phases += [dnnlib.EasyDict(name='Gboth', module=[G, F], opt=opt_G, interval=1)]

    opt_D = dnnlib.util.construct_class_by_name(D.parameters(), **D_opt_kwargs) # subclass of torch.optim.Optimizer
    phases += [dnnlib.EasyDict(name='Dmain', module=D, opt=opt_D, interval=1)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    imagesA, imagesB = None, None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, imagesB, imagesA = setup_snapshot_image_grid(eval_set=eval_set)
        save_image_grid(imagesA, imagesB, os.path.join(run_dir, 'reals.png'), drange=[-1,1], grid_size=grid_size)

        images_A = torch.from_numpy(imagesA).to(device).split(batch_gpu)
        images = torch.cat([G_ema(img).cpu() for img in images_A]).numpy()

        save_image_grid(imagesA, images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
        del images_A

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    if num_gpus > 1:  # broadcast loaded states to all
        torch.distributed.broadcast(__CUR_NIMG__, 0)
        torch.distributed.broadcast(__CUR_TICK__, 0)
        torch.distributed.broadcast(__BATCH_IDX__, 0)
        torch.distributed.broadcast(__PL_MEAN__, 0)
        torch.distributed.barrier()  # ensure all processes received this info
    cur_nimg = __CUR_NIMG__.item()
    cur_tick = __CUR_TICK__.item()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = __BATCH_IDX__.item()
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)
    if hasattr(loss, 'pl_mean'):
        loss.pl_mean.copy_(__PL_MEAN__)
    while True:

        with torch.autograd.profiler.record_function('data_fetch'):
            imgs = next(training_set_iterator)
            real_A, real_B = imgs['A'], imgs['B']
            real_A = real_A.to(device).to(torch.float32).split(batch_gpu)
            real_B = real_B.to(device).to(torch.float32).split(batch_gpu)

        # Execute training phases.
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            set_requires_grad(phase.module, True)

            if phase.name in ['Dmain', 'Dboth', 'Dreg'] and hasattr(phase.module, 'feature_network'):
                phase.module.feature_network.requires_grad_(False)

            for real_A_img, real_B_img in zip(real_A, real_B):
                loss.accumulate_gradients(phase=phase.name, real_A=real_A_img, real_B=real_B_img, gain=phase.interval, cur_nimg=cur_nimg)
            set_requires_grad(phase.module, False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                mods = phase.module if isinstance(phase.module, list) else [phase.module]
                params = []
                for mod in mods:
                    params += [ param for param in mod.parameters() if param.grad is not None]

                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Check for restart.
        if (rank == 0) and (restart_every > 0) and (time.time() - start_time > restart_every):
            print('Restart job...')
            __RESTART__ = torch.tensor(1., device=device)
        if num_gpus > 1:
            torch.distributed.broadcast(__RESTART__, 0)
        if __RESTART__:
            done = True
            print(f'Process {rank} leaving...')
            if num_gpus > 1:
                torch.distributed.barrier()

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images_A = torch.from_numpy(imagesA).to(device).split(batch_gpu)
            images = torch.cat([G_ema(img).cpu() for img in images_A]).numpy()
            img_path = os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png')
            buf = io.BytesIO()
            save_image_grid(imagesA, images, buf, drange=[-1,1], grid_size=grid_size)
            with io.open(img_path, 'wb') as f:
                f.write(buf.getvalue())
            if logger is not None:
                logger.sendBlobFile(buf, os.path.basename(img_path), f"/validation_images/{desc}/{os.path.basename(img_path)}", f"{desc}/validationImage")
            del images_A

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, F=F, G_ema=G_ema, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    snapshot_data[key] = value
                del value # conserve memory

        # Save Checkpoint if needed
        if (rank == 0) and (restart_every > 0) and (network_snapshot_ticks is not None) and (
                done or cur_tick % network_snapshot_ticks == 0):
            snapshot_pkl = misc.get_ckpt_path(run_dir)
            # save as tensors to avoid error for multi GPU
            snapshot_data['progress'] = {
                'cur_nimg': torch.LongTensor([cur_nimg]),
                'cur_tick': torch.LongTensor([cur_tick]),
                'batch_idx': torch.LongTensor([batch_idx]),
                'best_fid': best_fid,
            }
            if hasattr(loss, 'pl_mean'):
                snapshot_data['progress']['pl_mean'] = loss.pl_mean.cpu()

            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        if cur_tick and (snapshot_data is not None) and (len(metrics) > 0):
            fid_val = best_fid + 1
            if rank == 0:
                print('Evaluating metrics...')
            result = eval_metrics(lambda img: G_ema(img), eval_set=eval_set, cur_nimg=cur_nimg, run_dir=run_dir, device=device)
            _, kid, fid_val = result[0]
            kid_val, _ = kid
            if rank == 0:
                training_stats.report('Metrics/FID', fid_val)
                training_stats.report('Metrics/KID', kid_val)

            # save best fid ckpt
            snapshot_pkl = os.path.join(run_dir, f'best_model.pkl')
            cur_nimg_txt = os.path.join(run_dir, f'best_nimg.txt')
            if rank == 0:
                if fid_val < best_fid:
                    best_fid = fid_val

                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)
                    # save curr iteration number (directly saving it to pkl leads to problems with multi GPU)
                    with open(cur_nimg_txt, 'w') as f:
                        f.write(str(cur_nimg))

        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None) and \
                    not (phase.start_event.cuda_event == 0 and phase.end_event.cuda_event == 0):            # Both events were not initialized yet, can happen with restart
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()
        training_stats.reset()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if logger is not None:
            keys = list(stats_dict.keys())
            tables = set(map(lambda key: key.split('/')[0], keys))
            for tbl in tables:
                data = {}
                match_keys = [ k for k in keys if k.startswith(tbl + '/') ]
                for mk in match_keys:
                    key = mk[len(tbl)+1:]
                    val = stats_dict[mk].mean
                    data[key] = val
                logger.send(data, group=f'{desc}/{tbl}')
            logger.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------