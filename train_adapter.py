import argparse
import copy
import logging
import os
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.adapter import create_adapter
from models.dmft import DMFT_models
from models.sit import SiT_models
from loss import AdapterLoss
from samplers import euler_sampler, adapter_euler_sampler

from diffusers.models import AutoencoderKL
import wandb
import math
from torchvision.utils import make_grid
import functools

from inverse_problems.measurements import Identity, InpaintingRandom, InpaintingBox, SuperResolution, GaussianBlur, MotionBlur


import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder


torch.set_num_threads(4)
torch.set_num_interop_threads(1)
logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x

def vae_decode_func(vae, latents_scale, latents_bias, latents):
    latents = (latents - latents_bias) / latents_scale
    images = vae.decode(latents).sample
    return images


def initialize_inverse_problems(device='cuda'):
    '''
        INVERSE_PROBLEMS = {
            0: null, # Identity() without sigma i.e. no noise added
            1: Identity(), # sigma=0.05
            2: InpaintingRandom(), # resolution=256, prob_range=(0.3, 0.7), device='cuda', sigma=0.05
            3: InpaintingBox(), # resolution=256, box_range=(32, 128), device='cuda', sigma=0.05
            4: SuperResolution(), # resolution=256, scale_factor=4, device='cuda', sigma=0.05
            5: GaussianBlur(), # kernel_size=61, intensity=3.0, device='cuda', sigma=0.05
            6: MotionBlur(), # kernel_size=61, intensity=0.5, device='cuda', sigma=0.05
        }
    '''
    identity = Identity(device=device)
    inpaint_random = InpaintingRandom(device=device)
    inpaint_box = InpaintingBox(device=device)
    super_resolution = SuperResolution(device=device)
    gaussian_blur = GaussianBlur(device=device)
    motion_blur = MotionBlur(device=device)

    INVERSE_PROBLEMS = {
        # 0: identity,  # Identity() without noise added --> not used in final run
        0: identity,
        1: inpaint_random,
        2: inpaint_box,
        3: super_resolution,
        4: gaussian_blur,
        5: motion_blur,
    }

    return INVERSE_PROBLEMS

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float = 0.9999,
    only_trainable: bool = True,   # skip params with requires_grad=False
    update_buffers: bool = True    # keep buffers (e.g., running stats) in sync
):
    """
    Exponential moving average update.
    - Parameters with requires_grad=False are skipped when only_trainable=True.
    - Buffers are copied directly (no decay) if update_buffers=True.
    """
    # Map names consistently (DDP may prepend "module.")
    def _strip(name: str) -> str:
        for pre in ("module.", "orig_mod.", "_orig_mod."):
            if name.startswith(pre):
                return name[len(pre):]
        return name

    # Build quick-lookup sets & dicts
    trainable = {_strip(n) for n, p in model.named_parameters() if p.requires_grad}
    ema_params = { _strip(n): p for n, p in ema_model.named_parameters() }
    model_params = { _strip(n): p for n, p in model.named_parameters() }

    # EMA on parameters
    for name, p_model in model_params.items():
        if only_trainable and name not in trainable:
            continue
        p_ema = ema_params.get(name, None)
        if p_ema is None:
            continue
        # Guard: only EMA float tensors
        if p_ema.is_floating_point():
            p_ema.mul_(decay).add_(p_model.data, alpha=1.0 - decay)
        else:
            p_ema.copy_(p_model.data)

    # Keep buffers in sync (no EMA decay)
    if update_buffers:
        ema_buffers = { _strip(n): b for n, b in ema_model.named_buffers() }
        for name, b_model in model.named_buffers():
            name = _strip(name)
            b_ema = ema_buffers.get(name, None)
            if b_ema is not None:
                b_ema.copy_(b_model)


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    block_kwargs = {"attn_func": args.attn_func, "qk_norm": args.qk_norm}
    model = DMFT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg_embedding=(args.cfg_prob > 0.0),
        dmf_depth=args.dmf_depth,
        use_logvar=False,
        pretrained_sigma=args.pretrained_sigma,
        **block_kwargs
    )
    # Adapter Network
    adapter_model = create_adapter(
        variant=args.adapter_model,
        num_inverse_problems=args.num_inv_probs,
        image_size=args.resolution,
        latent_size=latent_size,
        latent_channels=4,
    ).to(device)

    if args.pretrained and args.resume_step == 0:
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained model from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        msg = model.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            logger.info(msg)

    if args.adapter_pretrained:
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained adapter model from {args.adapter_pretrained}")
        ckpt_adapter = torch.load(args.adapter_pretrained, map_location="cpu", weights_only=False)
        msg = adapter_model.load_state_dict(ckpt_adapter, strict=False)
        if accelerator.is_main_process:
            logger.info(msg)

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    adapter_model = adapter_model.to(device)
    ema_adapter = deepcopy(adapter_model).to(device)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    vae.eval()

    requires_grad(ema, False)
    requires_grad(ema_adapter, False)
    if args.g_type == "distill":
        model_t = SiT_models["SiT-XL/2"](
            input_size=latent_size,
            num_classes=args.num_classes,
            use_cfg_embedding=True,
            **block_kwargs
        )
        msg = model_t.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            logger.info(msg)
        model_t.requires_grad_(False)
        model_t = model_t.to(device)
        model_t.eval()
    else:
        model_t = None
        
    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.0] * 4).view(1, 4, 1, 1).to(device)
    decode_func = functools.partial(vae_decode_func, vae, latents_scale, latents_bias)

    # create loss function
    loss_config = dict(
        P_mean=args.P_mean,
        P_std=args.P_std,
        P_mean_t=args.P_mean_t,
        P_std_t=args.P_std_t,
        P_mean_r=args.P_mean_r,
        P_std_r=args.P_std_r,
        cfg_prob=args.cfg_prob,
        omega=args.omega,
        g_min=args.g_min,
        g_max=args.g_max,
        path_type=args.path_type,
        g_type=args.g_type,
    )
    loss_fn = AdapterLoss(**loss_config)

    if accelerator.is_main_process:
        logger.info(f"Training Parameters for Flow Map: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Training Parameters for Adapter: {sum(p.numel() for p in adapter_model.parameters()):,}")
    # Setup optimizer
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    model_params = list(p for p in model.parameters() if p.requires_grad)
    adapter_params = list(p for p in adapter_model.parameters() if p.requires_grad)
    if args.train_adapter_only:
        combined_params = adapter_params
    else:
        combined_params = model_params + adapter_params
    if not set(model_params).isdisjoint(set(adapter_params)) or len(set(combined_params)) != len(combined_params):
        raise ValueError("Model and adapter parameters overlap!")
    # Separate optimizers so the flow map and adapter can use different LRs.
    optimizer_flow = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    optimizer_adapter = torch.optim.AdamW(
        adapter_model.parameters(),
        lr=args.adapter_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resolution)),
        # transforms.RandomHorizontalFlip(), # REPA and DMF do not use but others use, TODO: confirm
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_dataset = ImageFolder(args.data_dir, transform=transform)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0.0, only_trainable=False, update_buffers=True)
    update_ema(ema_adapter, adapter_model, decay=0.0, only_trainable=False, update_buffers=True)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    adapter_model.train()
    ema.eval()  # EMA model should always be in eval mode
    ema_adapter.eval()

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False
        )
        msg = model.load_state_dict(ckpt['model'])
        msg_adapter = adapter_model.load_state_dict(ckpt['adapter_model'])
        if accelerator.is_main_process:
            logger.info(msg)
            logger.info(msg_adapter)
        ema.load_state_dict(ckpt['ema'])
        ema_adapter.load_state_dict(ckpt['ema_adapter'])
        optimizer_flow.load_state_dict(ckpt['opt_flow'])
        optimizer_adapter.load_state_dict(ckpt['opt_adapter'])
        global_step = ckpt['steps']
        if accelerator.is_main_process:
            logger.info(f"Resume from {global_step}")
            
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    adapter_model.train()
    ema.eval()  # EMA model should always be in eval mode
    ema_adapter.eval()

    model, adapter_model, optimizer_flow, optimizer_adapter, train_dataloader = accelerator.prepare(
        model, adapter_model, optimizer_flow, optimizer_adapter, train_dataloader
    )  

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name=args.project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}", "dir": f"{args.wandb_dir}"}},
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # initialize inverse problems
    INVERSE_PROBLEMS = initialize_inverse_problems(device=device)

    # Create sampling noise:
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, gt_labels = [], [], []
    if sample_batch_size // local_batch_size == 0:
        batch = next(iter(train_dataloader))
        gt_raw_images.append(batch[0][:sample_batch_size])
        gt_xs.append(vae.encode(batch[0][:sample_batch_size]).latent_dist.sample().mul_(0.18215))
        gt_labels.append(batch[1][:sample_batch_size])
        sample_batch_size = gt_raw_images[0].shape[0]
    else:
        for i in range(sample_batch_size // local_batch_size):
            batch = next(iter(train_dataloader))
            gt_raw_images.append(batch[0])
            gt_xs.append(vae.encode(batch[0]).latent_dist.sample().mul_(0.18215))
            gt_labels.append(batch[1])
        sample_batch_size = local_batch_size * len(gt_raw_images)
    gt_raw_images = torch.cat(gt_raw_images, dim=0)
    assert sample_batch_size == gt_raw_images.shape[0]
    gt_xs = torch.cat(gt_xs, dim=0)
    gt_labels = torch.cat(gt_labels, dim=0)
    assert gt_raw_images.shape[-1] == args.resolution
    gt_raw_images, gt_xs, gt_labels = gt_raw_images.to(device)[:sample_batch_size], gt_xs.to(device)[:sample_batch_size], gt_labels.to(device)[:sample_batch_size]
    print("range of gt_raw_images:", gt_raw_images.min().item(), gt_raw_images.max().item())
    ys = torch.randint(0, 1000, (sample_batch_size,), device=device)
    x1 = torch.randn((sample_batch_size, 4, latent_size, latent_size), device=device)
    inv_classes = torch.randint(0, args.num_inv_probs, (sample_batch_size,), device=device)
    inv_classes_list = inv_classes.tolist()
    operators = [INVERSE_PROBLEMS[idx] for idx in inv_classes_list]
    for op in operators:
        op.update(seed=42)
    y_list_tmp = [operators[i](gt_raw_images[i:i+1]) for i in range(sample_batch_size)]
    y_list_fwd = [y_list_tmp[i] + 0.05 * torch.randn_like(y_list_tmp[i]) for i in range(sample_batch_size)]
    y_list = [operators[i].transpose(y_list_fwd[i]) for i in range(sample_batch_size)]
    measurements = torch.cat(y_list, dim=0)
    print("range of measurements:", measurements.min().item(), measurements.max().item())
    print("shape of measurements:", measurements.shape)
    sampling_kwargs = dict(
        num_steps=4,
        cfg_scale=1.0,
        g_min=0.0,
        g_max=1.0,
        shift=1.0,
        path_type=args.path_type,
        reverse=args.pretrained_sigma,
    )
    sampling_kwargs_onestep = sampling_kwargs.copy()
    sampling_kwargs_onestep['num_steps'] = 1
    for epoch in range(args.epochs):
        model.train()
        adapter_model.train()
        for img, label in train_dataloader:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.no_grad():
                latent = vae.encode(img).latent_dist.sample().mul_(0.18215)
            with accelerator.accumulate(model, adapter_model):
                with accelerator.autocast():
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_adapter_model = accelerator.unwrap_model(adapter_model)
                    loss, loss_dict = loss_fn(model, unwrapped_model, ema, 
                                            adapter_model, unwrapped_adapter_model, ema_adapter, 
                                            img, latent, label, model_t=model_t, 
                                            decode_func=decode_func, inv_probs=INVERSE_PROBLEMS, data_loss_learn_fm=args.data_loss_learn_fm,
                                            data_loss_weight=args.data_loss_weight, FM_loss_weight=args.FM_loss_weight, KL_loss_weight=args.KL_loss_weight,
                                            meas_noise=args.measurement_noise, reverse=args.pretrained_sigma, train_adapter_only=args.train_adapter_only)
                loss = loss.mean()
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm_flow = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    grad_norm_adapter = accelerator.clip_grad_norm_(adapter_model.parameters(), args.max_grad_norm)
                optimizer_flow.step()
                optimizer_adapter.step()
                optimizer_flow.zero_grad(set_to_none=True)
                optimizer_adapter.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model, decay=args.ema_decay, only_trainable=True, update_buffers=True)
                    update_ema(ema_adapter, adapter_model, decay=args.ema_decay, only_trainable=True, update_buffers=True)

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_adapter_model = accelerator.unwrap_model(adapter_model)
                    checkpoint = {
                        "model": unwrapped_model.state_dict(),
                        "adapter_model": unwrapped_adapter_model.state_dict(),
                        "ema": ema.state_dict(),
                        "ema_adapter": ema_adapter.state_dict(),
                        "opt_flow": optimizer_flow.state_dict(),
                        "opt_adapter": optimizer_adapter.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == args.resume_step+10 or global_step == 10 or global_step == 100 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                with torch.no_grad():
                    with accelerator.autocast():
                        samples = euler_sampler(ema, x1, ys, **sampling_kwargs).to(torch.float32)
                        samples_onestep = euler_sampler(ema, x1, ys, **sampling_kwargs_onestep).to(torch.float32)
                        adapter_samples = adapter_euler_sampler(ema, gt_labels, x1, ema_adapter, measurements, inv_classes, **sampling_kwargs).to(torch.float32)
                        adapter_samples_onestep = adapter_euler_sampler(ema, gt_labels, x1, ema_adapter, measurements, inv_classes, **sampling_kwargs_onestep).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    samples_onestep = vae.decode((samples_onestep -  latents_bias) / latents_scale).sample
                    samples_onestep = (samples_onestep + 1) / 2.
                    adapter_samples = vae.decode((adapter_samples -  latents_bias) / latents_scale).sample
                    adapter_samples = (adapter_samples + 1) / 2.
                    adapter_samples_onestep = vae.decode((adapter_samples_onestep -  latents_bias) / latents_scale).sample
                    adapter_samples_onestep = (adapter_samples_onestep + 1) / 2.
                    gt_samples_decoded = (vae.decode((gt_xs -  latents_bias) / latents_scale).sample + 1) / 2.
                    gt_samples = (gt_raw_images + 1) / 2.
                    measurement_images = (measurements + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                adapter_samples = accelerator.gather(adapter_samples.to(torch.float32))
                gt_samples_decoded = accelerator.gather(gt_samples_decoded.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                samples_onestep = accelerator.gather(samples_onestep.to(torch.float32))
                adapter_samples_onestep = accelerator.gather(adapter_samples_onestep.to(torch.float32))
                measurement_images = accelerator.gather(measurement_images.to(torch.float32))
                accelerator.log({
                    "samples": wandb.Image(array2grid(out_samples)),
                    "adapter_samples": wandb.Image(array2grid(adapter_samples)),
                    "gt_samples_decoded": wandb.Image(array2grid(gt_samples_decoded)), # for sanity check of latents being aligned
                    "gt_samples": wandb.Image(array2grid(gt_samples)),
                    "samples_onestep": wandb.Image(array2grid(samples_onestep)),
                    "adapter_samples_onestep": wandb.Image(array2grid(adapter_samples_onestep)),
                    "measurement_images": wandb.Image(array2grid(measurement_images))
                })
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
            }
            if accelerator.sync_gradients:
                    logs["grad_norm_flow"] = accelerator.gather(grad_norm_flow).mean().detach().item()
                    logs["grad_norm_adapter"] = accelerator.gather(grad_norm_adapter).mean().detach().item()
            for key, value in loss_dict.items():
                logs[key] = accelerator.gather(value).mean().detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()
    adapter_model.eval()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="./exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--project-name", type=str, required=True)
    parser.add_argument("--wandb-dir", type=str, default="./wandb/")

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--dmf-depth", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # adapter model
    parser.add_argument("--adapter-model", type=str, default="base")
    parser.add_argument("--num-inv-probs", type=int, default=6)
    parser.add_argument("--qk-norm-adapter",  action=argparse.BooleanOptionalAction, default=False)

    # Fine-tuning
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--pretrained-sigma", action=argparse.BooleanOptionalAction, default=False, help="SiT models from original codebase have pretrained sigma head that we should ignore and it also affect the time flow to be from 0 to 1.")

    # Fine-tuning of adapter (if any)
    parser.add_argument("--adapter-pretrained", type=str, default=None)
    parser.add_argument("--data-loss-learn-fm", type=str, default="ema_path", choices=["ema_path", "freeze", "learn"])
    parser.add_argument("--data-loss-weight", type=float, default=1.0, help="Weight for the data consistency loss.")
    parser.add_argument("--FM-loss-weight", type=float, default=1.0, help="Weight for the flow map loss.")
    parser.add_argument("--KL-loss-weight", type=float, default=1.0, help="Weight for the KL loss.")
    parser.add_argument("--measurement-noise", type=float, default=0.05, help="Standard deviation of the measurement noise added during training.")

    #training loss
    parser.add_argument("--g-type", type=str, default="default", choices=["default", "mg", "distill"])
    parser.add_argument("--P-mean", type=float, default=0.0, help="Mean of the P distribution.")
    parser.add_argument("--P-std", type=float, default=1.0, help="Standard deviation of the P distribution.")
    parser.add_argument("--P-mean-t", type=float, default=0.4, help="Mean of the P_t distribution.")
    parser.add_argument("--P-std-t", type=float, default=1.0, help="Standard deviation of the P_t distribution.")
    parser.add_argument("--P-mean-r", type=float, default=-1.2, help="Mean of the P_r distribution.")
    parser.add_argument("--P-std-r", type=float, default=1.0, help="Standard deviation of the P_r distribution.")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--attn-func", type=str, default="base", choices=["base", "torch_sdpa", "fa2", "fa3"])
    parser.add_argument("--omega", type=float, default=0.5, help="Scale for the mean flow loss.")
    parser.add_argument("--g-min", type=float, default=0.0, help="Minimum value for the cfg index.")
    parser.add_argument("--g-max", type=float, default=1.0, help="Maximum value for the cfg index.")

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    ## adam-beta2 is set to 0.95 in the original DiT paper
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="Exponential moving average decay.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-adapter-only", action=argparse.BooleanOptionalAction, default=False, help="Whether to train only the adapter and keep the main model frozen.")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
