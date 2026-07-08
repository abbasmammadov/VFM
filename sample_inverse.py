"""
One/few-step conditional sampling for inverse problems with a trained VFM.

Given an observation y, the noise adapter q_phi(z|y) proposes an
observation-dependent initial noise, which the flow map f_theta maps to a data
sample in one (or a few) steps. Running with a batch size > 1 draws multiple
posterior samples for the same observation.

Example:
    python sample_inverse.py \
        --ckpt exps/vfm-dmft-b2-256/checkpoints/0100000.pt \
        --image path/to/image.png \
        --problem super_resolution \
        --class-label 279 \
        --num-samples 8 \
        --out samples_inverse
"""
import os
import argparse

import numpy as np
import torch
from PIL import Image

from diffusers.models import AutoencoderKL

from models.dmft import DMFT_models
from models.adapter import create_adapter
from samplers import adapter_euler_sampler
from inverse_problems.measurements import (
    Identity, InpaintingRandom, InpaintingBox,
    SuperResolution, GaussianBlur, MotionBlur,
)

# Inverse-problem registry. The index must match the `--num-inv-probs` ordering
# the checkpoint was trained with (default: 6 problems, indices 0-5).
PROBLEMS = {
    "identity": (0, Identity),
    "inpaint_random": (1, InpaintingRandom),
    "inpaint_box": (2, InpaintingBox),
    "super_resolution": (3, SuperResolution),
    "gaussian_blur": (4, GaussianBlur),
    "motion_blur": (5, MotionBlur),
}


def center_crop(pil_image, size):
    """Center-crop to a square and resize to `size` (ADM-style, matches training)."""
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - size) // 2
    cx = (arr.shape[1] - size) // 2
    return arr[cy: cy + size, cx: cx + size]


def load_image(path, resolution, device):
    """Load an image into a [-1, 1] tensor of shape (1, 3, resolution, resolution)."""
    arr = center_crop(Image.open(path).convert("RGB"), resolution)
    x = torch.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1.0
    return x.unsqueeze(0).to(device)


def to_uint8(img):
    """(B, 3, H, W) in [0, 1] -> (B, H, W, 3) uint8 numpy."""
    img = torch.clamp(255.0 * img, 0, 255)
    return img.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    latent_size = args.resolution // 8
    block_kwargs = {"attn_func": "torch_sdpa", "qk_norm": args.qk_norm}

    # --- flow map (DMFT backbone) ---
    model = DMFT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg_embedding=True,
        dmf_depth=args.dmf_depth,
        pretrained_sigma=True,
        **block_kwargs,
    )
    # --- noise adapter ---
    adapter = create_adapter(
        variant=args.adapter_model,
        num_inverse_problems=args.num_inv_probs,
        image_size=args.resolution,
        latent_size=latent_size,
        latent_channels=4,
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_sd = ckpt["ema"] if "ema" in ckpt else ckpt.get("model", ckpt)
    adapter_sd = ckpt["ema_adapter"] if "ema_adapter" in ckpt else ckpt["adapter_model"]
    print("flow map:", model.load_state_dict(model_sd, strict=False))
    print("adapter :", adapter.load_state_dict(adapter_sd, strict=False))
    model = model.to(device=device, dtype=dtype).eval()
    adapter = adapter.to(device=device, dtype=dtype).eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)

    if args.problem not in PROBLEMS:
        raise ValueError(f"Unknown problem '{args.problem}'. Choose from {list(PROBLEMS)}")
    inv_idx, op_cls = PROBLEMS[args.problem]
    operator = op_cls(device=device)
    operator.update(seed=args.seed)

    # --- build the observation y from the clean image ---
    x0 = load_image(args.image, args.resolution, device)
    y_clean = operator(x0)
    y_fwd = y_clean + args.meas_noise * torch.randn_like(y_clean)
    y = operator.transpose(y_fwd)  # back to (1, 3, res, res) image space

    n = args.num_samples
    measurements = y.repeat(n, 1, 1, 1).to(dtype)
    inv_classes = torch.full((n,), inv_idx, device=device, dtype=torch.long)
    if args.class_label is None:
        labels = torch.randint(0, args.num_classes, (n,), device=device)
    else:
        labels = torch.full((n,), args.class_label, device=device, dtype=torch.long)
    latents = torch.randn(n, 4, latent_size, latent_size, device=device, dtype=dtype)

    with torch.no_grad():
        samples = adapter_euler_sampler(
            model, labels, latents, adapter, measurements, inv_classes,
            num_steps=args.num_steps, cfg_scale=args.cfg_scale,
            g_min=0.0, g_max=1.0, shift=1.0, reverse=True,
        ).to(torch.float32)
        images = vae.decode(samples / latents_scale).sample
        images = (images + 1) / 2.0

    os.makedirs(args.out, exist_ok=True)
    # Save the observation for reference, then each posterior sample.
    Image.fromarray(to_uint8((measurements.float() + 1) / 2.0)[0]).save(os.path.join(args.out, "observation.png"))
    for i, img in enumerate(to_uint8(images)):
        Image.fromarray(img).save(os.path.join(args.out, f"sample_{i:03d}.png"))
    print(f"Saved observation + {n} conditional sample(s) to {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VFM conditional sampling for inverse problems.")
    parser.add_argument("--ckpt", type=str, required=True, help="Trained VFM checkpoint (.pt).")
    parser.add_argument("--image", type=str, required=True, help="Clean input image; the observation is synthesised from it.")
    parser.add_argument("--problem", type=str, default="super_resolution", choices=list(PROBLEMS),
                        help="Inverse problem to solve.")
    parser.add_argument("--out", type=str, default="samples_inverse", help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of posterior samples for the observation.")
    parser.add_argument("--num-steps", type=int, default=1, help="Flow-map steps (1 = one-step).")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--class-label", type=int, default=None, help="ImageNet class id to condition on (default: random).")
    parser.add_argument("--meas-noise", type=float, default=0.05, help="Observation noise std (matches training).")
    # model / checkpoint config (defaults match the released DMFT-B/2 checkpoint)
    parser.add_argument("--model", type=str, default="DMFT-B/2")
    parser.add_argument("--adapter-model", type=str, default="base")
    parser.add_argument("--dmf-depth", type=int, default=8)
    parser.add_argument("--num-inv-probs", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
