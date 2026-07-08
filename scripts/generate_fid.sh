#!/usr/bin/env bash
# Class-conditional ImageNet-256 sampling + FID evaluation for the flow map.
# Generates 50k samples with the (few-step) Euler flow-map sampler and computes
# FID against a reference statistics file.
#
#   CKPT     : trained VFM checkpoint (.pt); the EMA weights are used if present
#   REF_PKL  : reference statistics for FID (see preprocessing/README.md)
set -euo pipefail

CKPT="${CKPT:?Set CKPT to a trained checkpoint .pt}"
REF_PKL="${REF_PKL:?Set REF_PKL to the FID reference statistics .pkl}"
SAMPLE_DIR="${SAMPLE_DIR:-./samples}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_STEPS="${NUM_STEPS:-1}"          # 1 = one-step generation; try 2/4 for few-step
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"

# Generate samples. --reverse matches backbones trained with `pretrained_sigma`
# (t = 0 -> data, t = 1 -> noise), as used by the released checkpoint.
torchrun \
  --nnodes 1 \
  --nproc_per_node "${NUM_GPUS}" \
  --master-port 29502 \
  generate.py \
  --ckpt "${CKPT}" \
  --sample-dir "${SAMPLE_DIR}" \
  --num-fid-samples "${NUM_FID_SAMPLES}" \
  --per-proc-batch-size 128 \
  --model DMFT-B/2 \
  --dmf-depth 8 \
  --resolution 256 \
  --mode euler \
  --num-steps "${NUM_STEPS}" \
  --global-seed 0 \
  --reverse

# Compute FID (and related metrics) against the reference statistics.
# calculate_metrics.py must be run from inside preprocessing/ (it imports the
# local dataset module). Resolve the sample dir / ref to absolute paths first.
ABS_SAMPLE_DIR="$(cd "${SAMPLE_DIR}" && pwd)"
case "${REF_PKL}" in
  http*|/*) ABS_REF="${REF_PKL}" ;;      # URL or absolute path: use as-is
  *)        ABS_REF="$(cd "$(dirname "${REF_PKL}")" && pwd)/$(basename "${REF_PKL}")" ;;
esac
cd preprocessing
python calculate_metrics.py calc \
  --images="${ABS_SAMPLE_DIR}" \
  --ref="${ABS_REF}" \
  --num "${NUM_FID_SAMPLES}"
