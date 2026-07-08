#!/usr/bin/env bash
# Joint training of the flow map + noise adapter on ImageNet-256 for amortized
# inverse problems. This reproduces the configuration of the main VFM checkpoint
# (DMFT-B/2 backbone, 6 inverse problems, MeanFlow guidance).
#
# Fill in the paths below (or export them in your environment) before running.
#   DATA_DIR        : ImageNet train folder (ImageFolder layout: <DATA_DIR>/<class>/*.JPEG)
#   PRETRAINED_CKPT : pretrained flow-map / MeanFlow (DMF) backbone checkpoint
#   OUTPUT_DIR      : where experiment folders + checkpoints are written
set -euo pipefail

DATA_DIR="${DATA_DIR:?Set DATA_DIR to your ImageNet train directory}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the pretrained DMF backbone}"
OUTPUT_DIR="${OUTPUT_DIR:-./exps}"
EXP_NAME="${EXP_NAME:-vfm-dmft-b2-256}"
WANDB_PROJECT="${WANDB_PROJECT:-vfm}"

# Attention backend: "fa3" (Hopper) / "fa2" / "torch_sdpa" (no flash-attn install).
ATTN_FUNC="${ATTN_FUNC:-fa2}"
# Number of local GPUs (single node). For multi-node, use scripts/slurm/train.slurm.
NUM_GPUS="${NUM_GPUS:-8}"

accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_GPUS}" \
  --mixed_precision bf16 \
  train_adapter.py \
  --project-name "${WANDB_PROJECT}" \
  --exp-name "${EXP_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --pretrained "${PRETRAINED_CKPT}" \
  --pretrained-sigma \
  --model DMFT-B/2 \
  --dmf-depth 8 \
  --adapter-model base \
  --num-inv-probs 6 \
  --resolution 256 \
  --batch-size 256 \
  --attn-func "${ATTN_FUNC}" \
  --g-type mg \
  --omega 0.5 \
  --g-min 0.0 \
  --g-max 1.0 \
  --P-mean 0.0 \
  --P-mean-t 0.4 \
  --P-mean-r -1.2 \
  --data-loss-weight 1.0 \
  --FM-loss-weight 1.0 \
  --KL-loss-weight 1.0 \
  --seed 0 \
  --mixed-precision bf16 \
  --max-train-steps 4000000 \
  --checkpointing-steps 50000 \
  --sampling-steps 1000
