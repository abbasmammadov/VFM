#!/usr/bin/env bash
# Minimal 2D toy demo of Variational Flow Maps on the checkerboard distribution.
# Trains the noise adapter jointly with a MeanFlow map, starting from the
# provided pretrained flow-matching checkpoint. Runs on a single GPU (or CPU).
set -euo pipefail

cd "$(dirname "$0")/../checkerboard"

python train_vfm.py \
  --sigma 0.1 \
  --tau 1.0 \
  --alpha 0.5 \
  --num-iter 20000 \
  --K 4 \
  --load-fm checkpoints/fm_model.pt \
  "$@"
