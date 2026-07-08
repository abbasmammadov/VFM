#!/bin/bash
# Multi-node joint training via SLURM + Accelerate.
# Adjust the SBATCH resources, module loads, and the paths below to your cluster.
#SBATCH --job-name=vfm-train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=100GB
#SBATCH --time=24:00:00

set -euo pipefail

# --- environment (edit for your cluster) ---
# source "$HOME/miniforge3/bin/activate" && conda activate vfm
# module load cuda/12.6

# --- paths (edit or export before submitting) ---
: "${DATA_DIR:?Set DATA_DIR to your ImageNet train directory}"
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the pretrained DMF backbone}"
OUTPUT_DIR="${OUTPUT_DIR:-./exps}"
EXP_NAME="${EXP_NAME:-vfm-dmft-b2-256}"
WANDB_PROJECT="${WANDB_PROJECT:-vfm}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
GPUS_PER_NODE=4
NUM_PROCESSES=$((SLURM_NNODES * GPUS_PER_NODE))

echo "MASTER_ADDR=$MASTER_ADDR  NNODES=$SLURM_NNODES  NUM_PROCESSES=$NUM_PROCESSES"

srun --ntasks-per-node=1 bash -c '
accelerate launch \
  --num_machines '"$SLURM_NNODES"' \
  --num_processes '"$NUM_PROCESSES"' \
  --multi_gpu \
  --main_process_ip '"$MASTER_ADDR"' \
  --main_process_port '"$MASTER_PORT"' \
  --machine_rank $SLURM_NODEID \
  --mixed_precision bf16 \
  train_adapter.py \
  --project-name "'"$WANDB_PROJECT"'" \
  --exp-name "'"$EXP_NAME"'" \
  --output-dir "'"$OUTPUT_DIR"'" \
  --data-dir "'"$DATA_DIR"'" \
  --pretrained "'"$PRETRAINED_CKPT"'" \
  --pretrained-sigma \
  --model DMFT-B/2 \
  --dmf-depth 8 \
  --adapter-model base \
  --num-inv-probs 6 \
  --resolution 256 \
  --batch-size 256 \
  --attn-func fa3 \
  --g-type mg \
  --omega 0.5 \
  --P-mean 0.0 --P-mean-t 0.4 --P-mean-r -1.2 \
  --data-loss-weight 1.0 --FM-loss-weight 1.0 --KL-loss-weight 1.0 \
  --seed 0 --mixed-precision bf16 \
  --max-train-steps 4000000 --checkpointing-steps 50000 --sampling-steps 1000
'
