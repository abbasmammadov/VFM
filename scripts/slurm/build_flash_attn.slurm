#!/bin/bash
# Build FlashAttention-3 (Hopper) from source. FA3 gives the best throughput on
# H100-class GPUs; FA2 works elsewhere, and the code also falls back to PyTorch
# SDPA (--attn-func torch_sdpa) if neither is installed.
#SBATCH --job-name=build_flash_attn
#SBATCH --output=build_flash_attn_%j.out
#SBATCH --error=build_flash_attn_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

set -euo pipefail

# --- environment (edit for your cluster) ---
# source ~/miniforge3/bin/activate && conda activate vfm
# module load gcc-native/12.3
# module load cuda/12.6

python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"

# Path to a checkout of https://github.com/Dao-AILab/flash-attention
FLASH_ATTN_DIR="${FLASH_ATTN_DIR:?Set FLASH_ATTN_DIR to your flash-attention checkout}"

export MAX_JOBS="${MAX_JOBS:-4}"

# FlashAttention-3 (Hopper). For FA2 instead, run `pip install flash-attn` and
# skip this build.
cd "${FLASH_ATTN_DIR}/hopper"
python setup.py install
echo "FlashAttention build completed at $(date)"
