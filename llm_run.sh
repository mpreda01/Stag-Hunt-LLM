#!/bin/bash
# =============================================================================
#  llm_run.sh  —  SLURM job script for Stag Hunt GRPO LLM training
#
#  GPU CHOICE NOTES
#  ----------------
#  RTX 2080 (11 GB):  Can run Qwen2.5-3B in 4-bit NF4 for inference, but
#                     backprop through LoRA + activation storage pushes peak
#                     VRAM to ~12-14 GB.  Use ONLY if you set a very small
#                     batch (rollouts_per_ep=1, max_new_tokens=128) and are
#                     prepared for OOM crashes.  Queue is shorter though.
#
#  L40 (48 GB):       Comfortable.  Fits 4-bit Qwen2.5-3B + full LoRA grad
#                     with rollouts_per_ep=4 and max_new_tokens=256 to spare.
#                     Recommended for serious training runs.
#                     Queue is longer — submit with enough --time headroom.
#
#  To switch between them, change the --partition and --gres lines below.
#  The script auto-detects VRAM and adjusts hyperparameters accordingly.
# =============================================================================

#SBATCH --job-name=staghunt-grpo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# ---- GPU PARTITION: uncomment ONE block ----

# # Option A: L40 (48 GB)  — recommended
# #SBATCH --partition=l40
# #SBATCH --gres=gpu:1

# Option B: RTX 2080 (11 GB)  — only for quick smoke-tests
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1

# ---- Output ----
#SBATCH --output=/scratch.hpc/matteo.preda/logs/grpo_%j.out
#SBATCH --error=/scratch.hpc/matteo.preda/logs/grpo_%j.err
#SBATCH --chdir=/scratch.hpc/matteo.preda/Stag-Hunt-LLM

# =============================================================================
set -e

PROJECT_DIR="/scratch.hpc/matteo.preda/Stag-Hunt-LLM"
VENV_DIR="/scratch.hpc/matteo.preda/rl"
CACHE_DIR="/scratch.hpc/matteo.preda/hf_cache"
CKPT_DIR="$PROJECT_DIR/checkpoints_llm"

# ---- Redirect all caches to scratch (avoids home-quota issues) ----
export HF_HOME="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TORCH_HOME="/scratch.hpc/matteo.preda/torch_cache"
export PIP_CACHE_DIR="/scratch.hpc/matteo.preda/pip_cache"
export TMPDIR="/scratch.hpc/matteo.preda/tmp"

mkdir -p "$CACHE_DIR" \
         "/scratch.hpc/matteo.preda/torch_cache" \
         "/scratch.hpc/matteo.preda/pip_cache" \
         "/scratch.hpc/matteo.preda/tmp" \
         "/scratch.hpc/matteo.preda/logs" \
         "$CKPT_DIR"

echo "============================================================"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Start    : $(date)"
echo "  Project  : $PROJECT_DIR"
echo "============================================================"

# ---- Activate virtualenv ----
source "$VENV_DIR/bin/activate"
echo "==> Python : $(which python3) — $(python3 --version)"

# ---- Install / verify dependencies ----
python3 -c "import peft"          2>/dev/null || pip install peft --quiet
python3 -c "import bitsandbytes"  2>/dev/null || pip install bitsandbytes --quiet
python3 -c "import transformers"  2>/dev/null || pip install transformers --quiet

echo "==> GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# =============================================================================
# Auto-select hyperparameters based on available VRAM
# =============================================================================
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "==> Detected VRAM: ${VRAM_MB} MB"

if [ "$VRAM_MB" -ge 40000 ]; then
    # L40 or better — use comfortable settings
    EPOCHS=300
    ROLLOUTS_PER_EP=4
    MAX_NEW_TOKENS=256
    LR=5e-5
    KL_COEFF=0.02
    GRID_SIZE=5
    echo "==> Using L40 config  (epochs=$EPOCHS, rollouts=$ROLLOUTS_PER_EP)"
else
    # RTX 2080 or similar — reduce memory pressure
    EPOCHS=100
    ROLLOUTS_PER_EP=1
    MAX_NEW_TOKENS=128
    LR=1e-4
    KL_COEFF=0.05
    GRID_SIZE=5
    echo "==> Using RTX 2080 config  (epochs=$EPOCHS, rollouts=$ROLLOUTS_PER_EP)"
    echo "    WARNING: training on RTX 2080 may hit OOM — monitor with nvidia-smi"
fi

# =============================================================================
# Determine run mode: resume if checkpoint exists, else fresh start
# =============================================================================
LATEST_CKPT="$CKPT_DIR/lora_latest"
if [ -d "$LATEST_CKPT" ]; then
    echo "==> Resuming from checkpoint: $LATEST_CKPT"
    CKPT_ARG="--checkpoint $LATEST_CKPT"
else
    echo "==> Starting fresh training run"
    CKPT_ARG=""
fi

# =============================================================================
# Training
# =============================================================================
cd "$PROJECT_DIR"

echo ""
echo "==> Launching GRPO training …"
python3 main_q.py \
    --mode      train \
    --epochs    "$EPOCHS" \
    --lr        "$LR" \
    --kl_coeff  "$KL_COEFF" \
    --grid_size "$GRID_SIZE" \
    --device    cuda \
    $CKPT_ARG

# =============================================================================
# Quick evaluation after training
# =============================================================================
echo ""
echo "==> Running post-training evaluation (20 episodes) …"
python3 main_q.py \
    --mode       eval \
    --checkpoint "$CKPT_DIR/lora_final" \
    --eval_ep    20 \
    --grid_size  "$GRID_SIZE" \
    --device     cuda

echo ""
echo "============================================================"
echo "  Done.  End time: $(date)"
echo "============================================================"
