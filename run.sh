#!/bin/bash
#SBATCH --job-name=stag-hunt-lora-ppo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx2080
#SBATCH --output=/scratch.hpc/matteo.preda/logs/job_%j.out
#SBATCH --chdir=/scratch.hpc/matteo.preda/Stag-Hunt-LLM
#SBATCH --gres=gpu:1



set -e

PROJECT_DIR="/scratch.hpc/matteo.preda/Stag-Hunt-LLM"
VENV_DIR="/scratch.hpc/matteo.preda/rl"
CACHE_DIR="/scratch.hpc/matteo.preda/hf_cache"

MODE="train"
CHECKPOINT="$PROJECT_DIR/checkpoints/policy_latest"

# Redirect all caches to scratch
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
         "/scratch.hpc/matteo.preda/logs"

echo "============================================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $SLURMD_NODENAME"
echo "  Start time:  $(date)"
echo "  Project dir: $PROJECT_DIR"
echo "  Mode:        $MODE"
echo "  Prompt type: $PROMPT_TYPE"
echo "============================================================"

source "$VENV_DIR/bin/activate"
echo "==> Python: $(which python3) — $(python3 --version)"

export WANDB_API_KEY="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"

echo "==> GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

cd "$PROJECT_DIR"

if [ "$MODE" = "train" ]; then
    echo "==> Starting training"
    python3 main.py --mode train 

elif [ "$MODE" = "eval" ]; then
    echo "==> Starting evaluation"
    python3 main.py --mode eval --checkpoint "$CHECKPOINT"
fi

echo ""
echo "============================================================"
echo "  Done. End time: $(date)"
echo "============================================================"
