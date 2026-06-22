#!/bin/bash
#SBATCH --job-name=stag-hunt-llm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=31G
#SBATCH --partition=rtx2080
#SBATCH --output=/scratch.hpc/matteo.preda/logs/job_%j.out
#SBATCH --chdir=/scratch.hpc/matteo.preda/Stag-Hunt-LLM
#SBATCH --gres=gpu:1

# =============================================================================
# run.sh — SLURM job script for LLM+REINFORCE training on Stag Hunt
# =============================================================================

set -e

PROJECT_DIR="/scratch.hpc/matteo.preda/Stag-Hunt-LLM"
VENV_DIR="/scratch.hpc/matteo.preda/rl"
CACHE_DIR="/scratch.hpc/matteo.preda/hf_cache"

# --- Job config ---
MODE="train"
PROMPT_TYPE="4"
CHECKPOINT_A="$PROJECT_DIR/checkpoints/agent_A_latest.pt"
CHECKPOINT_B="$PROJECT_DIR/checkpoints/agent_B_latest.pt"

# ---------------------------------------------------------------------------
# Redirect ALL caches to scratch BEFORE activating venv or running Python.
# Setting these here in the shell means every child process inherits them,
# so huggingface_hub never touches /home/students at all.
# ---------------------------------------------------------------------------

export HF_HOME="$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TORCH_HOME="/scratch.hpc/matteo.preda/torch_cache"
export PIP_CACHE_DIR="/scratch.hpc/matteo.preda/pip_cache"
export TMPDIR="/scratch.hpc/matteo.preda/tmp"

# Create cache dirs if they don't exist
mkdir -p "$CACHE_DIR"
mkdir -p "/scratch.hpc/matteo.preda/torch_cache"
mkdir -p "/scratch.hpc/matteo.preda/pip_cache"
mkdir -p "/scratch.hpc/matteo.preda/tmp"

# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

echo "============================================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $SLURMD_NODENAME"
echo "  Start time:  $(date)"
echo "  Project dir: $PROJECT_DIR"
echo "  HF cache:    $HF_HOME"
echo "  Mode:        $MODE"
echo "  Prompt type: $PROMPT_TYPE"
echo "============================================================"

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "==> Python: $(which python3) — $(python3 --version)"

# Show GPU info
echo "==> GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

cd "$PROJECT_DIR"

if [ "$MODE" = "train" ]; then
    echo "==> Starting training (prompt_type=$PROMPT_TYPE)"
    python3 main.py --mode train --prompt_type "$PROMPT_TYPE"

elif [ "$MODE" = "eval" ]; then
    echo "==> Starting evaluation"
    python3 main.py --mode eval \
        --checkpoint_a "$CHECKPOINT_A" \
        --checkpoint_b "$CHECKPOINT_B"
fi

echo ""
echo "============================================================"
echo "  Done. End time: $(date)"
echo "============================================================"