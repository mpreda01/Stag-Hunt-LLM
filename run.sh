#!/bin/bash
#SBATCH --job-name=stag-hunt-llm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=31G
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/matteo.preda/logs/job_%j.out
#SBATCH --chdir=/scratch.hpc/matteo.preda
#SBATCH --gres=gpu:1

# =============================================================================
# run.sh — SLURM job script for LLM+REINFORCE training on Stag Hunt
#
# Submit with:
#   sbatch run.sh                        # train, two-shot (default)
#   sbatch run.sh --prompt_type 2        # train, zero-shot
#   sbatch run.sh --prompt_type 3        # train, one-shot
#
# Eval (pass extra args after --)  — not supported with sbatch arg forwarding,
# edit MODE and ARGS below instead.
# =============================================================================

set -e

PROJECT_DIR="/scratch.hpc/matteo.preda"
VENV_DIR="$PROJECT_DIR/rl"

# --- Job config ---
MODE="train"            # "train" or "eval"
PROMPT_TYPE="4"         # "2"=zero-shot "3"=one-shot "4"=two-shot
CHECKPOINT_A="$PROJECT_DIR/checkpoints/agent_A_latest.pt"
CHECKPOINT_B="$PROJECT_DIR/checkpoints/agent_B_latest.pt"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

echo "============================================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $SLURMD_NODENAME"
echo "  Start time:  $(date)"
echo "  Project dir: $PROJECT_DIR"
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
