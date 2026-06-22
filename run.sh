#!/bin/bash
# =============================================================================
# run.sh — Setup and launch LLM+REINFORCE training on SSH cluster
#
# Usage:
#   bash run.sh                          # train with default two-shot prompt
#   bash run.sh --prompt_type 2          # zero-shot
#   bash run.sh --mode eval \
#     --checkpoint_a checkpoints/agent_A_latest.pt \
#     --checkpoint_b checkpoints/agent_B_latest.pt
#
# All extra arguments are forwarded directly to main.py.
# =============================================================================

set -e  # exit immediately on any error

# ---------------------------------------------------------------------------
# Config — edit these for your cluster
# ---------------------------------------------------------------------------

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
REPO_URL="https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt.git"
STAG_HUNT_DIR="$PROJECT_DIR/Gymnasium-Stag-Hunt"
PYTHON="python3"

# HuggingFace model — downloaded once, cached in ~/.cache/huggingface
HF_MODEL="Qwen/Qwen3-4B"

# ---------------------------------------------------------------------------
# 1. Create virtual environment (skip if already exists)
# ---------------------------------------------------------------------------

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment at $VENV_DIR"
    $PYTHON -m venv "$VENV_DIR"
else
    echo "==> Virtual environment already exists, skipping creation"
fi

source "$VENV_DIR/bin/activate"
echo "==> Python: $(which python) — $(python --version)"

# ---------------------------------------------------------------------------
# 2. Upgrade pip
# ---------------------------------------------------------------------------

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA (cluster typically has CUDA 12.1)
#    Adjust the --index-url if your cluster uses a different CUDA version.
#    Check with: nvcc --version
# ---------------------------------------------------------------------------

echo "==> Installing PyTorch (CUDA 12.1)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet


# ---------------------------------------------------------------------------
# 5. Clone and install Gymnasium-Stag-Hunt (skip if already installed)
# ---------------------------------------------------------------------------

if ! python -c "import gymnasium_stag_hunt" 2>/dev/null; then
    echo "==> Cloning Gymnasium-Stag-Hunt"
    if [ ! -d "$STAG_HUNT_DIR" ]; then
        git clone "$REPO_URL" "$STAG_HUNT_DIR"
    fi
    echo "==> Installing Gymnasium-Stag-Hunt"
    pip install -e "$STAG_HUNT_DIR" --quiet
else
    echo "==> gymnasium_stag_hunt already installed, skipping"
fi

# ---------------------------------------------------------------------------
# 6. Pre-download HuggingFace model (optional but avoids timeout mid-training)
# ---------------------------------------------------------------------------

echo "==> Pre-downloading HuggingFace model: $HF_MODEL"
python - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
print(f"Downloading tokenizer...")
AutoTokenizer.from_pretrained("$HF_MODEL")
print(f"Downloading model config (weights downloaded at first training run)...")
from transformers import AutoConfig
AutoConfig.from_pretrained("$HF_MODEL")
print("Done.")
EOF

# ---------------------------------------------------------------------------
# 7. Create output directories
# ---------------------------------------------------------------------------

mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/results"

# ---------------------------------------------------------------------------
# 8. Launch main.py — all arguments forwarded from run.sh invocation
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Launching main.py with args: $@"
echo "  Project dir: $PROJECT_DIR"
echo "  Start time:  $(date)"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
python main.py "$@"

echo ""
echo "============================================================"
echo "  Done. End time: $(date)"
echo "============================================================"
