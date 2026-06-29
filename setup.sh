#!/bin/bash
# =============================================================================
# setup.sh — Run ONCE manually on the cluster to create the environment
#
# Usage (from login node):
#   cd /scratch.hpc/matteo.preda
#   bash Stag-Hunt-LLM/setup.sh
# =============================================================================

set -e

PROJECT_DIR="/scratch.hpc/matteo.preda/Stag-Hunt-LLM"
VENV_DIR="/scratch.hpc/matteo.preda/rl"
REPO_URL="https://github.com/mpreda01/Stag-Hunt-LLM.git"
STAG_HUNT_DIR="/scratch.hpc/matteo.preda/Stag-Hunt-LLM"

# ---------------------------------------------------------------------------
# 1. Create virtual environment
# ---------------------------------------------------------------------------

echo "==> Creating virtual environment 'rl' at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "==> Python: $(which python3) — $(python3 --version)"

# ---------------------------------------------------------------------------
# 2. Upgrade pip
# ---------------------------------------------------------------------------

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA 12.1
#    Check cluster CUDA version with: nvcc --version
# ---------------------------------------------------------------------------

echo "==> Installing PyTorch (CUDA 12.1)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 4. Install all project dependencies directly (no requirements.txt needed)
# ---------------------------------------------------------------------------

echo "==> Installing project dependencies"
pip install \
    numpy \
    pandas \
    tqdm \
    opencv-python \
    gymnasium \
    pygame \
    pettingzoo \
    ollama \
    "transformers>=4.40.0" \
    "accelerate>=0.27.0" \
    "bitsandbytes>=0.43.0"

# ---------------------------------------------------------------------------
# 5. Clone and install Gymnasium-Stag-Hunt
# ---------------------------------------------------------------------------

echo "==> Cloning Gymnasium-Stag-Hunt"
if [ ! -d "$STAG_HUNT_DIR" ]; then
    git clone "$REPO_URL" "$STAG_HUNT_DIR"
else
    echo "   Already cloned, pulling latest"
    git -C "$STAG_HUNT_DIR" pull
fi

echo "==> Installing Gymnasium-Stag-Hunt"
pip install -e "$STAG_HUNT_DIR"

# ---------------------------------------------------------------------------
# 6. Install the project itself from setup.py
# ---------------------------------------------------------------------------

echo "==> Installing Stag-Hunt-LLM project (setup.py)"
pip install -e "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 7. Create output directories
# ---------------------------------------------------------------------------

echo "==> Creating output directories"
mkdir -p "/scratch.hpc/matteo.preda/logs"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/results"

# ---------------------------------------------------------------------------
# 8. Verify key imports
# ---------------------------------------------------------------------------

echo "==> Verifying installation"
python3 - << EOF
import torch
import transformers
import pandas
import cv2
import gymnasium
import gymnasium_stag_hunt
import tqdm
import accelerate
print(f"  torch:          {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"  transformers:   {transformers.__version__}")
print(f"  pandas:         {pandas.__version__}")
print(f"  opencv:         {cv2.__version__}")
print(f"  gymnasium:      {gymnasium.__version__}")
print(f"  tqdm:           {tqdm.__version__}")
print(f"  accelerate:     {accelerate.__version__}")
print("All imports OK.")
EOF

echo ""
echo "============================================================"
echo "  Setup complete. Virtual environment 'rl' is ready."
echo "  Submit the training job with: sbatch Stag-Hunt-LLM/run.sh"
echo "============================================================"