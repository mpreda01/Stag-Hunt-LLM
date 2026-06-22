#!/bin/bash
# =============================================================================
# setup.sh — Run ONCE manually on the cluster to create the environment
#
# Usage (from login node):
#   cd /scratch.hpc/matteo.preda
#   bash setup.sh
# =============================================================================

set -e

PROJECT_DIR="/scratch.hpc/matteo.preda"
VENV_DIR="$PROJECT_DIR/rl"
REPO_URL="https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt.git"
STAG_HUNT_DIR="$PROJECT_DIR/Gymnasium-Stag-Hunt"

echo "==> Creating virtual environment 'rl' at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "==> Python: $(which python3) — $(python3 --version)"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing PyTorch with CUDA 12.1"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "==> Installing project dependencies"
pip install -r "$PROJECT_DIR/requirements.txt" --quiet

echo "==> Cloning and installing Gymnasium-Stag-Hunt"
if [ ! -d "$STAG_HUNT_DIR" ]; then
    git clone "$REPO_URL" "$STAG_HUNT_DIR"
fi
pip install -e "$STAG_HUNT_DIR" --quiet

echo "==> Creating output directories"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/results"
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "Setup complete. Virtual environment 'rl' is ready."
echo "Submit the job with: sbatch run.sh"
