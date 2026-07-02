#!/bin/bash
#SBATCH --job-name=staghunt-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matteo.preda2@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch.hpc/matteo.preda/logs/test_%j.out
#SBATCH --error=/scratch.hpc/matteo.preda/logs/test_%j.err
#SBATCH --chdir=/scratch.hpc/matteo.preda/Stag-Hunt-LLM

set -e

PROJECT_DIR="/scratch.hpc/matteo.preda/Stag-Hunt-LLM"
VENV_DIR="/scratch.hpc/matteo.preda/rl"
SCRATCH="/scratch.hpc/matteo.preda"

# ---- Redirect ALL caches to scratch to avoid home quota ----
export HF_HOME="$SCRATCH/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix: redirect mesa shader cache away from home (was filling quota)
export MESA_SHADER_CACHE_DIR="$SCRATCH/mesa_cache"
export XDG_CACHE_HOME="$SCRATCH/xdg_cache"
mkdir -p "$SCRATCH/mesa_cache" "$SCRATCH/xdg_cache"

# Ollama stores models here — keep off home quota
export OLLAMA_MODELS="$SCRATCH/ollama_models"
mkdir -p "$OLLAMA_MODELS"

echo "============================================================"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  Start  : $(date)"
echo "============================================================"

source "$VENV_DIR/bin/activate"
echo "==> Python: $(which python3) — $(python3 --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Start Ollama server in the background ----
echo "==> Starting Ollama server ..."
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama is ready to accept connections (max 60s)
echo "==> Waiting for Ollama to be ready ..."
for i in $(seq 1 60); do
    if ollama list > /dev/null 2>&1; then
        echo "==> Ollama ready after ${i}s"
        break
    fi
    sleep 1
done

# Pull the model if not already cached in OLLAMA_MODELS
echo "==> Checking qwen3:4b-q4_K_M model ..."
ollama pull qwen3:4b-q4_K_M

cd "$PROJECT_DIR"

# ---- Run test.py non-interactively ----
# MODE: 1=random, 2=zero-shot, 3=one-shot, 4=few-shot
MODE=2
OUTPUT_PATH="$SCRATCH/Stag-Hunt-LLM/outputs/zero_shot/"

mkdir -p "$OUTPUT_PATH"

echo "==> Running test.py  (mode=$MODE, output=$OUTPUT_PATH)"
printf "%s\n%s\n" "$MODE" "$OUTPUT_PATH" | python3 test.py

# ---- Cleanup: stop Ollama server ----
echo "==> Stopping Ollama server (PID $OLLAMA_PID) ..."
kill "$OLLAMA_PID" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Done. End time: $(date)"
echo "============================================================"