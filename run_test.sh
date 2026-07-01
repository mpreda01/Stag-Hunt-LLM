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

export HF_HOME="/scratch.hpc/matteo.preda/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  Start  : $(date)"
echo "============================================================"

source "$VENV_DIR/bin/activate"
echo "==> Python: $(which python3) — $(python3 --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$PROJECT_DIR"

# test.py uses input() which won't work in a batch job.
# We pipe in the mode and output path non-interactively.
# Edit MODE and OUTPUT_PATH below to change behaviour:
#   MODE=1  → random agent
#   MODE=2  → Qwen4b zero shot
#   MODE=3  → Qwen4b one shot
#   MODE=4  → Qwen4b few shot
MODE=4
OUTPUT_PATH="/scratch.hpc/matteo.preda/Stag-Hunt-LLM/outputs/few_shot"

mkdir -p "$OUTPUT_PATH"

echo "==> Running test.py  (mode=$MODE, output=$OUTPUT_PATH)"
printf "%s\n%s\n" "$MODE" "$OUTPUT_PATH" | python3 test.py

echo ""
echo "============================================================"
echo "  Done. End time: $(date)"
echo "============================================================"
