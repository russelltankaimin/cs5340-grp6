#!/bin/bash
#SBATCH --job-name=cs5340-pipeline
#SBATCH --partition=gpu            # adjust to your cluster
#SBATCH --gres=gpu:h100-96               # or comment out if CPU only
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Optional: activate your environment / modules ---
VENV="${HOME}/cs5340-venv"
source "${VENV}/bin/activate"

mkdir -p logs

# Paths (adjust as needed). Assumes both files live in the same directory.
SCRIPT_DIR="${HOME}/cs5340"
PY_DRIVER="${SCRIPT_DIR}/pipeline.py"

# Variables
INPUT="${SCRIPT_DIR}/data_full/test-clean/61/70968/61-70968-0000.flac"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${SLURM_JOB_ID}"

# Move to the project root
cd "${SCRIPT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Explicitly add the project root to Python's search path
export PYTHONPATH="${SCRIPT_DIR}"
export LD_LIBRARY_PATH=/home/m/muchen/cs5340-venv/lib/python3.12/site-packages/nvidia/nvrtc/lib:$LD_LIBRARY_PATH

nvidia-smi

# Run
python "${PY_DRIVER}" \
  --audio-path "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --fill-last-clip

