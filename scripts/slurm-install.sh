#!/bin/bash
#SBATCH --job-name=cs5340-install
#SBATCH --partition=gpu            # adjust to your cluster
#SBATCH --gres=gpu:1               # or comment out if CPU only
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
REQUIREMENTS="${SCRIPT_DIR}/requirements_new.txt"

nvidia-smi
pip uninstall torchcodec torch torchvision torchaudio -y
pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu126
