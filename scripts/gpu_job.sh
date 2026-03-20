#!/bin/bash

# Job name
#SBATCH --job-name=cs5340_reconstruction
#SBATCH --partition=gpu

## Request the appropriate GPU:
#SBATCH --gres=gpu:h100-96:1  # Use H100-96 GPU

## Set the runtime duration (adjust based on how long you expect the job to take)
#SBATCH --time=02:59:00  # HH:MM:SS (change as necessary)

# Resources: single task, single CPU core, 32 GB of memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

## Log file names for output and error
#SBATCH --output=./logs/stdout_recon_%j.slurmlog
#SBATCH --error=./logs/stderr_recon_%j.slurmlog

# Display GPU status
nvidia-smi

# Activate virtual environment
source ".venv/bin/activate"

# Remove cache
rm -r __pycache__

# Run the Bayesian reconstruction script as a module to ensure paths resolve correctly.
# Ensure your input audio is placed in the 'data/' directory (or update the path below).
python -m scripts.run_reconstruction \
    --input data/distorted_output_sinus.wav \
    --output data/reconstructed.wav \
    --corruption sinusoidal \
    --corruption-kwargs '{"noise_level": 0.1}' \
    --prior-stats latent_stats.json \
    --vae-checkpoint vae_ckpt/ear_vae_44k.pyt \
    --vae-config vae_ckpt/model_config.json \
    --steps 10000 \
    --lr 1e-3
