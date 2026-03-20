"""
File: scripts/compute_stats.py
Description: Computes the prior statistics (mean and standard deviation) of the 
latent space for a given clean audio file. This is crucial for calculating the L_w 
(Gaussian prior) loss during the Bayesian reconstruction update.
"""
import argparse
import json
import torch
import numpy as np
import os

from ear_vae.ear_vae import EAR_VAE
from utils.audio_io import load_audio

def compute_latent_stats(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on: {device}")

    # 1. Load Model Config
    with open(args.vae_config, 'r') as f:
        config = json.load(f)
    target_sr = config.get("sample_rate", 44100)

    # 2. Initialise and load model
    model = EAR_VAE(model_config=config).to(device)
    model.load_state_dict(torch.load(args.vae_checkpoint, map_location="cpu"))
    model.eval()

    # 3. Load audio using standardised IO
    print(f"Loading audio: {args.input}")
    audio = load_audio(args.input, target_sr=target_sr, device=device)

    # 4. Compute Latent Statistics
    with torch.no_grad():
        latent_z = model.encode(audio, use_sample=False) # Use mean directly for stable stats
    
    # Flatten across time dimension for global statistics
    latent_np = latent_z.squeeze(0).cpu().numpy() # Shape: (latent_dim, time_steps)
    
    # Calculate mean and std for each latent dimension across all time steps
    means = np.mean(latent_np, axis=1).tolist()
    stds = np.std(latent_np, axis=1).tolist()

    # Prevent pure zero standard deviations
    stds = [max(s, 1e-6) for s in stds]

    stats = {
        "mean": means,
        "stds": stds
    }

    # 5. Save to JSON
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Latent statistics successfully saved to {args.output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Compute prior stats for Bayesian Reconstruction.")
    p.add_argument('--input', type=str, required=True, help='Path to clean reference audio.')
    p.add_argument('--output', type=str, default='latent_stats.json', help='Output JSON path.')
    p.add_argument('--vae-checkpoint', type=str, required=True, help='Path to model weights.')
    p.add_argument('--vae-config', type=str, required=True, help='Path to model config.')
    
    compute_latent_stats(p.parse_args())