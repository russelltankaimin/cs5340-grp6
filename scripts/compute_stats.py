"""
File: scripts/compute_stats.py
Description: Computes the prior statistics (mean and standard deviation) of the 
latent space for a given clean audio file. This matches the original implementation 
by breaking the audio into fixed-length clips to capture temporal variance across clips.
"""
import argparse
import json
import torch
import os
from tqdm import tqdm

from ear_vae.ear_vae import EAR_VAE
from utils.audio_io import load_audio

def split_clips(waveform: torch.Tensor, sr: int, clip_seconds: float) -> torch.Tensor:
    """Splits the audio into fixed-length clips."""
    # waveform shape from load_audio is expected to be (1, channels, total_samples)
    samples_per_clip = int(sr * clip_seconds)
    total_samples = waveform.shape[-1]
    num_clips = total_samples // samples_per_clip

    if num_clips == 0:
        raise ValueError(
            f'Audio too short for T={clip_seconds}s. '
            f'Total duration: {total_samples / sr:.3f}s.'
        )

    # Truncate to a multiple of samples_per_clip
    clipped = waveform[..., : num_clips * samples_per_clip]
    
    # Squeeze the batch dimension, reshape to (channels, num_clips, samples_per_clip) 
    # and then permute to (num_clips, channels, samples_per_clip)
    clipped = clipped.squeeze(0) 
    clips = clipped.reshape(clipped.shape[0], num_clips, samples_per_clip).permute(1, 0, 2)
    return clips

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
    audio = load_audio(args.input, target_sr=target_sr, device=device) # (1, 2, T)

    # 4. Split into clips
    clips = split_clips(audio, target_sr, args.clip_seconds)
    print(f"Number of clips: {clips.shape[0]}, Clip shape: {clips.shape[1:]}")

    # 5. Compute Latent Statistics
    latents = []
    with torch.no_grad():
        for clip in tqdm(clips, desc="Computing latents", total=clips.shape[0]):
            clip_batch = clip.unsqueeze(0)  # (1, channels, samples)
            # Use the mean of the latent distribution (use_sample=False) for stable stats
            latent = model.encode(clip_batch, use_sample=False) 
            latents.append(latent)

    # Concatenate all latents -> Shape: (num_clips, 64, latent_duration_dim)
    all_latents = torch.cat(latents, dim=0)  

    # Mean is computed across clips (dim=0) and time (dim=2) -> Shape: (64,)
    mean = all_latents.mean(dim=(0, 2))
    
    # Std is computed across clips ONLY (dim=0) -> Shape: (64, latent_duration_dim)
    # This preserves the time-dependent variance structure required by the objective loss
    stds = all_latents.std(dim=0, unbiased=False)
    
    # Prevent pure zero standard deviations to avoid division by zero in loss_w
    stds = torch.clamp(stds, min=1e-6)

    stats = {
        "mean": mean.detach().cpu().tolist(),
        "stds": stds.detach().cpu().tolist()
    }

    # 6. Save to JSON
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Latent statistics successfully saved to {args.output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Compute prior stats for Bayesian Reconstruction.")
    p.add_argument('--input', type=str, required=True, help='Path to clean reference audio.')
    p.add_argument('--output', type=str, default='latent_stats.json', help='Output JSON path.')
    p.add_argument('--vae-checkpoint', type=str, required=True, help='Path to model weights.')
    p.add_argument('--vae-config', type=str, required=True, help='Path to model config.')
    p.add_argument('--clip-seconds', type=float, default=5.0, help='Length of sub-clips in seconds.')
    
    compute_latent_stats(p.parse_args())