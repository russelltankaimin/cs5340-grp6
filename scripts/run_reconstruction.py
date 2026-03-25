"""
File: scripts/run_reconstruction.py
Description: The main Bayesian optimization script. It locks the VAE weights and 
optimizes the latent vector `z` using Adam and Cosine Annealing.
"""

import argparse
import json
import torch

from ear_vae.ear_vae import EAR_VAE
from corruptions.registry import CORRUPTION_REGISTRY
from utils.losses import loss_w, loss_colin, loss_waveform, get_mel_loss_fn, loss_trajectory
from utils.audio_io import load_audio, save_audio

def load_vae(config_path: str, ckpt_path: str, device: str) -> EAR_VAE:
    """Initialises the pre-trained, frozen EAR_VAE model."""
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model = EAR_VAE(model_config=cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def reconstruct(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on: {device}")
    
    # 1. Load configuration and files
    with open(args.vae_config, "r") as f:
        target_sr = json.load(f).get("sample_rate", 44100)
        
    vae = load_vae(args.vae_config, args.vae_checkpoint, device)
    corrupted_audio = load_audio(args.input, target_sr=target_sr, device=device)
    
    with open(args.prior_stats, "r") as f:
        stats = json.load(f)
    mu = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, -1, 1)
    sigma = torch.tensor(stats["stds"], dtype=torch.float32, device=device).unsqueeze(0)

    # 2. Setup the target differentiable corruption
    corruption_fn = CORRUPTION_REGISTRY[args.corruption]
    corruption_kwargs = json.loads(args.corruption_kwargs)

    # 3. Warm start the latent vector using the encoder
    with torch.no_grad():
        z = vae.encode(corrupted_audio).clone().detach().requires_grad_(True)

    # 4. Setup Optimizer & Loss
    loss_mel = get_mel_loss_fn(target_sr, device, args.n_fft, args.hop_length, args.n_mels)
    optimiser = torch.optim.Adam([z], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.steps, eta_min=1e-5)

    best_loss = float("inf")
    best_z = z.clone().detach()

    # 5. Backpropagation Loop
    for step in range(1, args.steps + 1):
        optimiser.zero_grad()

        # Forward pass: Decode -> Re-corrupt
        recon = vae.decode(z)
        corrupted_recon = corruption_fn(recon.squeeze(0), **corruption_kwargs).unsqueeze(0)

        # Compute objective components
        Lw    = loss_w(z, mu, sigma)
        Lcol  = loss_colin(z)
        Lwav  = loss_waveform(corrupted_audio, corrupted_recon)
        Lmel  = loss_mel(corrupted_audio.squeeze(0), corrupted_recon.squeeze(0))
        Ltraj = loss_trajectory(z, args.lambda_0, args.lambda_1, args.lambda_2)

        total = (args.lambda_w * Lw + args.lambda_colin * Lcol + 
                 args.lambda_wav * Lwav + args.lambda_mel * Lmel + Ltraj)

        total.backward()
        optimiser.step()
        scheduler.step()

        if total.item() < best_loss:
            best_loss = total.item()
            best_z = z.clone().detach()

        if step % args.log_every == 0 or step == 1:
            print(f"[{step}/{args.steps}] Loss: {total.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 6. Save final output
    with torch.no_grad():
        final_audio = vae.decode(best_z)
    save_audio(args.output, final_audio, target_sr)
    print(f"Reconstruction saved to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="reconstructed.wav")
    p.add_argument("--vae-checkpoint", type=str, required=True)
    p.add_argument("--vae-config", type=str, required=True)
    p.add_argument("--prior-stats", type=str, required=True)
    p.add_argument("--K", type=int, default=215)
    p.add_argument("--corruption", type=str, required=True)
    p.add_argument("--corruption-kwargs", type=str, default="{}")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--lambda-w", type=float, default=1.0)
    p.add_argument("--lambda-colin", type=float, default=1.0)
    p.add_argument("--lambda-wav", type=float, default=1.0)
    p.add_argument("--lambda-mel", type=float, default=1.0)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=80)
    
    # Trajectory Prior Weights
    p.add_argument("--lambda-0", type=float, default=1.0)
    p.add_argument("--lambda-1", type=float, default=0.1)
    p.add_argument("--lambda-2", type=float, default=0.1)
    
    reconstruct(p.parse_args())