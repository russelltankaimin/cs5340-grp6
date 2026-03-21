"""
Usage
-----
    python exp_v1.py \
        --input corrupted.wav \
        --corruption soft_clip \
        --corruption-kwargs '{"drive": 15.0}' \
        --prior-stats latent_stats.json \
        --steps 500 --lr 1e-3

Adding a new corruption function
--------------------------------
1. Define a function  f(waveform: Tensor, **kwargs) -> Tensor  that is
   differentiable w.r.t. `waveform` and implement it in corruptors/*.py.
2. Register it:  CORRUPTION_REGISTRY["my_fn"] = my_fn
"""

import argparse
import json
import os

import torch
import torchaudio

# =========================================================================
# 1.  CORRUPTION FUNCTION REGISTRY
# =========================================================================

from corruptors.waveform_sinus_dist import sinusoidal_noise
from corruptors.waveform_soft_clip_dist import soft_clip_distortion
from corruptors.waveform_tap_n_flutter import tape_wow_flutter

# Register built-in corruption functions here.
# To add your own, just append:  CORRUPTION_REGISTRY["name"] = your_fn
CORRUPTION_REGISTRY: dict[str, callable] = {
    "sinusoidal": sinusoidal_noise,
    "soft_clip":  soft_clip_distortion,
    "tape_wow_flutter": tape_wow_flutter,
}

# ── VAE paths (defaults – override via CLI) ──────────────────────────────
DEFAULT_VAE_CKPT   = os.path.join(".", "vae_ckpt", "ear_vae_44k.pyt")
DEFAULT_VAE_CONFIG = os.path.join(".", "vae_ckpt", "model_config.json")

# =========================================================================
# 2.  LOSS FUNCTIONS
# =========================================================================

def loss_w(z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian prior loss on the latent vector (element-wise).
    L_w = Σ_i ((z_i − μ_i) / σ_i)²
    """
    return ((z - mu) / sigma).pow(2).mean()


def loss_colin(z: torch.Tensor, K: int) -> torch.Tensor:
    """Colinearity loss – encourages K equal-sized sub-vectors of z to be
    mutually aligned (high cosine similarity).
    L_colin = −Σ_{i<j} cos_sim(v_i, v_j)
    Minimising this maximises pairwise cosine similarity.
    """
    vectors = z.reshape(K, -1)                          # (K, D)
    # Normalise once, then use dot products
    normed = torch.nn.functional.normalize(vectors, dim=1)
    sim_matrix = normed @ normed.T                      # (K, K)
    # Upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(K, K, device=z.device), diagonal=1).bool()
    return -sim_matrix[mask].mean()


def loss_waveform(
    corrupted_input: torch.Tensor,
    corrupted_recon: torch.Tensor,
) -> torch.Tensor:
    """L_waveform = ‖I − f(G(z))‖²"""
    T = min(corrupted_input.shape[-1], corrupted_recon.shape[-1])
    return (corrupted_input[..., :T] - corrupted_recon[..., :T]).pow(2).mean()


def loss_mel(
    corrupted_input: torch.Tensor,
    corrupted_recon: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    """L_mel = ‖log-mel(I) − log-mel(f(G(z)))‖²"""
    T = min(corrupted_input.shape[-1], corrupted_recon.shape[-1])
    mel_in  = torch.log(mel_transform(corrupted_input[..., :T])  + 1e-9)
    mel_rec = torch.log(mel_transform(corrupted_recon[..., :T]) + 1e-9)
    return (mel_in - mel_rec).pow(2).mean()


# =========================================================================
# 3.  AUDIO I/O HELPERS
# =========================================================================

def load_audio(path: str, target_sr: int = 44100, device: str = "cpu") -> torch.Tensor:
    """Load audio → stereo float32 tensor of shape (1, 2, T) at target_sr."""
    y, sr = torchaudio.load(path, backend="ffmpeg")
    if sr != target_sr:
        y = torchaudio.transforms.Resample(sr, target_sr)(y)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if y.shape[0] == 1:
        y = y.repeat(2, 1)           # mono → stereo
    return y.unsqueeze(0).to(device)  # (1, 2, T)


def load_vae(config_path: str, ckpt_path: str, device: str):
    """Load the frozen EAR_VAE model."""
    from ear_vae.ear_vae import EAR_VAE

    with open(config_path, "r") as f:
        cfg = json.load(f)
    model = EAR_VAE(model_config=cfg).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =========================================================================
# 4.  RECONSTRUCTION LOOP
# =========================================================================

def reconstruct(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model & audio ───────────────────────────────────────────────
    vae = load_vae(args.vae_config, args.vae_checkpoint, device)

    corrupted_audio = load_audio(args.input, device=device)  # (1, 2, T)
    print(f"Corrupted audio shape: {corrupted_audio.shape}")

    # ── Latent prior statistics ──────────────────────────────────────────
    with open(args.prior_stats, "r") as f:
        stats = json.load(f)
    mu    = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, -1, 1)
    sigma = torch.tensor(stats["stds"], dtype=torch.float32, device=device).unsqueeze(0)

    # ── Corruption function ──────────────────────────────────────────────
    corruption_fn     = CORRUPTION_REGISTRY[args.corruption]
    corruption_kwargs = json.loads(args.corruption_kwargs)
    print(f"Corruption: {args.corruption}  kwargs: {corruption_kwargs}")

    # ── Initialise latent from encoder (warm start) ──────────────────────
    with torch.no_grad():
        z_init = vae.encode(corrupted_audio)
    z = z_init.clone().detach().requires_grad_(True)
    print(f"Latent shape: {z.shape}  (K={args.K} sub-vectors)")

    # ── Mel-spectrogram transform (differentiable) ───────────────────────
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    ).to(device)

    # ── Optimiser ────────────────────────────────────────────────────────
    optimiser = torch.optim.Adam([z], lr=args.lr)

    # ── Optimisation ─────────────────────────────────────────────────────
    best_loss  = float("inf")
    best_z     = z.clone().detach()

    for step in range(1, args.steps + 1):
        optimiser.zero_grad()

        # Decode current latent → clean estimate  G(z)
        recon = vae.decode(z)                            # (1, 2, T')

        # Apply known corruption  f(G(z))  –  squeeze/unsqueeze batch dim
        corrupted_recon = corruption_fn(
            recon.squeeze(0), **corruption_kwargs
        ).unsqueeze(0)

        # ── Individual losses ────────────────────────────────────────────
        # print(f"z shape: {z.shape}  mu shape: {mu.shape}  sigma shape: {sigma.shape}")
        Lw    = loss_w(z, mu, sigma)
        Lcol  = loss_colin(z, args.K)
        Lwav  = loss_waveform(corrupted_audio, corrupted_recon)
        Lmel  = loss_mel(
            corrupted_audio.squeeze(0),
            corrupted_recon.squeeze(0),
            mel_transform,
        )

        total = (
            args.lambda_w       * Lw
            + args.lambda_colin * Lcol
            + args.lambda_wav   * Lwav
            + args.lambda_mel   * Lmel
        )

        total.backward()
        optimiser.step()

        # Track best
        if total.item() < best_loss:
            best_loss = total.item()
            best_z    = z.clone().detach()

        if step % args.log_every == 0 or step == 1:
            print(
                f"[{step:>5}/{args.steps}]  total={total.item():.4f}  "
                f"L_w={Lw.item():.4f}  L_colin={Lcol.item():.4f}  "
                f"L_wav={Lwav.item():.4f}  L_mel={Lmel.item():.4f}",
                flush=True
            )

    # ── Decode best latent & save ────────────────────────────────────────
    with torch.no_grad():
        final_audio = vae.decode(best_z).squeeze(0).cpu()  # (2, T')
    torchaudio.save(args.output, final_audio, sample_rate=44100, backend="ffmpeg")
    print(f"\nSaved reconstructed audio → {args.output}")


# =========================================================================
# 5.  CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian Audio Reconstruction through Generative Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input",  type=str, required=True,
                   help="Path to the corrupted .wav input.")
    p.add_argument("--output", type=str, default="reconstructed.wav",
                   help="Path for the reconstructed .wav output.")

    # VAE
    p.add_argument("--vae-checkpoint", type=str, default=DEFAULT_VAE_CKPT)
    p.add_argument("--vae-config",     type=str, default=DEFAULT_VAE_CONFIG)

    # Latent prior
    p.add_argument("--prior-stats", type=str, required=True,
                   help='JSON file with keys "mean" and "stds".')
    p.add_argument("--K", type=int, default=215,
                   help="Number of latent sub-vectors for L_colin. 5s -> 215")

    # Corruption
    p.add_argument("--corruption", type=str, required=True,
                   choices=list(CORRUPTION_REGISTRY.keys()),
                   help="Name of the corruption function to invert.")
    p.add_argument("--corruption-kwargs", type=str, default="{}",
                   help='JSON string of kwargs forwarded to the corruption fn.')

    # Optimisation
    p.add_argument("--steps",     type=int,   default=500)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--log-every", type=int,   default=50)

    # Loss weights (λ)
    p.add_argument("--lambda-w",     type=float, default=1.0)
    p.add_argument("--lambda-colin", type=float, default=1.0)
    p.add_argument("--lambda-wav",   type=float, default=1.0)
    p.add_argument("--lambda-mel",   type=float, default=1.0)

    # Mel spectrogram
    p.add_argument("--n-fft",      type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-mels",     type=int, default=80)

    return p.parse_args()


if __name__ == "__main__":
    reconstruct(parse_args())