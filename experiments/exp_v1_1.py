"""
exp_v2_xtraj_jacobian.py

What this version does
----------------------
1. Keeps a weak latent-space trajectory prior for optimization stability.
2. Adds a stronger x-space trajectory prior on the reconstructed clean waveform.
3. Keeps the rectified corruption Jacobian term for supported corruptions:
   - sinusoidal: exact, zero
   - soft_clip: exact
   - tape_wow_flutter: continuous-time approximation
4. Disables colinearity by default.
5. Disables second-order latent trajectory by default.
6. Enables first/second-order x-space trajectory terms.

Example
-------
python exp_v2_xtraj_jacobian.py \
  --input corrupted.wav \
  --output reconstructed.wav \
  --corruption soft_clip \
  --corruption-kwargs '{"drive": 15.0}' \
  --prior-stats latent_stats.json \
  --steps 500 \
  --lr 1e-3 \
  --include-jacobian \
  --lambda-logdet 1e-6
"""

import argparse
import json
import math
import os
# for cases with multiple GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torchaudio
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

DEFAULT_VAE_CKPT = os.path.join(".", "vae_ckpt", "ear_vae_44k.pyt")
DEFAULT_VAE_CONFIG = os.path.join(".", "vae_ckpt", "model_config.json")

EPS = 1.0e-12


# =========================================================================
# 1. CORRUPTION FUNCTIONS
# =========================================================================


def sinusoidal_noise(
    waveform: torch.Tensor,
    noise_level: float = 0.1,
    num_components: int = 64,
) -> torch.Tensor:
    num_samples = waveform.shape[-1]
    t = torch.arange(num_samples, device=waveform.device, dtype=waveform.dtype)
    noise = torch.zeros_like(waveform)
    for i in range(1, num_components + 1):
        freq = i * math.sqrt(2) * math.pi * (1 + i * 0.618033)
        phase = i * i * 1.3579
        noise = noise + torch.sin(freq * t / num_samples * 2 * math.pi + phase)
    noise = noise / noise.std().clamp_min(1e-8)
    return waveform + noise_level * noise


def soft_clip_distortion(
    waveform: torch.Tensor,
    drive: float = 5.0,
) -> torch.Tensor:
    return torch.tanh(waveform * drive)

def soft_clip_distortion_rms(
    waveform: torch.Tensor,
    drive: float = 50.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Tanh soft-clipping distortion (fully differentiable, RMS-matched)."""
    distorted = torch.tanh(waveform * drive)
    rms_in = waveform.square().mean().sqrt() + eps
    rms_out = distorted.square().mean().sqrt() + eps
    return distorted * (rms_in / rms_out)


def tape_wow_flutter(
    waveform: torch.Tensor,
    wow_freq: float = 1.5,
    wow_depth: float = 0.003,
    flutter_freq: float = 28.0,
    flutter_depth: float = 0.0005,
    sample_rate: int = 44100,
) -> torch.Tensor:
    T = waveform.shape[-1]
    t = torch.arange(T, device=waveform.device, dtype=waveform.dtype) / sample_rate

    displacement = (
        wow_depth * torch.sin(2 * math.pi * wow_freq * t)
        + flutter_depth * torch.sin(2 * math.pi * flutter_freq * t)
    ) * sample_rate

    indices = torch.arange(T, device=waveform.device, dtype=waveform.dtype) + displacement
    indices = indices.clamp(0, T - 1)

    idx_floor = indices.long().clamp(0, T - 2)
    idx_ceil = idx_floor + 1
    frac = (indices - idx_floor.float()).unsqueeze(0)

    output = waveform[:, idx_floor] * (1 - frac) + waveform[:, idx_ceil] * frac
    return output


@dataclass(frozen=True)
class CorruptionSpec:
    fn: Callable[..., torch.Tensor]
    jacobian_mode: str
    jacobian_fn: Callable[..., torch.Tensor] | None = None
    note: str = ""


# =========================================================================
# 2. LOSSES
# =========================================================================


def loss_w(z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return ((z - mu) / sigma).pow(2).mean()


def loss_colin(z: torch.Tensor, K: int) -> torch.Tensor:
    vectors = z.reshape(K, -1)
    normed = torch.nn.functional.normalize(vectors, dim=1)
    sim_matrix = normed @ normed.T
    mask = torch.triu(torch.ones(K, K, device=z.device), diagonal=1).bool()
    return -sim_matrix[mask].mean()


def loss_waveform(
    corrupted_input: torch.Tensor,
    corrupted_recon: torch.Tensor,
) -> torch.Tensor:
    T = min(corrupted_input.shape[-1], corrupted_recon.shape[-1])
    return (corrupted_input[..., :T] - corrupted_recon[..., :T]).pow(2).mean()


def loss_mel(
    corrupted_input: torch.Tensor,
    corrupted_recon: torch.Tensor,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    T = min(corrupted_input.shape[-1], corrupted_recon.shape[-1])
    mel_in = torch.log(mel_transform(corrupted_input[..., :T]) + 1e-9)
    mel_rec = torch.log(mel_transform(corrupted_recon[..., :T]) + 1e-9)
    return (mel_in - mel_rec).pow(2).mean()


def latent_trajectory_terms(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    z shape: (B, D, T')
    Returns:
      L0z = average latent energy
      L1z = first-order latent temporal smoothness
      L2z = second-order latent temporal smoothness
    """
    w = z.squeeze(0).T  # (T', D)
    T_prime = w.shape[0]

    L0z = w.pow(2).sum(dim=1).mean()

    if T_prime > 1:
        diff1 = w[1:] - w[:-1]
        L1z = diff1.pow(2).sum(dim=1).mean()
    else:
        L1z = z.new_zeros(())

    if T_prime > 2:
        diff2 = w[2:] - 2 * w[1:-1] + w[:-2]
        L2z = diff2.pow(2).sum(dim=1).mean()
    else:
        L2z = z.new_zeros(())

    return L0z, L1z, L2z


def x_trajectory_terms(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x shape: (B, C, T)
    Returns:
      L0x = average waveform energy
      L1x = first-order waveform smoothness
      L2x = second-order waveform smoothness
    """
    L0x = x.pow(2).mean()

    if x.shape[-1] > 1:
        dx1 = x[..., 1:] - x[..., :-1]
        L1x = dx1.pow(2).mean()
    else:
        L1x = x.new_zeros(())

    if x.shape[-1] > 2:
        dx2 = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
        L2x = dx2.pow(2).mean()
    else:
        L2x = x.new_zeros(())

    return L0x, L1x, L2x


# =========================================================================
# 3. CORRUPTION JACOBIAN LOG-DETERMINANTS
# =========================================================================


def logabsdet_jacobian_zero(
    clean_waveform: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    return clean_waveform.new_zeros(())


def logabsdet_jacobian_soft_clip(
    clean_waveform: torch.Tensor,
    drive: float = 15.0,
    **kwargs: Any,
) -> torch.Tensor:
    drive_t = torch.as_tensor(
        drive, dtype=clean_waveform.dtype, device=clean_waveform.device
    )
    if torch.any(torch.abs(drive_t) < EPS):
        raise ValueError("soft_clip drive must be non-zero.")

    scaled = drive_t * clean_waveform
    sech_sq = torch.cosh(scaled).pow(-2)
    diag_abs = torch.abs(drive_t) * sech_sq
    return torch.log(diag_abs.clamp_min(EPS)).sum()

def logabsdet_jacobian_soft_clip_rms(
    clean_waveform: torch.Tensor,
    drive: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    x = clean_waveform
    N = x.numel()

    # --- forward quantities ---
    scaled = drive * x
    d = torch.tanh(scaled)
    sech_sq = torch.cosh(scaled).pow(-2)
    s = drive * sech_sq                       # d(tanh)/dx per sample

    # --- RMS quantities ---
    sigma_in = x.square().mean().sqrt()
    sigma_out = d.square().mean().sqrt()
    rms_in = sigma_in + eps
    rms_out = sigma_out + eps
    R = rms_in / rms_out

    # --- Jacobian = diag(a) + u vᵀ ---
    # diagonal:  a_i = R · s_i
    a = R * s

    # rank-1 vectors:  u_i = d_i,  v_j = ∂R/∂x_j
    #   ∂R/∂x_j = x_j / (N·σ_in·rms_out)
    #            − R·d_j·s_j / (N·σ_out·rms_out)
    dR_dx = (x / (N * sigma_in.clamp_min(eps) * rms_out)
             - R * d * s / (N * sigma_out.clamp_min(eps) * rms_out))

    # --- matrix determinant lemma ---
    # log|det(J)| = Σ log|a_i|  +  log|1 + vᵀ diag(a)⁻¹ u|
    log_diag = torch.log(a.abs().clamp_min(eps)).sum()
    correction = (dR_dx * d / a.clamp_min(eps)).sum()
    log_rank1 = torch.log((1.0 + correction).abs().clamp_min(eps))

    return log_diag + log_rank1

def logabsdet_jacobian_tape_wow_flutter(
    clean_waveform: torch.Tensor,
    wow_freq: float = 1.5,
    wow_depth: float = 0.003,
    flutter_freq: float = 28.0,
    flutter_depth: float = 0.0005,
    sample_rate: int = 44100,
    **kwargs: Any,
) -> torch.Tensor:
    T = clean_waveform.shape[-1]
    t = torch.arange(T, device=clean_waveform.device, dtype=clean_waveform.dtype) / sample_rate

    delta_prime = (
        wow_depth * (2 * math.pi * wow_freq) * torch.cos(2 * math.pi * wow_freq * t)
        + flutter_depth * (2 * math.pi * flutter_freq) * torch.cos(2 * math.pi * flutter_freq * t)
    ) * sample_rate

    jac_diag = torch.abs(1.0 + delta_prime).clamp_min(EPS)
    return torch.log(jac_diag).sum()


CORRUPTION_REGISTRY: dict[str, CorruptionSpec] = {
    "sinusoidal": CorruptionSpec(
        fn=sinusoidal_noise,
        jacobian_mode="constant_zero",
        jacobian_fn=logabsdet_jacobian_zero,
        note="Additive deterministic pseudo-noise: J = I.",
    ),
    "soft_clip": CorruptionSpec(
        fn=soft_clip_distortion,
        jacobian_mode="exact",
        jacobian_fn=logabsdet_jacobian_soft_clip,
        note="Elementwise tanh soft clip with exact diagonal Jacobian.",
    ),
    "soft_clip_rms": CorruptionSpec(
        fn=soft_clip_distortion_rms,
        jacobian_mode="exact",
        jacobian_fn=logabsdet_jacobian_soft_clip_rms,
        note="Elementwise tanh soft clip with RMS-matching and exact Jacobian via matrix determinant lemma.",
    ),
    "tape_wow_flutter": CorruptionSpec(
        fn=tape_wow_flutter,
        jacobian_mode="approximate",
        jacobian_fn=logabsdet_jacobian_tape_wow_flutter,
        note="Continuous-time approximation for time-warp log-determinant.",
    ),
}


# =========================================================================
# 4. AUDIO I/O
# =========================================================================


def load_audio(path: str, target_sr: int = 44100, device: str = "cpu") -> torch.Tensor:
    decoder = AudioDecoder(path)
    samples = decoder.get_all_samples()
    y = samples.data
    sr = samples.sample_rate

    if sr != target_sr:
        y = torchaudio.transforms.Resample(sr, target_sr)(y)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if y.shape[0] == 1:
        y = y.repeat(2, 1)
    return y.unsqueeze(0).to(device)


def load_vae(config_path: str, ckpt_path: str, device: str):
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
# 5. RECONSTRUCTION
# =========================================================================


def compute_logabsdet_term(
    spec: CorruptionSpec,
    clean_waveform: torch.Tensor,
    corruption_kwargs: dict[str, Any],
    include_jacobian: bool,
    allow_unsupported_jacobian: bool,
) -> torch.Tensor:
    if not include_jacobian:
        return clean_waveform.new_zeros(())

    if spec.jacobian_fn is None:
        return clean_waveform.new_zeros(())

    try:
        return spec.jacobian_fn(clean_waveform=clean_waveform, **corruption_kwargs)
    except NotImplementedError:
        if allow_unsupported_jacobian:
            return clean_waveform.new_zeros(())
        raise


def reconstruct(args: argparse.Namespace) -> None:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    vae = load_vae(args.vae_config, args.vae_checkpoint, device)

    corrupted_audio = load_audio(args.input, device=device)
    print(f"Corrupted audio shape: {corrupted_audio.shape}")

    with open(args.prior_stats, "r") as f:
        stats = json.load(f)

    mu = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, -1, 1)
    sigma = torch.tensor(stats["stds"], dtype=torch.float32, device=device).unsqueeze(0)
    sigma = torch.clamp(sigma, min=1e-2)

    spec = CORRUPTION_REGISTRY[args.corruption]
    corruption_kwargs = json.loads(args.corruption_kwargs)
    corruption_fn = spec.fn
    print(f"Corruption: {args.corruption}  kwargs: {corruption_kwargs}")
    print(f"Jacobian mode: {spec.jacobian_mode} — {spec.note}")

    with torch.no_grad():
        z_init = vae.encode(corrupted_audio)
    z = z_init.clone().detach().requires_grad_(True)
    print(f"Latent shape: {z.shape}")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    ).to(device)

    optimiser = torch.optim.Adam([z], lr=args.lr)

    best_loss = float("inf")
    best_z = z.clone().detach()

    for step in range(1, args.steps + 1):
        optimiser.zero_grad()

        recon = vae.decode(z)              # (B, C, T)
        clean_recon = recon.squeeze(0)     # (C, T)
        corrupted_recon = corruption_fn(clean_recon, **corruption_kwargs).unsqueeze(0)

        # Priors / data terms
        Lw = loss_w(z, mu, sigma)

        if args.lambda_colin != 0.0:
            Lcol = loss_colin(z, args.K)
        else:
            Lcol = z.new_zeros(())

        Lwav = loss_waveform(corrupted_audio, corrupted_recon)
        Lmel = loss_mel(
            corrupted_audio.squeeze(0),
            corrupted_recon.squeeze(0),
            mel_transform,
        )

        # Latent trajectory
        L0z, L1z, L2z = latent_trajectory_terms(z)
        Ltraj_z = (
            args.lambda_z0 * L0z
            + args.lambda_z1 * L1z
            + (args.lambda_z2 * L2z if args.enable_latent_second_order else z.new_zeros(()))
        )

        # x-space trajectory on clean reconstruction
        L0x, L1x, L2x = x_trajectory_terms(recon)
        Ltraj_x = (
            args.lambda_x0 * L0x
            + args.lambda_x1 * L1x
            + (args.lambda_x2 * L2x if args.enable_x_second_order else z.new_zeros(()))
        )

        logabsdet_Jf = compute_logabsdet_term(
            spec=spec,
            clean_waveform=clean_recon,
            corruption_kwargs=corruption_kwargs,
            include_jacobian=args.include_jacobian,
            allow_unsupported_jacobian=args.allow_unsupported_jacobian,
        )

        total = (
            args.lambda_w * Lw
            + args.lambda_colin * Lcol
            + args.lambda_wav * Lwav
            + args.lambda_mel * Lmel
            + Ltraj_z
            + Ltraj_x
            - args.lambda_logdet * logabsdet_Jf
        )

        total.backward()
        optimiser.step()

        if total.item() < best_loss:
            best_loss = total.item()
            best_z = z.clone().detach()

        if step % args.log_every == 0 or step == 1:
            print(
                f"[{step:>5}/{args.steps}] total={total.item():.6f}  "
                f"Lw={Lw.item():.6f}  "
                f"Lcol={Lcol.item():.6f}  "
                f"Lwav={Lwav.item():.6f}  "
                f"Lmel={Lmel.item():.6f}  "
                f"Ltraj_z={Ltraj_z.item():.6f} (z0={L0z.item():.6f}, z1={L1z.item():.6f}, z2={L2z.item():.6f})  "
                f"Ltraj_x={Ltraj_x.item():.6f} (x0={L0x.item():.6f}, x1={L1x.item():.6f}, x2={L2x.item():.6f})  "
                f"log|detJf|={logabsdet_Jf.item():.6f}",
                flush=True,
            )

    with torch.no_grad():
        final_audio = vae.decode(best_z).squeeze(0).cpu()
    AudioEncoder(final_audio, sample_rate=44100).to_file(args.output)
    print(f"\nSaved reconstructed audio -> {args.output}")


# =========================================================================
# 6. CLI
# =========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audio reconstruction with weak latent trajectory, strong x-space trajectory, and optional corruption Jacobians",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="reconstructed.wav")

    p.add_argument("--vae-checkpoint", type=str, default=DEFAULT_VAE_CKPT)
    p.add_argument("--vae-config", type=str, default=DEFAULT_VAE_CONFIG)

    p.add_argument("--prior-stats", type=str, required=True)
    p.add_argument("--K", type=int, default=215)

    p.add_argument(
        "--corruption",
        type=str,
        required=True,
        choices=list(CORRUPTION_REGISTRY.keys()),
    )
    p.add_argument("--corruption-kwargs", type=str, default="{}")

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)

    # Main priors / data
    p.add_argument("--lambda-w", type=float, default=0.1)
    p.add_argument("--lambda-colin", type=float, default=0.0)
    p.add_argument("--lambda-wav", type=float, default=1.0)
    p.add_argument("--lambda-mel", type=float, default=1.0)

    # Jacobian
    p.add_argument("--include-jacobian", action="store_true")
    p.add_argument("--lambda-logdet", type=float, default=1e-6)
    p.add_argument("--allow-unsupported-jacobian", action="store_true")

    # Mel
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=80)

    # Latent trajectory: weak by default
    p.add_argument("--lambda-z0", type=float, default=1e-3, help="Latent energy term.")
    p.add_argument("--lambda-z1", type=float, default=1e-3, help="Latent 1st-order smoothness.")
    p.add_argument("--lambda-z2", type=float, default=1e-4, help="Latent 2nd-order smoothness.")
    p.add_argument("--enable-latent-second-order", action="store_true")

    # x-space trajectory: stronger by default
    p.add_argument("--lambda-x0", type=float, default=0.0, help="Waveform energy term.")
    p.add_argument("--lambda-x1", type=float, default=1e-2, help="Waveform 1st-order smoothness.")
    p.add_argument("--lambda-x2", type=float, default=1e-3, help="Waveform 2nd-order smoothness.")
    p.add_argument("--enable-x-second-order", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    reconstruct(parse_args())