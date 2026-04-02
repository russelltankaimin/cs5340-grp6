"""
pipeline.py — End-to-end Bayesian audio reconstruction pipeline.

Takes an arbitrary-length audio file, slices it into 5-second clips,
applies corruption(s), reconstructs via exp_v1 and exp_v2, computes
metrics, generates visualizations, and saves all artefacts to a
timestamped output directory.

Usage
-----
    # Run all corruption types (default):
    python pipeline.py --audio-path data/sample.wav

    # Run a single corruption type:
    python pipeline.py --audio-path data/sample.wav --corruption soft_clip

    # Override per-corruption kwargs:
    python pipeline.py --audio-path data/sample.wav \\
        --corruption-kwargs '{"soft_clip": {"drive": 20.0}, "sinusoidal": {"noise_level": 0.05}}'

Output
------
    results/<YYYYMMDD_HHMMSS>_<audio_stem>/
        latent_stats.json
        summary.json
        clip_00/
            clean.wav
            soft_clip/
                corrupted.wav
                reconstructed_v1.wav
                reconstructed_v2.wav
                metrics_v1.json
                metrics_v2.json
                spectrogram_v1.png
                spectrogram_v2.png
                metrics_comparison_v1.png
                metrics_comparison_v2.png
            sinusoidal/   ...
            tape_wow_flutter/  ...
        clip_01/  ...
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from corruptors.waveform_sinus_dist import sinusoidal_noise

# ── Corruption functions ──────────────────────────────────────────────────────
from corruptors.waveform_soft_clip_dist import soft_clip_distortion
from corruptors.waveform_tap_n_flutter import tape_wow_flutter
from utils.audio_io import load_audio
from utils.compute_stats import (
    compute_latent_stats,
    load_model,
    preprocess_audio,
    split_clips,
)

# ── Reused utilities ──────────────────────────────────────────────────────────
from utils.extract import extract_clip
from utils.metrics import evaluate_all
from utils.visualise import plot_metrics_comparison, plot_spectrograms

CORRUPTION_REGISTRY = {
    "soft_clip": soft_clip_distortion,
    "sinusoidal": sinusoidal_noise,
    "tape_wow_flutter": tape_wow_flutter,
}

DEFAULT_CORRUPTION_KWARGS = {
    "soft_clip": {"drive": 15.0},
    "sinusoidal": {"noise_level": 0.1},
    "tape_wow_flutter": {},
}

# ── Experiment reconstruction functions ───────────────────────────────────────
from experiments.exp_v1 import reconstruct as reconstruct_v1
from experiments.exp_v2 import reconstruct as reconstruct_v2

DEFAULT_VAE_CKPT = os.path.join(".", "vae_ckpt", "ear_vae_44k.pyt")
DEFAULT_VAE_CONFIG = os.path.join(".", "vae_ckpt", "model_config.json")


# =============================================================================
# Helpers
# =============================================================================


def build_reconstruct_args(
    input_path: str,
    output_path: str,
    prior_stats_path: str,
    corruption_name: str,
    corruption_kwargs: dict,
    pipeline_args: argparse.Namespace,
) -> argparse.Namespace:
    """Build the argparse.Namespace expected by reconstruct() in exp_v1/v2."""
    return argparse.Namespace(
        input=input_path,
        output=output_path,
        vae_checkpoint=pipeline_args.vae_checkpoint,
        vae_config=pipeline_args.vae_config,
        prior_stats=prior_stats_path,
        K=pipeline_args.K,
        corruption=corruption_name,
        corruption_kwargs=json.dumps(corruption_kwargs),
        steps=pipeline_args.steps,
        lr=pipeline_args.lr,
        log_every=pipeline_args.log_every,
        lambda_w=pipeline_args.lambda_w,
        lambda_colin=pipeline_args.lambda_colin,
        lambda_wav=pipeline_args.lambda_wav,
        lambda_mel=pipeline_args.lambda_mel,
        n_fft=pipeline_args.n_fft,
        hop_length=pipeline_args.hop_length,
        n_mels=pipeline_args.n_mels,
        lambda_0=pipeline_args.lambda_0,
        lambda_1=pipeline_args.lambda_1,
        lambda_2=pipeline_args.lambda_2,
    )


# =============================================================================
# Main pipeline
# =============================================================================


def run_pipeline(args: argparse.Namespace) -> None:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── 1. Setup output directory ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_stem = Path(args.audio_path).stem
    run_dir = Path(args.output_dir) / f"{timestamp}_{audio_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_dir}")

    # ── 2. Determine which corruptions to run ─────────────────────────────────
    corruptions_to_run = (
        [args.corruption] if args.corruption else list(CORRUPTION_REGISTRY.keys())
    )
    per_corruption_kwargs = {k: dict(v) for k, v in DEFAULT_CORRUPTION_KWARGS.items()}
    if args.corruption_kwargs:
        user_kwargs = json.loads(args.corruption_kwargs)
        per_corruption_kwargs.update(user_kwargs)
    print(f"Corruptions: {corruptions_to_run}")

    # ── 3. Compute prior stats from full audio ────────────────────────────────
    stats_path = str(run_dir / "latent_stats.json")
    print("\n=== Computing latent prior statistics ===")
    model = load_model(device)
    waveform, sr = preprocess_audio(args.audio_path, target_sr=44100, device=device)
    clips_for_stats, num_clips = split_clips(waveform, sr, args.clip_seconds, fill=args.fill_last_clip)
    mean, stds = compute_latent_stats(model, clips_for_stats)
    with open(stats_path, "w") as f:
        json.dump(
            {
                "mean": mean.detach().cpu().tolist(),
                "stds": stds.detach().cpu().tolist(),
            },
            f,
        )
    print(f"Saved latent stats → {stats_path}")

    # ── 4. Determine number of clips ─────────────────────────────────────────
    decoder = AudioDecoder(args.audio_path)
    info = decoder.metadata

    total_duration = info.duration_seconds
    print(
        f"\nAudio duration: {total_duration:.2f}s → {num_clips} clips of {args.clip_seconds}s"
    )

    # ── 5. Process each clip ──────────────────────────────────────────────────
    summary = {
        "audio_path": args.audio_path,
        "num_clips": num_clips,
        "clip_seconds": args.clip_seconds,
        "corruptions": corruptions_to_run,
        "clips": {},
    }

    for clip_idx in range(num_clips):
        start_time = clip_idx * args.clip_seconds
        clip_label = f"clip_{clip_idx:02d}"
        clip_dir = run_dir / clip_label
        clip_dir.mkdir(exist_ok=True)

        print(f"\n{'=' * 60}")
        print(
            f"Processing {clip_label}  (t={start_time:.1f}s - {start_time + args.clip_seconds:.1f}s)"
        )
        print(f"{'=' * 60}")

        # Extract and save clean clip via extract.py
        clean_clip, clip_sr = extract_clip(
            args.audio_path, start_time, args.clip_seconds, fill=args.fill_last_clip
        )
        clean_path = str(clip_dir / f"clean_{clip_idx:02d}.wav")
        encoder_clean = AudioEncoder(clean_clip, sample_rate=clip_sr)
        encoder_clean.to_file(clean_path)

        summary["clips"][clip_label] = {}

        for corruption_name in corruptions_to_run:
            corruption_fn = CORRUPTION_REGISTRY[corruption_name]
            ckwargs = per_corruption_kwargs.get(corruption_name, {})

            corr_dir = clip_dir / corruption_name
            corr_dir.mkdir(exist_ok=True)

            print(f"\n--- {corruption_name}  kwargs={ckwargs} ---")

            # a) Apply corruption and save
            corrupted_clip = (
                corruption_fn(clean_clip.to(device), **ckwargs).detach().cpu()
            )
            corrupted_path = str(corr_dir / f"corrupted_{corruption_name}_{clip_idx:02d}.wav")
            encoder_corr = AudioEncoder(corrupted_clip, sample_rate=clip_sr)
            encoder_corr.to_file(corrupted_path)

            # b) Reconstruct with exp_v1 and exp_v2
            recon_v1_path = str(corr_dir / f"reconstructed_v1_{corruption_name}_{clip_idx:02d}.wav")
            recon_v2_path = str(corr_dir / f"reconstructed_v2_{corruption_name}_{clip_idx:02d}.wav")

            print("  [exp_v1] Reconstructing...")
            reconstruct_v1(
                build_reconstruct_args(
                    corrupted_path,
                    recon_v1_path,
                    stats_path,
                    corruption_name,
                    ckwargs,
                    args,
                )
            )

            print("  [exp_v2] Reconstructing...")
            reconstruct_v2(
                build_reconstruct_args(
                    corrupted_path,
                    recon_v2_path,
                    stats_path,
                    corruption_name,
                    ckwargs,
                    args,
                )
            )

            # c) Compute metrics
            print("  Computing metrics...")
            clean_t = load_audio(clean_path, target_sr=44100, device=device)
            corr_t = load_audio(corrupted_path, target_sr=44100, device=device)
            recon_v1_t = load_audio(recon_v1_path, target_sr=44100, device=device)
            recon_v2_t = load_audio(recon_v2_path, target_sr=44100, device=device)

            # Trim all to shortest length to prevent shape mismatches
            min_len = min(
                clean_t.shape[-1],
                corr_t.shape[-1],
                recon_v1_t.shape[-1],
                recon_v2_t.shape[-1],
            )
            clean_t = clean_t[..., :min_len]
            corr_t = corr_t[..., :min_len]
            recon_v1_t = recon_v1_t[..., :min_len]
            recon_v2_t = recon_v2_t[..., :min_len]

            baseline_metrics = evaluate_all(clean_t, corr_t, sr=44100)
            v1_metrics = evaluate_all(clean_t, recon_v1_t, sr=44100)
            v2_metrics = evaluate_all(clean_t, recon_v2_t, sr=44100)

            with open(corr_dir / f"metrics_v1_{corruption_name}_{clip_idx:02d}.json", "w") as f:
                json.dump(
                    {"baseline": baseline_metrics, "reconstructed": v1_metrics},
                    f,
                    indent=4,
                )
            with open(corr_dir / f"metrics_v2_{corruption_name}_{clip_idx:02d}.json", "w") as f:
                json.dump(
                    {"baseline": baseline_metrics, "reconstructed": v2_metrics},
                    f,
                    indent=4,
                )

            # d) Generate visualizations
            print("  Generating visualizations...")
            plot_spectrograms(
                clean_t,
                corr_t,
                recon_v1_t,
                sr=44100,
                save_path=str(corr_dir / f"spectrogram_v1_{corruption_name}_{clip_idx:02d}.png"),
            )
            plot_spectrograms(
                clean_t,
                corr_t,
                recon_v2_t,
                sr=44100,
                save_path=str(corr_dir / f"spectrogram_v2_{corruption_name}_{clip_idx:02d}.png"),
            )
            plot_metrics_comparison(
                baseline_metrics,
                v1_metrics,
                save_path=str(corr_dir / f"metrics_comparison_v1_{corruption_name}_{clip_idx:02d}.png"),
            )
            plot_metrics_comparison(
                baseline_metrics,
                v2_metrics,
                save_path=str(corr_dir / f"metrics_comparison_v2_{corruption_name}_{clip_idx:02d}.png"),
            )

            summary["clips"][clip_label][corruption_name] = {
                "baseline": baseline_metrics,
                "v1": v1_metrics,
                "v2": v2_metrics,
            }

    # ── 6. Save summary ───────────────────────────────────────────────────────
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete. Results saved to: {run_dir}")
    print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full audio reconstruction pipeline: slice → corrupt → reconstruct → evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / output
    p.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to input audio file of arbitrary length.",
    )
    p.add_argument(
        "--clip-seconds",
        type=float,
        default=5.0,
        help="Duration of each clip in seconds.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root directory for timestamped output folders.",
    )
    p.add_argument(
        "--fill-last-clip",
        action="store_true",
        help=(
            "Whether to fill the last clip with zeros if it is shorter than clip_seconds. "
            "If False, the last partial clip is discarded."
        ),
    )

    # Corruption
    p.add_argument(
        "--corruption",
        type=str,
        default=None,
        choices=list(CORRUPTION_REGISTRY.keys()),
        help="Corruption type to run. Omit to run all three.",
    )
    p.add_argument(
        "--corruption-kwargs",
        type=str,
        default=None,
        help=(
            "JSON dict mapping corruption name → kwargs dict. "
            'E.g. \'{"soft_clip": {"drive": 20.0}}\'. '
            "Unspecified corruptions use defaults."
        ),
    )

    # VAE
    p.add_argument("--vae-checkpoint", type=str, default=DEFAULT_VAE_CKPT)
    p.add_argument("--vae-config", type=str, default=DEFAULT_VAE_CONFIG)

    # Latent
    p.add_argument(
        "--K",
        type=int,
        default=215,
        help="Latent sub-vectors for L_colin (215 for 5s clips at 44.1 kHz).",
    )

    # Optimisation
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)

    # Loss weights (shared by v1 and v2)
    p.add_argument("--lambda-w", type=float, default=1.0)
    p.add_argument("--lambda-colin", type=float, default=1.0)
    p.add_argument("--lambda-wav", type=float, default=1.0)
    p.add_argument("--lambda-mel", type=float, default=1.0)

    # Trajectory prior weights (exp_v2 only)
    p.add_argument(
        "--lambda-0",
        type=float,
        default=1.0,
        help="Zeroth-order trajectory prior weight (exp_v2).",
    )
    p.add_argument(
        "--lambda-1",
        type=float,
        default=0.1,
        help="First-order temporal smoothness weight (exp_v2).",
    )
    p.add_argument(
        "--lambda-2",
        type=float,
        default=0.1,
        help="Second-order temporal smoothness weight (exp_v2).",
    )

    # Mel spectrogram
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=80)

    return p.parse_args()


if __name__ == "__main__":
    try:
        run_pipeline(parse_args())
    except Exception as e:
        print(f"Error occurred: {e}")