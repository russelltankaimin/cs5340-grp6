import argparse
import json
import os

import torch
import torchaudio
from tqdm import tqdm

from ear_vae.ear_vae import EAR_VAE

MODEL_PATH = os.path.join(".", "vae_ckpt", "ear_vae_44k.pyt")
CONFIG_PATH = os.path.join(".", "vae_ckpt", "model_config.json")


def load_model(device: str) -> EAR_VAE:
    with open(CONFIG_PATH, "r") as f:
        vae_model_config = json.load(f)

    model = EAR_VAE(model_config=vae_model_config).to(device)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_audio(
    audio_path: str, target_sr: int, device: str
) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(audio_path, backend="ffmpeg")

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr).to(device)
        waveform = resampler(waveform.to(device))
        sr = target_sr

    waveform = waveform.to(device, dtype=torch.float32)

    # Match `vae_sample.py` behavior: use stereo input.
    if waveform.shape[0] == 1:
        waveform = torch.cat([waveform, waveform], dim=0)

    return waveform, sr


def split_clips(waveform: torch.Tensor, sr: int, clip_seconds: float) -> torch.Tensor:
    samples_per_clip = int(sr * clip_seconds)
    total_samples = waveform.shape[1]
    num_clips = total_samples // samples_per_clip

    if num_clips == 0:
        raise ValueError(
            f"Audio too short for T={clip_seconds}s. "
            f"Total duration: {total_samples / sr:.3f}s."
        )

    clipped = waveform[:, : num_clips * samples_per_clip]
    clips = clipped.reshape(waveform.shape[0], num_clips, samples_per_clip).permute(
        1, 0, 2
    )
    return clips


def compute_latent_stats(
    model: EAR_VAE, clips: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = []
    with torch.no_grad():
        for clip in tqdm(clips, desc="Computing latents", total=clips.shape[0]):
            clip_batch = clip.unsqueeze(0)  # (1, channels, samples)
            latent = model.encode(clip_batch)  # expected: (1, 64, latent_duration_dim)
            latents.append(latent)

    all_latents = torch.cat(latents, dim=0)  # (num_clips, 64, latent_duration_dim)

    mean = all_latents.mean(dim=(0, 2))
    stds = all_latents.std(dim=0, unbiased=False)

    return mean, stds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute VAE latent mean/std stats across fixed-length audio sub-clips."
    )
    parser.add_argument(
        "--audio-path", type=str, default=os.path.join("data", "sample_2.wav")
    )
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--output-path", type=str, default="latent_stats.json")
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Audio path: {args.audio_path}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Device: {device}")
    print(f"Clip seconds (T): {args.clip_seconds}")

    model = load_model(device)
    waveform, sr = preprocess_audio(args.audio_path, target_sr=44100, device=device)
    clips = split_clips(waveform, sr, args.clip_seconds)

    print(f"Waveform shape after preprocessing: {waveform.shape}")
    print(f"Number of clips: {clips.shape[0]}")
    print(f"Each clip shape: {clips.shape[1:]}")

    mean, stds = compute_latent_stats(model, clips)

    print(f"Mean shape: {tuple(mean.shape)}")
    print(f"Stds shape: {tuple(stds.shape)}")

    payload = {
        "mean": mean.detach().cpu().tolist(),
        "stds": stds.detach().cpu().tolist(),
    }

    with open(args.output_path, "w") as f:
        json.dump(payload, f)

    print(f"Saved stats to: {args.output_path}")


if __name__ == "__main__":
    main()
