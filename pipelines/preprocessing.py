import os
import argparse
import json
import torch
import torch.nn.functional as F
from torchcodec.decoders import AudioDecoder
from torchaudio import save_with_torchcodec

from utils.compute_stats import load_model, split_clips, preprocess_audio, compute_latent_stats

import warnings

# Suppress the specific PyTorch weight_norm FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*`torch.nn.utils.weight_norm` is deprecated.*"
)

def process_audio(input_path, output_dir, clip_seconds, target_sr=24000, pad_silence=True):
    # Set up GPU acceleration if available for post-decode processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and preprocess audio using the established utility function
    # This ensures correct stereo upmixing and resampling for the VAE
    print(f"Decoding and resampling to {target_sr} Hz...")
    waveform, sr = preprocess_audio(input_path, target_sr, device)

    # 2. Calculate split points based on time
    total_samples = waveform.shape[1]
    total_seconds = total_samples / target_sr

    # Determine the raw 20% duration and force it to be a multiple of clip_seconds
    target_end_seconds = total_seconds * 0.2
    num_clips_end = round(target_end_seconds / clip_seconds)
    adjusted_end_seconds = num_clips_end * clip_seconds

    # Fallback to ensure we don't exceed total length
    if adjusted_end_seconds > total_seconds:
        adjusted_end_seconds = (total_seconds // clip_seconds) * clip_seconds

    split_point_seconds = total_seconds - adjusted_end_seconds
    split_point_samples = int(split_point_seconds * target_sr)

    # Split the waveform
    split_80 = waveform[:, :split_point_samples]
    split_20 = waveform[:, split_point_samples:]

    # 3. Chunk to clip_seconds and apply optional padding
    chunks_80 = split_clips(split_80, target_sr, clip_seconds, fill=pad_silence)[0]

    # 4. Compute latent stats
    model = load_model(device)

    mean_80, std_80 = compute_latent_stats(model, chunks_80)

    # 5. Save the results
    os.makedirs(output_dir, exist_ok=True)

    # Save using the TorchCodec encoding backend via Torchaudio
    if split_80.numel() > 0:
        save_with_torchcodec(os.path.join(output_dir, "split_80.wav"), split_80.cpu(), target_sr)
    
    if split_20.numel() > 0:
        save_with_torchcodec(os.path.join(output_dir, "split_20.wav"), split_20.cpu(), target_sr)
    
    # Save the latent stats
    prior_path = os.path.join(output_dir, "latent_stats.json")
    with open(prior_path, "w") as f:
        json.dump({
            "mean": mean_80.detach().cpu().tolist(),
            "stds": std_80.detach().cpu().tolist()
        }, f, indent=4)

    print("Processing complete.")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split audio files.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the outputs.")
    parser.add_argument("--clip_seconds", type=float, required=True, help="Duration of clips in seconds.")
    parser.add_argument("--target_sr", type=int, default=24000, help="Target sampling rate (default 24000).")
    parser.add_argument("--no_pad", action="store_true", help="Disable silence padding on the final clips.")
    
    args = parser.parse_args()
    
    process_audio(
        input_path=args.input,
        output_dir=args.output,
        clip_seconds=args.clip_seconds,
        target_sr=args.target_sr,
        pad_silence=not args.no_pad
    )