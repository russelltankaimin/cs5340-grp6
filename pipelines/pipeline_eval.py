import argparse
import copy
import csv
import json
import os

import torch
import torchaudio
from torchaudio import save_with_torchcodec

# Import the target function and registry directly
from experiments.exp_v1_1 import CORRUPTION_REGISTRY, reconstruct
from utils.metrics import evaluate_all
from utils.compute_stats import preprocess_audio, split_clips

import warnings

# Suppress the specific PyTorch FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*`torch.nn.utils.weight_norm` is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*`torch.backends.cuda.sdp_kernel()` is deprecated.*"
)

# =========================================================================
# PERSISTENCE UTILITY
# =========================================================================

def save_result_row(csv_path: str, row_data: dict):
    """
    Writes a single result row to the CSV immediately. 
    Handles header creation if the file does not exist.
    """
    file_exists = os.path.isfile(csv_path)
    
    # We use 'a' (append) mode to ensure data is added without overwriting
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        
        # Write header only once at the start of the file
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(row_data)

# =========================================================================
# PIPELINE EXECUTION
# =========================================================================

def run_pipeline(args: argparse.Namespace):
    # Dynamically detect available hardware acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pipeline orchestration executing on: {device.upper()}")

    # 1. Process Clean Audio using the provided utility
    clean_waveform, actual_sr = preprocess_audio(args.input, args.target_sr, device)
    
    # 2. Split Audio into Clips using compute_stats utility
    # We use fill=False to ensure only complete segments of clip_seconds are processed
    clips, num_clips = split_clips(
        clean_waveform, 
        args.target_sr, 
        args.clip_seconds, 
        fill=False
    )
    print(f"Total clips to process: {num_clips}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "metrics_report.csv")

    # 3. Iterate over Clips (Row-major traversal)
    for i in range(num_clips):
        print(f"\n==================================================")
        print(f"Processing Clip {i+1}/{num_clips}...")
        print(f"==================================================")
        
        # clips[i] has shape (Channels, Samples)
        c_flat = clips[i]

        # 4. Iterate over Corruptions for the current clip
        for corruption_name in args.corruptions:
            try:
                print(f"  -> Applying Corruption: {corruption_name}")
                spec = CORRUPTION_REGISTRY[corruption_name]
                c_kwargs_dict = args.corruption_kwargs.get(corruption_name, {})
                
                c_out_dir = os.path.join(args.output_dir, corruption_name)
                os.makedirs(c_out_dir, exist_ok=True)

                # Apply corruption natively in memory
                corrupted_clip_flat = spec.fn(c_flat, **c_kwargs_dict)
                
                # Define I/O paths
                clean_path = os.path.join(c_out_dir, f"clip_{i}_clean.wav")
                corr_path = os.path.join(c_out_dir, f"clip_{i}_corrupted.wav")
                recon_path = os.path.join(c_out_dir, f"clip_{i}_recon.wav")

                # Serialise tensors to disk (must be on CPU for torchcodec)
                save_with_torchcodec(clean_path, c_flat.cpu(), args.target_sr)
                save_with_torchcodec(corr_path, corrupted_clip_flat.cpu(), args.target_sr)

                # Reconstruct via exp_v1_1 logic
                clip_args = copy.deepcopy(args)
                clip_args.input = corr_path
                clip_args.output = recon_path
                clip_args.corruption = corruption_name
                clip_args.corruption_kwargs = json.dumps(c_kwargs_dict)

                # Internal optimization loop
                reconstruct(clip_args)

                # Load reconstruction back to device for metric evaluation
                recon_clip, _ = preprocess_audio(recon_path, args.target_sr, device)

                # Compute Perceptual Metrics
                metrics_corr = evaluate_all(c_flat, corrupted_clip_flat, args.target_sr)
                metrics_recon = evaluate_all(c_flat, recon_clip, args.target_sr)
                metrics_clean = evaluate_all(c_flat, c_flat, args.target_sr)

                # Aggregate data for CSV
                row_data = {"clip_index": i, "corruption": corruption_name}
                for key, val in metrics_clean.items():
                    row_data[f"clean_{key}"] = val
                for key, val in metrics_corr.items():
                    row_data[f"corrupted_{key}"] = val
                for key, val in metrics_recon.items():
                    row_data[f"reconstructed_{key}"] = val

                save_result_row(csv_path, row_data)
            except Exception as e:
                print(f"!!! CRITICAL ERROR: Skipping Clip {i} due to decoding failure.")
                print(f"Details: {e}")
                
                # Log this failure to the CSV so you know which data points are missing
                error_row = {
                    "clip_index": i,
                    "corruption": corruption_name,
                    "notes": str(e).split('\n')[0]
                }
                save_result_row(csv_path, error_row)
                continue

    print(f"\nPipeline complete. Metrics saved to: {csv_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Iterative corruption and reconstruction pipeline wrapper.")

    p.add_argument("--input", type=str, required=True, help="Path to clean input audio.")
    p.add_argument("--output-dir", type=str, required=True, help="Parent output directory.")
    p.add_argument("--clip-seconds", type=float, default=5.0, help="Duration to chunk the audio.")
    p.add_argument("--target-sr", type=int, default=44100)
    
    p.add_argument("--corruptions", nargs='+', default=list(CORRUPTION_REGISTRY.keys()), choices=list(CORRUPTION_REGISTRY.keys()), help="List of corruptions to apply.")
    p.add_argument(
        "--corruption_kwargs", 
        type=json.loads, 
        default="{}", 
        help='JSON dictionary mapping corruption names to kwargs. e.g. \'{"soft_clip": {"drive": 15.0}}\''
    )

    # Arguments mirrored directly to satisfy reconstruct()
    p.add_argument("--vae-checkpoint", type=str, default=os.path.join(".", "vae_ckpt", "ear_vae_44k.pyt"))
    p.add_argument("--vae-config", type=str, default=os.path.join(".", "vae_ckpt", "model_config.json"))
    p.add_argument("--prior-stats", type=str, required=True)
    p.add_argument("--K", type=int, default=215)

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--lambda-w", type=float, default=0.1)
    p.add_argument("--lambda-colin", type=float, default=0.0)
    p.add_argument("--lambda-wav", type=float, default=1.0)
    p.add_argument("--lambda-mel", type=float, default=1.0)
    
    p.add_argument("--include-jacobian", action="store_true")
    p.add_argument("--lambda-logdet", type=float, default=1e-6)
    p.add_argument("--allow-unsupported_jacobian", action="store_true")

    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=80)

    p.add_argument("--lambda-z0", type=float, default=1e-3)
    p.add_argument("--lambda-z1", type=float, default=1e-3)
    p.add_argument("--lambda-z2", type=float, default=1e-4)
    p.add_argument("--enable-latent-second-order", action="store_true")

    p.add_argument("--lambda-x0", type=float, default=0.0)
    p.add_argument("--lambda-x1", type=float, default=1e-2)
    p.add_argument("--lambda-x2", type=float, default=1e-3)
    p.add_argument("--enable-x-second-order", action="store_true")

    args = p.parse_args()
    run_pipeline(args)