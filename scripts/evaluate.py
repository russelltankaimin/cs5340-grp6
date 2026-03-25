"""
File: scripts/evaluate.py
Description: CLI script to evaluate a reconstructed audio file against its clean reference.
It computes both standard and perceptual metrics, outputs a comparative table, 
and generates visualisations of the spectrograms and performance improvements.
"""

import argparse
import json
import torch
from pathlib import Path

from utils.audio_io import load_audio
from utils.metrics import evaluate_all
from utils.visualise import plot_spectrograms, plot_metrics_comparison

def main(args):
    # Dynamically allocate to GPU to accelerate mathematical operations if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing evaluation on: {device}")
    
    print("Loading audio files...")
    clean = load_audio(args.clean, target_sr=args.sr, device=device)
    corrupted = load_audio(args.corrupted, target_sr=args.sr, device=device)
    recon = load_audio(args.recon, target_sr=args.sr, device=device)
    
    # Ensure all tensors strictly match in temporal length to prevent broadcasting errors
    min_len = min(clean.shape[-1], corrupted.shape[-1], recon.shape[-1])
    clean = clean[..., :min_len]
    corrupted = corrupted[..., :min_len]
    recon = recon[..., :min_len]

    print("Computing baseline metrics (Corrupted vs Clean)...")
    baseline_metrics = evaluate_all(clean, corrupted, args.sr)
    
    print("Computing reconstruction metrics (Reconstructed vs Clean)...")
    recon_metrics = evaluate_all(clean, recon, args.sr)
    
    results = {
        "baseline_corrupted": baseline_metrics,
        "reconstructed": recon_metrics
    }
    
    # Prepare the output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the raw numerical data to JSON
    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Output a formatted CLI table for immediate review
    print("\nResults:")
    print(f"{'Metric':<10} | {'Baseline':<12} | {'Reconstructed':<12}")
    print("-" * 40)
    for k in baseline_metrics.keys():
        print(f"{k:<10} | {baseline_metrics[k]:<12.4f} | {recon_metrics[k]:<12.4f}")

    print("\nGenerating visualisations...")
    # Generate and save the graphical artefacts
    plot_spectrograms(
        clean, 
        corrupted, 
        recon, 
        args.sr, 
        str(out_dir / "spectrogram_comparison.png")
    )
    
    plot_metrics_comparison(
        baseline_metrics, 
        recon_metrics, 
        str(out_dir / "metrics_comparison.png")
    )
    
    print(f"Evaluation complete. Artefacts saved to {args.output_dir}/")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate Audio Reconstruction Fidelity")
    p.add_argument("--clean", type=str, required=True, help="Path to the original clean audio")
    p.add_argument("--corrupted", type=str, required=True, help="Path to the corrupted audio input")
    p.add_argument("--recon", type=str, required=True, help="Path to the reconstructed audio output")
    p.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save graphs and JSON")
    p.add_argument("--sr", type=int, default=44100, help="Target sample rate for evaluation")
    
    main(p.parse_args())