"""
File: scripts/corrupt_dataset.py
Description: Generates the corrupted datasets for evaluation. By loading the PyTorch 
corruption functions from `corruptions/registry.py`, it guarantees that the forward model 
used to break the audio is exactly identical to the one the optimizer attempts to invert.
"""

import argparse
import json
import torch
from corruptions.registry import CORRUPTION_REGISTRY
from utils.audio_io import load_audio, save_audio
from utils.regions import apply_corruption_to_regions

def main(args):
    device = "cpu"  # Data generation can run locally
    
    # 1. Load Audio
    clean_audio = load_audio(args.input, target_sr=args.sr, device=device)
    
    # 2. Extract Corruption Model
    if args.corruption not in CORRUPTION_REGISTRY:
        raise ValueError(f"Corruption '{args.corruption}' not found in registry.")
        
    corrupt_fn = CORRUPTION_REGISTRY[args.corruption]
    kwargs = json.loads(args.corruption_kwargs)
    
    # Bind kwargs to the function
    def bound_corrupt_fn(segment: torch.Tensor) -> torch.Tensor:
        return corrupt_fn(segment, **kwargs)

    # 3. Apply region logic
    # Assume args.regions is parsed as a list of tuples e.g. [(start, end)]
    # For a full file, the region is simply the entire length
    regions = [(0, clean_audio.shape[-1])] 
    
    corrupted_audio = apply_corruption_to_regions(
        clean=clean_audio,
        sr=args.sr,
        regions=regions,
        corrupt_fn=bound_corrupt_fn,
        fade_ms=args.fade_ms
    )
    
    # 4. Save
    save_audio(args.output, corrupted_audio, sr=args.sr)
    print(f"Corrupted file saved to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--corruption", type=str, required=True)
    p.add_argument("--corruption-kwargs", type=str, default="{}")
    p.add_argument("--fade-ms", type=float, default=10.0)
    
    main(p.parse_args())