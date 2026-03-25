"""
File: scripts/corrupt_dataset.py
Description: Generates the corrupted datasets for evaluation. Supports region selection,
randomised segments, crossfading, and metadata extraction to match the original CLI tools.
"""

import argparse
import json
import torch
from pathlib import Path
from corruptions.registry import CORRUPTION_REGISTRY
from utils.audio_io import load_audio, save_audio
from utils.regions import (
    build_regions, 
    apply_corruption_to_regions,
    write_metadata_txt,
    write_regions_csv
)

def main(args):
    device = "cpu"  # Data generation can run locally
    
    # Set manual seed for reproducibility of random regions
    torch.manual_seed(args.seed)
    
    # 1. Load Audio
    clean_audio = load_audio(args.input, target_sr=args.sr, device=device)
    total_samples = clean_audio.shape[-1]
    
    # 2. Extract Corruption Model
    if args.corruption not in CORRUPTION_REGISTRY:
        raise ValueError(f"Corruption '{args.corruption}' not found in registry.")
        
    corrupt_fn = CORRUPTION_REGISTRY[args.corruption]
    kwargs = json.loads(args.corruption_kwargs)
    
    # Bind kwargs to the function
    def bound_corrupt_fn(segment: torch.Tensor) -> torch.Tensor:
        return corrupt_fn(segment, **kwargs)

    # 3. Apply region logic
    regions = build_regions(
        region_mode=args.region_mode,
        total_samples=total_samples,
        sr=args.sr,
        ranges_text=args.ranges,
        num_segments=args.num_segments,
        segment_duration=args.segment_duration,
        min_segment_duration=args.min_segment_duration,
        max_segment_duration=args.max_segment_duration,
        allow_overlap=args.allow_overlap,
    )
    
    if not regions:
        raise ValueError("No valid regions were selected for corruption.")
    
    corrupted_audio, region_metadata = apply_corruption_to_regions(
        clean=clean_audio,
        sr=args.sr,
        regions=regions,
        corrupt_fn=bound_corrupt_fn,
        fade_ms=args.fade_ms
    )
    
    # 4. Save Audio
    save_audio(args.output, corrupted_audio, sr=args.sr)
    print(f"Corrupted file saved to {args.output}")
    
    # 5. Save Metadata Logs
    base = Path(args.output)
    txt_path = str(base.with_suffix(".txt"))
    csv_path = str(base.with_name(base.stem + "_regions.csv"))

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "sample_rate": args.sr,
        "num_samples": total_samples,
        "duration_sec": total_samples / args.sr,
        "corruption": args.corruption,
        "corruption_kwargs": args.corruption_kwargs,
        "region_mode": args.region_mode,
        "num_regions": len(region_metadata),
        "fade_ms": args.fade_ms,
        "seed": args.seed,
        "regions": region_metadata,
    }

    write_metadata_txt(txt_path, metadata)
    write_regions_csv(csv_path, region_metadata)
    
    print(f"Saved metadata: {txt_path}")
    print(f"Saved region CSV: {csv_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    
    # I/O and Corruption settings
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--corruption", type=str, required=True, help=f"Corruption type. Available: {list(CORRUPTION_REGISTRY.keys())}")
    p.add_argument("--corruption-kwargs", type=str, default="{}")
    p.add_argument("--seed", type=int, default=42)
    
    # Region Selection settings
    p.add_argument("--region-mode", type=str, default="full", choices=["full", "time_ranges", "random_segments"])
    p.add_argument("--ranges", type=str, default="", help="Format: 0.5:1.2,3.0:4.0")
    p.add_argument("--num-segments", type=int, default=3)
    p.add_argument("--segment-duration", type=float, default=None)
    p.add_argument("--min-segment-duration", type=float, default=0.25)
    p.add_argument("--max-segment-duration", type=float, default=1.0)
    p.add_argument("--allow-overlap", action="store_true")
    p.add_argument("--fade-ms", type=float, default=10.0)
    
    main(p.parse_args())