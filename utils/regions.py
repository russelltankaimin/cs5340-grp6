"""
File: utils/regions.py
Description: Logic for applying differentiable corruptions to specific time segments 
of an audio file with crossfading to prevent clipping artefacts.
"""

import torch
from typing import Callable, Sequence, Tuple

def apply_corruption_to_regions(
    clean: torch.Tensor,
    sr: int,
    regions: Sequence[Tuple[int, int]],
    corrupt_fn: Callable[[torch.Tensor], torch.Tensor],
    fade_ms: float = 10.0,
) -> torch.Tensor:
    """
    Applies a corruption function to targeted regions and blends them using a linear crossfade.
    
    Args:
        clean (torch.Tensor): The original audio tensor (1, C, T).
        sr (int): Sample rate.
        regions (Sequence): List of (start_sample, end_sample) tuples.
        corrupt_fn (Callable): The differentiable PyTorch corruption function.
        fade_ms (float): Crossfade duration in milliseconds.
        
    Returns:
        torch.Tensor: The composite corrupted audio tensor.
    """
    out = clean.clone()
    fade_samples = max(0, int(round(fade_ms * sr / 1000.0)))
    
    for start, end in regions:
        if end <= start:
            continue
            
        segment = clean[..., start:end]
        corrupted_segment = corrupt_fn(segment)
        
        # Build the crossfade blend mask
        blend = torch.ones(segment.shape[-1], device=clean.device, dtype=clean.dtype)
        f = min(fade_samples, segment.shape[-1] // 2)
        
        if f > 0:
            ramp = torch.linspace(0.0, 1.0, steps=f, device=clean.device, dtype=clean.dtype)
            blend[:f] = ramp
            blend[-f:] = torch.flip(ramp, dims=[0])
            
        # Apply the blend
        out[..., start:end] = (1.0 - blend) * clean[..., start:end] + blend * corrupted_segment
        
    return out