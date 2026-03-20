"""
File: corruptions/nonlinear.py
Description: Contains nonlinear, time-domain distortion functions.
Operations like quantisation (Bit Crushing) use Straight-Through Estimators, 
and time-warping (Tape Flutter) uses grid sampling to ensure gradient flow.
"""

import torch
import math
import torch.nn.functional as F
from .registry import register_corruption

@register_corruption("soft_clip")
def soft_clip_distortion(waveform: torch.Tensor, drive: float = 15.0) -> torch.Tensor:
    """Tanh soft-clipping distortion."""
    return torch.tanh(waveform * drive)

@register_corruption("sinusoidal")
def sinusoidal_noise(waveform: torch.Tensor, noise_level: float = 0.05, num_components: int = 64) -> torch.Tensor:
    """Injects deterministic pseudo-noise built from a sum of irrational sinusoids."""
    num_samples = waveform.shape[-1]
    t = torch.arange(num_samples, device=waveform.device, dtype=waveform.dtype)
    noise = torch.zeros_like(waveform)
    
    for i in range(1, num_components + 1):
        freq = i * math.sqrt(2) * math.pi * (1 + i * 0.618033)
        phase = i * i * 1.3579
        noise = noise + torch.sin(freq * t / num_samples * 2 * math.pi + phase)
        
    noise = noise / noise.std()
    return waveform + (noise_level * noise)

@register_corruption("bit_crush")
def bit_crush_corruption(waveform: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Applies bit-depth reduction using a Straight-Through Estimator (STE).
    This ensures that gradients pass through the mathematically non-differentiable quantisation step.
    """
    if bits >= 32:
        return waveform
        
    q = 2 ** (bits - 1)
    
    # Non-differentiable mathematical quantisation
    quantised = torch.round(waveform * q) / q
    
    # Straight-Through Estimator (STE)
    # Forward pass evaluates to `quantised`. Backward pass ignores the bracket.
    ste_quantised = waveform + (quantised - waveform).detach()
    
    return ste_quantised

@register_corruption("tape_and_flutter")
def tape_and_flutter(waveform: torch.Tensor, sr: int = 44100, wow_depth: float = 0.001, wow_freq: float = 1.0, flutter_depth: float = 0.0005, flutter_freq: float = 15.0) -> torch.Tensor:
    """
    Simulates analog tape wow (low-frequency pitch drift) and flutter (high-frequency wobble).
    Uses fully differentiable grid sampling to warp the time domain.
    """
    B, C, T = waveform.shape
    t = torch.arange(T, device=waveform.device, dtype=torch.float32) / sr
    
    # Calculate time delay modulation in seconds
    delay = (wow_depth * torch.sin(2 * math.pi * wow_freq * t) + 
             flutter_depth * torch.sin(2 * math.pi * flutter_freq * t))
    
    # Create base reading grid [-1, 1]
    base_grid = torch.linspace(-1, 1, T, device=waveform.device)
    
    # Convert time delay to grid coordinate shifts
    shift = delay * sr * 2.0 / T
    grid = base_grid + shift
    
    # Format grid for PyTorch's grid_sample: (B, H, W, 2)
    grid = grid.view(1, 1, T, 1)
    grid = torch.cat([grid, torch.zeros_like(grid)], dim=-1) # Add dummy Y dimension
    grid = grid.expand(B, -1, -1, -1)
    
    # Reshape waveform to act as an image (B, C, H, W)
    wav_reshaped = waveform.unsqueeze(2) 
    
    # Resample audio using bilinear interpolation
    resampled = F.grid_sample(wav_reshaped, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    
    return resampled.squeeze(2)