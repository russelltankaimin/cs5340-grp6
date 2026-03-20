"""
File: corruptions/additive.py
Description: Fully differentiable implementations of additive noise models.
Includes Gaussian, Uniform, Pink, Brown, Hum, Salt & Pepper, and Impulsive noises.
"""

import torch
import math
from .registry import register_corruption

def _scale_noise_for_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Scales a noise tensor to meet a target Signal-to-Noise Ratio (SNR)."""
    clean_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-12)
    noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-12)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    return noise * (target_noise_rms / noise_rms)

@register_corruption("gaussian")
def gaussian_noise(waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    """Additive White Gaussian Noise (AWGN)."""
    noise = torch.randn_like(waveform)
    return waveform + _scale_noise_for_snr(waveform, noise, snr_db)

@register_corruption("uniform")
def uniform_noise(waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    """Uniformly distributed white noise."""
    noise = (torch.rand_like(waveform) * 2.0) - 1.0
    return waveform + _scale_noise_for_snr(waveform, noise, snr_db)

@register_corruption("pink")
def pink_noise(waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    """Generates pink noise (1/f) via frequency domain shaping."""
    n = waveform.shape[-1]
    white_noise = torch.randn_like(waveform)
    
    X = torch.fft.rfft(white_noise)
    freqs = torch.fft.rfftfreq(n, device=waveform.device)
    
    shaping = torch.ones_like(freqs)
    shaping[1:] = 1.0 / torch.sqrt(freqs[1:])
    
    X_pink = X * shaping
    noise = torch.fft.irfft(X_pink, n=n)
    noise = noise - torch.mean(noise)
    
    return waveform + _scale_noise_for_snr(waveform, noise, snr_db)

@register_corruption("brown")
def brown_noise(waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    """Generates brownian noise via cumulative sum of Gaussian noise."""
    noise = torch.cumsum(torch.randn_like(waveform), dim=-1)
    noise = noise - torch.mean(noise)
    return waveform + _scale_noise_for_snr(waveform, noise, snr_db)

@register_corruption("hum")
def hum_noise(waveform: torch.Tensor, snr_db: float = 10.0, hum_freq: float = 50.0, sr: int = 44100) -> torch.Tensor:
    """Mains electrical hum (base frequency + harmonics)."""
    t = torch.arange(waveform.shape[-1], device=waveform.device, dtype=torch.float32) / sr
    noise = (
        torch.sin(2 * math.pi * hum_freq * t) +
        0.5 * torch.sin(2 * math.pi * 2 * hum_freq * t) +
        0.25 * torch.sin(2 * math.pi * 3 * hum_freq * t)
    )
    noise = noise.unsqueeze(0).unsqueeze(0).expand_as(waveform)
    noise = noise - torch.mean(noise)
    return waveform + _scale_noise_for_snr(waveform, noise, snr_db)

@register_corruption("salt_pepper")
def salt_pepper_noise(waveform: torch.Tensor, noise_level: float = 0.05) -> torch.Tensor:
    """Random sparse maximal spikes (-1 or 1). Differentiable because it's additive."""
    mask = (torch.rand_like(waveform) < noise_level).float()
    signs = torch.sign(torch.randn_like(waveform))
    noise = mask * signs
    return waveform + noise

@register_corruption("impulsive")
def impulsive_noise(waveform: torch.Tensor, noise_level: float = 0.02) -> torch.Tensor:
    """Occasional large Gaussian spikes."""
    mask = (torch.rand_like(waveform) < noise_level).float()
    spikes = torch.randn_like(waveform) * 2.0  # High variance
    return waveform + (mask * spikes)