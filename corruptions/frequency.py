"""
File: corruptions/frequency.py
Description: Differentiable frequency-domain corruptions utilizing PyTorch's FFT ops.
Includes FFT masking, Phase/Magnitude noise, Band-limited noise, and Random EQ.
"""

import torch
import torch.nn.functional as F
from .registry import register_corruption
from .additive import _scale_noise_for_snr, gaussian_noise

@register_corruption("fft_corruption")
def fft_corruption(waveform: torch.Tensor, mask_ratio: float = 0.1, noise_std: float = 0.0, phase_noise_std: float = 0.0) -> torch.Tensor:
    """Combined Bin Masking, Magnitude Noise, and Phase Noise."""
    n = waveform.shape[-1]
    X = torch.fft.rfft(waveform)
    magnitudes = torch.abs(X)
    phases = torch.angle(X)
    
    # 1. Bin Masking
    if mask_ratio > 0:
        mask = (torch.rand_like(magnitudes) > mask_ratio).float()
        magnitudes = magnitudes * mask
        
    # 2. Magnitude Noise
    if noise_std > 0:
        avg_mag = torch.mean(magnitudes)
        mag_noise = torch.randn_like(magnitudes) * noise_std * avg_mag
        magnitudes = F.relu(magnitudes + mag_noise) # Prevent negative magnitudes
        
    # 3. Phase Noise
    if phase_noise_std > 0:
        phase_noise = torch.randn_like(phases) * phase_noise_std
        phases = phases + phase_noise
        
    # Reconstruct
    X_new = magnitudes * torch.exp(1j * phases)
    y = torch.fft.irfft(X_new, n=n)
    
    # Energy preservation
    clean_rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-12)
    y_rms = torch.sqrt(torch.mean(y ** 2) + 1e-12)
    return y * (clean_rms / y_rms)

@register_corruption("band_limited")
def band_limited_noise(waveform: torch.Tensor, snr_db: float = 10.0, band_low: float = 300.0, band_high: float = 3000.0, sr: int = 44100) -> torch.Tensor:
    """Generates noise restricted to a specific frequency band using FFT masking."""
    n = waveform.shape[-1]
    noise = torch.randn_like(waveform)
    
    X = torch.fft.rfft(noise)
    freqs = torch.fft.rfftfreq(n, d=1.0/sr, device=waveform.device)
    
    # Create an ideal bandpass mask
    mask = ((freqs >= band_low) & (freqs <= band_high)).float()
    X_band = X * mask
    
    filtered_noise = torch.fft.irfft(X_band, n=n)
    return waveform + _scale_noise_for_snr(waveform, filtered_noise, snr_db)

@register_corruption("random_eq")
def random_eq(waveform: torch.Tensor, bands: int = 12, max_gain_db: float = 8.0, sr: int = 44100) -> torch.Tensor:
    """Applies a smooth, random frequency-response distortion."""
    n = waveform.shape[-1]
    X = torch.fft.rfft(waveform)
    freqs = torch.fft.rfftfreq(n, d=1.0/sr, device=waveform.device)
    
    # Generate random gains for the bands
    random_gains_db = (torch.rand(bands, device=waveform.device) * 2 * max_gain_db) - max_gain_db
    
    # Interpolate gains smoothly across all frequency bins (simplified smoothing)
    random_gains_db = random_gains_db.unsqueeze(0).unsqueeze(-1) # (1, 1, bands)
    random_gains_db = F.interpolate(random_gains_db, size=freqs.shape[0], mode='linear', align_corners=True).squeeze()
    
    gains_lin = 10.0 ** (random_gains_db / 20.0)
    
    y = torch.fft.irfft(X * gains_lin, n=n)
    
    # Energy preservation
    clean_rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-12)
    y_rms = torch.sqrt(torch.mean(y ** 2) + 1e-12)
    return y * (clean_rms / y_rms)

@register_corruption("combo")
def combo_corruption(waveform: torch.Tensor, snr_db: float = 10.0, bands: int = 12, max_gain_db: float = 8.0) -> torch.Tensor:
    """Applies Random EQ followed by Gaussian Noise."""
    eq_waveform = random_eq(waveform, bands, max_gain_db)
    return gaussian_noise(eq_waveform, snr_db)