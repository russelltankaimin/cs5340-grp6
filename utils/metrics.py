"""
File: utils/metrics.py
Description: Mathematical implementations of standard and perceptual audio evaluation metrics.
Supports dynamic CPU/GPU execution based on input tensor allocation.
"""

import torch
import torchaudio
import numpy as np
from pesq import pesq
from pystoi import stoi

def _align_tensors(clean: torch.Tensor, recon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensures both tensors reside on the exact same computational device."""
    if clean.device != recon.device:
        recon = recon.to(clean.device)
    return clean, recon

def calculate_mae(clean: torch.Tensor, recon: torch.Tensor) -> float:
    """Calculates Mean Absolute Error."""
    clean, recon = _align_tensors(clean, recon)
    return torch.nn.functional.l1_loss(clean, recon).item()

def calculate_snr(clean: torch.Tensor, recon: torch.Tensor, eps: float = 1e-8) -> float:
    """Calculates Signal-to-Noise Ratio (SNR) in dB."""
    clean, recon = _align_tensors(clean, recon)
    noise = clean - recon
    signal_power = torch.sum(clean ** 2)
    noise_power = torch.sum(noise ** 2)
    return (10 * torch.log10(signal_power / (noise_power + eps))).item()

def calculate_sisdr(clean: torch.Tensor, recon: torch.Tensor, eps: float = 1e-8) -> float:
    """Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    clean, recon = _align_tensors(clean, recon)
    clean = clean - torch.mean(clean)
    recon = recon - torch.mean(recon)
    
    # Project recon onto clean
    alpha = torch.sum(recon * clean) / (torch.sum(clean ** 2) + eps)
    target = alpha * clean
    residual = recon - target
    
    target_power = torch.sum(target ** 2)
    residual_power = torch.sum(residual ** 2)
    return (10 * torch.log10(target_power / (residual_power + eps))).item()

def calculate_lsd(clean: torch.Tensor, recon: torch.Tensor, n_fft: int = 2048, eps: float = 1e-8) -> float:
    """Calculates Log-Spectral Distance (LSD)."""
    clean, recon = _align_tensors(clean, recon)
    
    # Convert stereo to mono for consistent STFT evaluation
    clean_mono = clean.mean(dim=1) if clean.dim() > 1 else clean
    recon_mono = recon.mean(dim=1) if recon.dim() > 1 else recon
    
    # Explicitly allocate the window to the correct device
    window = torch.hann_window(n_fft, device=clean.device)
    
    X_clean = torch.stft(clean_mono.squeeze(), n_fft=n_fft, window=window, return_complex=True)
    X_recon = torch.stft(recon_mono.squeeze(), n_fft=n_fft, window=window, return_complex=True)
    
    mag_clean = torch.abs(X_clean) ** 2
    mag_recon = torch.abs(X_recon) ** 2
    
    log_ratio = 10 * torch.log10((mag_clean + eps) / (mag_recon + eps))
    lsd = torch.mean(torch.sqrt(torch.mean(log_ratio ** 2, dim=0)))
    return lsd.item()

def calculate_pesq_stoi(clean: torch.Tensor, recon: torch.Tensor, sr: int) -> tuple[float, float]:
    """
    Calculates PESQ and STOI. 
    Resampling occurs on the target device (e.g., GPU), but the final 
    algorithmic execution must transition to CPU for NumPy compatibility.
    """
    clean, recon = _align_tensors(clean, recon)
    
    # PESQ and STOI strictly require 1D NumPy arrays (mono audio).
    # We project the stereo channels to mono via averaging, whilst keeping dimensions intact for torchaudio.
    clean_mono = clean.mean(dim=1, keepdim=True) if clean.dim() == 3 else clean
    recon_mono = recon.mean(dim=1, keepdim=True) if recon.dim() == 3 else recon
    
    # Squeeze completely to achieve the 1D shape (T,)
    clean_np = clean_mono.squeeze().detach().cpu().numpy()
    recon_np = recon_mono.squeeze().detach().cpu().numpy()
    
    stoi_score = stoi(clean_np, recon_np, sr, extended=False)
    
    # PESQ strictly mandates 16000 Hz (wideband) or 8000 Hz (narrowband).
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(clean.device)
        # Torchaudio resampler expects (..., time), so we pass the 2D tensor and squeeze the output
        clean_16k = resampler(clean_mono).squeeze().detach().cpu().numpy()
        recon_16k = resampler(recon_mono).squeeze().detach().cpu().numpy()
        calc_sr = 16000
    else:
        clean_16k, recon_16k = clean_np, recon_np
        calc_sr = sr
        
    try:
        pesq_score = pesq(calc_sr, clean_16k, recon_16k, 'wb')
    except Exception:
        # Fallback if audio is pure silence or causes a mathematical exception within PESQ
        pesq_score = 0.0 
        
    return pesq_score, stoi_score

def evaluate_all(clean: torch.Tensor, evaluated: torch.Tensor, sr: int) -> dict:
    """Returns a dictionary of all computed metrics, executing on the device of the input tensors."""
    return {
        "MAE": calculate_mae(clean, evaluated),
        "SNR": calculate_snr(clean, evaluated),
        "SI-SDR": calculate_sisdr(clean, evaluated),
        "LSD": calculate_lsd(clean, evaluated),
        "PESQ": calculate_pesq_stoi(clean, evaluated, sr)[0],
        "STOI": calculate_pesq_stoi(clean, evaluated, sr)[1]
    }