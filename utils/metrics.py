"""
File: utils/metrics.py
Description: Mathematical implementations of standard and perceptual audio evaluation metrics.
Supports dynamic CPU/GPU execution based on input tensor allocation.
"""

import torch
import torchaudio
import numpy as np
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
from torchmetrics.audio import (
    ComplexScaleInvariantSignalNoiseRatio, 
    ScaleInvariantSignalDistortionRatio, 
    ScaleInvariantSignalNoiseRatio, 
    SignalNoiseRatio, 
    SignalDistortionRatio
)

def _align_tensors(clean: torch.Tensor, recon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensures both tensors reside on the exact same computational device."""
    if clean.device != recon.device:
        recon = recon.to(clean.device)
    return clean, recon

def calculate_mae(clean: torch.Tensor, recon: torch.Tensor) -> float:
    """
    Calculates Mean Absolute Error.
    LOWER is better. (0 is perfect reconstruction).
    """
    clean, recon = _align_tensors(clean, recon)
    return torch.nn.functional.l1_loss(clean, recon).item()

def calculate_snr(clean: torch.Tensor, recon: torch.Tensor) -> float:
    """
    Calculates Signal-to-Noise Ratio (SNR) in dB.
    HIGHER is better.
    """
    clean, recon = _align_tensors(clean, recon)
    sns = SignalNoiseRatio().to(clean.device)
    return sns(recon, clean).item()

def calculate_sdr(clean: torch.Tensor, recon: torch.Tensor) -> float:
    """
    Calculates Signal-to-Distortion Ratio (SDR) in dB.
    HIGHER is better.
    """
    clean, recon = _align_tensors(clean, recon)
    sdr = SignalDistortionRatio().to(clean.device)
    return sdr(recon, clean).item()

def calculate_sisdr(clean: torch.Tensor, recon: torch.Tensor) -> float:
    """
    Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    HIGHER is better.
    """
    clean, recon = _align_tensors(clean, recon)
    si_sdr = ScaleInvariantSignalDistortionRatio().to(clean.device)
    return si_sdr(recon, clean).item()

def calculate_sisnr(clean: torch.Tensor, recon: torch.Tensor) -> float:
    """
    Calculates Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.
    HIGHER is better.
    """
    clean, recon = _align_tensors(clean, recon)
    si_snr = ScaleInvariantSignalNoiseRatio().to(clean.device)
    return si_snr(recon, clean).item()

def calculate_csisnr(clean: torch.Tensor, recon: torch.Tensor, n_fft: int = 512, hop_length: int = 256) -> float:
    """
    Calculates Complex Scale-Invariant Signal-to-Noise Ratio (CSISNR) in dB.
    HIGHER is better.
    """
    clean, recon = _align_tensors(clean, recon)
    
    # 1. Map 3D tensor (Batch, Channels, Time) to 2D (Batch * Channels, Time)
    # If the input is already 2D (Batch, Time), this reshape safely does nothing.
    B, C, T = clean.shape
    clean_2d = clean.reshape(B * C, T)
    recon_2d = recon.reshape(B * C, T)
    
    # 2. Compute the Short-Time Fourier Transform (STFT)
    clean_stft = torch.stft(
        clean_2d, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        return_complex=True, 
        pad_mode='constant'
    )
    recon_stft = torch.stft(
        recon_2d, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        return_complex=True, 
        pad_mode='constant'
    )
    
    # 3. Apply inverse mapping back to (Batch, Channels, Freq, Time)
    _, F, M = clean_stft.shape
    clean_stft = clean_stft.reshape(B, C, F, M)
    recon_stft = recon_stft.reshape(B, C, F, M)
    
    # 4. Extract the real and imaginary parts and stack them along a new final dimension
    # This transforms the shape to (Batch, Channels, Freq, Time, 2)
    clean_stft_formatted = torch.stack((clean_stft.real, clean_stft.imag), dim=-1)
    recon_stft_formatted = torch.stack((recon_stft.real, recon_stft.imag), dim=-1)
    
    # 5. Calculate the metric
    csisnr = ComplexScaleInvariantSignalNoiseRatio().to(clean.device)
    return csisnr(recon_stft_formatted, clean_stft_formatted).item()

def calculate_pesq(clean: torch.Tensor, recon: torch.Tensor, sr: int) -> float:
    """
    Calculates Perceptual Evaluation of Speech Quality (PESQ).
    HIGHER is better. (Range: -0.5 to 4.5)
    """
    clean, recon = _align_tensors(clean, recon)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(clean.device)
        clean_16k = resampler(clean)
        recon_16k = resampler(recon)
    else:
        clean_16k = clean
        recon_16k = recon
        
    pesq_metric = PerceptualEvaluationSpeechQuality(16000, 'wb').to(clean.device)
    return pesq_metric(recon_16k, clean_16k).item()

def calculate_stoi(clean: torch.Tensor, recon: torch.Tensor, sr: int) -> float:
    """
    Calculates Short-Time Objective Intelligibility (STOI).
    HIGHER is better. (Range: 0.0 to 1.0)
    """
    clean, recon = _align_tensors(clean, recon)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(clean.device)
        clean_16k = resampler(clean)
        recon_16k = resampler(recon)
    else:
        clean_16k = clean
        recon_16k = recon
        
    stoi_metric = ShortTimeObjectiveIntelligibility(16000, False).to(clean.device)
    return stoi_metric(recon_16k, clean_16k).item()

def calculate_dnsmos(recon: torch.Tensor, sr: int) -> tuple:
    """
    Calculates Deep Noise Suppression Mean Opinion Score (DNSMOS).
    HIGHER is better for all returned values. (Range: 1.0 to 5.0)
    """
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(recon.device)
        recon_16k = resampler(recon)
    else:
        recon_16k = recon
        
    dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(16000, False).to(recon.device)
    
    # DNSMOS returns a dictionary or multiple values depending on the torchmetrics version.
    # Usually, it returns a dict with p808_mos, mos_sig, mos_bak, mos_ovr.
    # Evaluating returns the tensor(s). We leave them as tensors to be parsed in evaluate_all.
    return dnsmos_metric(recon_16k)

def calculate_srmr(recon: torch.Tensor, sr: int) -> float:
    """
    Calculates Speech Reverberation Modulation Energy Ratio (SRMR).
    HIGHER is better.
    """
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(recon.device)
        recon_16k = resampler(recon)
    else:
        recon_16k = recon
        
    srmr_metric = SpeechReverberationModulationEnergyRatio(16000).to(recon.device)
    return srmr_metric(recon_16k).item()

def calculate_nisqa(recon: torch.Tensor, sr: int) -> tuple:
    """
    Calculates Non-Intrusive Speech Quality Assessment (NISQA).
    HIGHER is better for all returned values. (Range: 1.0 to 5.0)
    """
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(recon.device)
        recon_16k = resampler(recon)
    else:
        recon_16k = recon
        
    nisqa_metric = NonIntrusiveSpeechQualityAssessment(16000).to(recon.device)
    return nisqa_metric(recon_16k)

def evaluate_all(clean: torch.Tensor, evaluated: torch.Tensor, sr: int) -> dict:
    """Returns a dictionary of all computed metrics, executing on the device of the input tensors."""
    dnsmos = calculate_dnsmos(evaluated, sr)
    nisqa = calculate_nisqa(evaluated, sr)

    def extract(metric_output, dict_key, tensor_idx):
        if isinstance(metric_output, dict):
            return metric_output[dict_key].item()
        else:
            # If it is a tensor, it may be 1D or 2D (e.g., [1, 4]). flatten() ensures safe indexing.
            if isinstance(metric_output, torch.Tensor):
                metric_output = metric_output.flatten()
            return metric_output[tensor_idx].item()
    
    return {
        # "MAE": calculate_mae(clean, evaluated),
        # "SNR": calculate_snr(clean, evaluated),
        # "SI-SDR": calculate_sisdr(clean, evaluated),
        # "SI-SNR": calculate_sisnr(clean, evaluated),
        # "CSISNR": calculate_csisnr(clean, evaluated),
        # "PESQ": calculate_pesq(clean, evaluated, sr),
        # "STOI": calculate_stoi(clean, evaluated, sr),
        
        "DNSMOS_P808": extract(dnsmos, 'p808_mos', 0),
        "DNSMOS_SPEECH": extract(dnsmos, 'mos_sig', 1),
        "DNSMOS_BACKGROUND": extract(dnsmos, 'mos_bak', 2),
        "DNSMOS_OVERALL": extract(dnsmos, 'mos_ovr', 3),
        
        "SRMR": calculate_srmr(evaluated, sr),
        
        "NISQA_MOS": extract(nisqa, 'mos', 0),
        "NISQA_NOISINESS": extract(nisqa, 'noisiness', 1),
        "NISQA_DISCOUNTINUITY": extract(nisqa, 'discontinuity', 2),
        "NISQA_COLORATION": extract(nisqa, 'coloration', 3),
        "NISQA_LOUDNESS": extract(nisqa, 'loudness', 4),
    }