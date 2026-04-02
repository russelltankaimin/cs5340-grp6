"""
File: utils/audio_io.py
Description: Standardised loading and saving of audio tensors using torchaudio.
"""

import torch
import torchaudio

def load_audio(path: str, target_sr: int = 44100, device: str = "cpu") -> torch.Tensor:
    """
    Loads an audio file, resamples if necessary, and formats it as a stereo tensor.
    
    Args:
        path (str): Filepath to the audio file.
        target_sr (int): The required sample rate.
        device (str): Compute device ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: Audio tensor of shape (1, 2, T).
    """
    y, sr = torchaudio.load(path, backend="ffmpeg")
    if sr != target_sr:
        y = torchaudio.transforms.Resample(sr, target_sr)(y)
    
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if y.shape[0] == 1:
        y = y.repeat(2, 1)  # mono to stereo
        
    return y.unsqueeze(0).to(device)

def save_audio(path: str, audio: torch.Tensor, sr: int) -> None:
    """
    Saves an audio tensor back to disk.
    
    Args:
        path (str): Output filepath.
        audio (torch.Tensor): Audio tensor to save.
        sr (int): Output sample rate.
    """
    torchaudio.save(path, audio.squeeze(0).cpu(), sample_rate=sr, backend="ffmpeg")