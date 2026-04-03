"""
File: utils/audio_io.py
Description: Standardised loading and saving of audio tensors using torchaudio.
"""

import torch
import torchaudio
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

def load_audio(path: str, target_sr: int = 44100, device: str = "cpu") -> torch.Tensor:
    """Load audio → stereo float32 tensor of shape (1, 2, T) at target_sr."""
    decoder = AudioDecoder(path)
    samples = decoder.get_all_samples()
    y = samples.data
    sr = samples.sample_rate

    if sr != target_sr:
        y = torchaudio.transforms.Resample(sr, target_sr)(y)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if y.shape[0] == 1:
        y = y.repeat(2, 1)  # mono → stereo
    return y.unsqueeze(0).to(device)  # (1, 2, T)

def save_audio(path: str, audio: torch.Tensor, sr: int) -> None:
    """
    Saves an audio tensor back to disk.
    
    Args:
        path (str): Output filepath.
        audio (torch.Tensor): Audio tensor to save.
        sr (int): Output sample rate.
    """
    AudioEncoder(audio.squeeze(0).cpu(), sample_rate=sr).to_file(path)