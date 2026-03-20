import torch
import torchaudio

def soft_clip_distortion(waveform: torch.Tensor, drive: float = 5.0) -> torch.Tensor:
    """
    Applies tanh soft clipping distortion. Fully differentiable.
    
    Higher drive pushes more of the signal into the saturated region
    of tanh, creating a more aggressive distortion.
    
    Args:
        waveform: Tensor of shape (channels, num_samples)
        drive: Gain applied before clipping (controls distortion intensity)
    Returns:
        Distorted waveform, same shape as input, values in (-1, 1)
    """
    return torch.tanh(waveform * drive)

# Load audio
audio_path = "sample_5s.wav"
gt_y, sr = torchaudio.load(audio_path, backend="ffmpeg")
gt_y.requires_grad_(True)

# Apply distortion (differentiable)
distorted_y = soft_clip_distortion(gt_y, drive=15.0)

# Save
torchaudio.save(
    "distorted_output.wav",
    distorted_y.detach(),
    sr
)