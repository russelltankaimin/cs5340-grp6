import torch
import torchaudio

def soft_clip_distortion(
    waveform: torch.Tensor,
    drive: float = 15.0,
) -> torch.Tensor:
    """Tanh soft-clipping distortion (fully differentiable)."""
    return torch.tanh(waveform * drive)

# Load audio
audio_path = "../data/sample_5s.wav"
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