import torch
import torchaudio
import math

def sinusoidal_noise(
    waveform: torch.Tensor,
    noise_level: float = 0.1,
    num_components: int = 64,
) -> torch.Tensor:
    """Deterministic pseudo-noise built from a sum of sinusoids with
    irrational frequency ratios (fully differentiable)."""
    num_samples = waveform.shape[-1]
    t = torch.arange(num_samples, device=waveform.device, dtype=waveform.dtype)
    noise = torch.zeros_like(waveform)
    for i in range(1, num_components + 1):
        freq  = i * math.sqrt(2) * math.pi * (1 + i * 0.618033)
        phase = i * i * 1.3579
        noise = noise + torch.sin(freq * t / num_samples * 2 * math.pi + phase)
    noise = noise / noise.std()
    return waveform + noise_level * noise

# Load audio
audio_path = "../data/vshort.wav"
gt_y, sr = torchaudio.load(audio_path, backend="ffmpeg")
gt_y.requires_grad_(True)

# Apply distortion (differentiable)
distorted_y = sinusoidal_noise(gt_y, noise_level=0.1)

# Save
torchaudio.save(
    "distorted_output_sin.wav",
    distorted_y.detach(),
    sr
)