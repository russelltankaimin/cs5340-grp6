import torch
import torchaudio
import math

def add_deterministic_noise_sinusoidal(
    waveform: torch.Tensor,
    noise_level: float = 0.05,
    num_components: int = 64,
) -> torch.Tensor:
    """
    Adds deterministic pseudo-noise built from a sum of sinusoids
    with irrational frequency ratios, so the pattern never repeats
    in any perceptible way.
    """
    num_samples = waveform.shape[-1]
    t = torch.arange(num_samples, device=waveform.device, dtype=waveform.dtype)

    noise = torch.zeros_like(waveform)
    for i in range(1, num_components + 1):
        # Use irrational-ish frequency ratios so they never align
        freq = i * math.sqrt(2) * math.pi * (1 + i * 0.618033)  # golden ratio offset
        phase = i * i * 1.3579
        noise = noise + torch.sin(freq * t / num_samples * 2 * math.pi + phase)

    # Normalize to unit variance then scale
    noise = noise / noise.std()
    return waveform + noise_level * noise

# Load audio
audio_path = "./data/vshort.wav"
gt_y, sr = torchaudio.load(audio_path, backend="ffmpeg")
gt_y.requires_grad_(True)

# Apply distortion (differentiable)
distorted_y = add_deterministic_noise_sinusoidal(gt_y, noise_level=0.1)

# Save
torchaudio.save(
    "distorted_output_sin.wav",
    distorted_y.detach(),
    sr
)