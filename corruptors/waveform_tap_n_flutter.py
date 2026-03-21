import torch
import torchaudio
import math


def tape_wow_flutter(
    waveform: torch.Tensor,
    wow_freq: float = 1.5,
    wow_depth: float = 0.003,
    flutter_freq: float = 28.0,
    flutter_depth: float = 0.0005,
    sample_rate: int = 44100,
) -> torch.Tensor:
    """
    Simulates analog tape wow & flutter — the speed instability of a
    mechanical tape transport.

    "Wow" is slow, broad pitch drift (warped capstan, eccentric reel).
    "Flutter" is fast, shallow modulation (motor cogging, guide vibration).

    Both warp the time axis with low-frequency sinusoids, then resample
    via linear interpolation — fully differentiable w.r.t. `waveform`.

    Args:
        waveform:       Tensor of shape (channels, num_samples)
        wow_freq:       Wow modulation rate in Hz (typically 0.5–4 Hz)
        wow_depth:      Wow displacement in seconds (higher = more pitch drift)
        flutter_freq:   Flutter modulation rate in Hz (typically 10–50 Hz)
        flutter_depth:  Flutter displacement in seconds
        sample_rate:    Audio sample rate
    Returns:
        Time-warped waveform, same shape as input
    """
    T = waveform.shape[-1]
    t = torch.arange(T, device=waveform.device, dtype=waveform.dtype) / sample_rate

    # Time-varying displacement (in samples)
    displacement = (
        wow_depth     * torch.sin(2 * math.pi * wow_freq * t)
        + flutter_depth * torch.sin(2 * math.pi * flutter_freq * t)
    ) * sample_rate  # convert seconds → samples

    # Warped read-head positions
    indices = torch.arange(T, device=waveform.device, dtype=waveform.dtype) + displacement

    # Clamp to valid range for interpolation
    indices = indices.clamp(0, T - 1)

    # Differentiable linear interpolation
    idx_floor = indices.long().clamp(0, T - 2)
    idx_ceil  = idx_floor + 1
    frac      = (indices - idx_floor.float()).unsqueeze(0)  # (1, T) for broadcasting

    output = waveform[:, idx_floor] * (1 - frac) + waveform[:, idx_ceil] * frac
    return output


if __name__ == "__main__":
    audio_path = "sample_5s.wav"
    gt_y, sr = torchaudio.load(audio_path, backend="ffmpeg")
    gt_y.requires_grad_(True)

    # Apply corruption (differentiable)
    distorted_y = tape_wow_flutter(gt_y, sample_rate=sr)

    # Verify gradients flow
    loss = distorted_y.mean()
    loss.backward()
    print(f"Gradient flows: {gt_y.grad is not None}")  # True
    print(f"Grad shape:     {gt_y.grad.shape}")

    # Save
    torchaudio.save(
        "distorted_output_wow_flutter.wav",
        distorted_y.detach(),
        sr,
    )