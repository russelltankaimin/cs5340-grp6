import torch
import torchaudio

# Load audio
gt_y, sr = torchaudio.load("./data/vshort.wav", backend="ffmpeg")
gt_y.requires_grad_(True)

# This is fully differentiable — it's built on torch ops
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
)

mel_spec = mel_transform(gt_y)

# Log scaling (add small epsilon to avoid log(0))
log_mel = torch.log(mel_spec + 1e-9)

# Prove gradients flow through
loss = log_mel.mean()
loss.backward()

print(gt_y.grad is not None)  # True — gradients flow all the way back
print(gt_y.grad.shape)        # Same shape as gt_y