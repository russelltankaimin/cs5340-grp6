"""
File: utils/losses.py
Description: The objective functions used during the Bayesian gradient descent loop.
Includes prior constraints and reconstruction fidelities.
"""

import torch
import torchaudio

def loss_w(z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian prior loss on the latent vector."""
    return ((z - mu) / sigma).pow(2).mean()

def loss_colin(z: torch.Tensor) -> torch.Tensor:
    """
    Colinearity loss to encourage temporal alignment across latent time steps.
    Dynamically adapts to any audio length.
    """
    # z shape is (Batch, Channels, Time) e.g., (1, 64, 171)
    B, C, T = z.shape
    
    # Transpose to (Batch, Time, Channels) and reshape to (T, C)
    # This gives us T sub-vectors of size C
    vectors = z.transpose(1, 2).reshape(T, C)
    
    normed = torch.nn.functional.normalize(vectors, dim=1)
    sim_matrix = normed @ normed.T
    
    mask = torch.triu(torch.ones(T, T, device=z.device), diagonal=1).bool()
    return -sim_matrix[mask].sum()

def loss_waveform(corrupted_input: torch.Tensor, corrupted_recon: torch.Tensor) -> torch.Tensor:
    """Time-domain L2 waveform loss."""
    T = min(corrupted_input.shape[-1], corrupted_recon.shape[-1])
    return (corrupted_input[..., :T] - corrupted_recon[..., :T]).pow(2).mean()

def get_mel_loss_fn(sample_rate: int, device: str, n_fft=1024, hop_length=256, n_mels=80):
    """Factory function to build a differentiable Mel-Spectrogram loss function."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    ).to(device)
    
    def loss_mel(corrupted_input: torch.Tensor, corrupted_recon: torch.Tensor) -> torch.Tensor:
        T = min(corrupted_input.shape[-1], corrupted_recon.shape[-1])
        mel_in  = torch.log(mel_transform(corrupted_input[..., :T])  + 1e-9)
        mel_rec = torch.log(mel_transform(corrupted_recon[..., :T]) + 1e-9)
        return (mel_in - mel_rec).pow(2).mean()
    
    return loss_mel

def loss_trajectory(z: torch.Tensor, l0: float, l1: float, l2: float) -> torch.Tensor:
    """
    Implements the Latent Trajectory Prior.
    z shape: (batch, latent_dim, T_prime)
    """
    w = z.squeeze(0).T 
    T_prime = w.shape[0]
    
    # Term 1: Gaussian Region Constraint (Zeroth-order)
    loss_0 = torch.norm(w, p=2, dim=1).pow(2).sum() / T_prime
    
    # Term 2: First-order Temporal Smoothness (Velocity)
    if T_prime > 1:
        diff1 = w[1:] - w[:-1]
        loss_1 = torch.norm(diff1, p=2, dim=1).pow(2).sum() / (T_prime - 1)
    else:
        loss_1 = 0.0

    # Term 3: Second-order Smoothness (Acceleration)
    if T_prime > 2:
        diff2 = w[2:] - 2*w[1:-1] + w[:-2]
        loss_2 = torch.norm(diff2, p=2, dim=1).pow(2).sum() / (T_prime - 2)
    else:
        loss_2 = 0.0

    return (l0/2 * loss_0) + (l1/2 * loss_1) + (l2/2 * loss_2)