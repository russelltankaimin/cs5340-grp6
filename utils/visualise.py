"""
File: utils/visualise.py
Description: Generates visual comparisons for waveforms, spectrograms, and evaluation metrics.
"""

import matplotlib.pyplot as plt
import torch
import torchaudio
import numpy as np

def plot_spectrograms(clean: torch.Tensor, corrupted: torch.Tensor, recon: torch.Tensor, sr: int, save_path: str):
    """Plots Mel-Spectrogram comparisons."""
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)
    
    # Compute the log mel spectrogram and squeeze the batch dimension
    mel_clean_full = torch.log(mel_transform(clean.cpu()) + 1e-9).squeeze().numpy()
    mel_corr_full = torch.log(mel_transform(corrupted.cpu()) + 1e-9).squeeze().numpy()
    mel_recon_full = torch.log(mel_transform(recon.cpu()) + 1e-9).squeeze().numpy()
    
    # Handle stereo (2 channels) by selecting the first channel for visualisation
    # If the audio was mono, ndim would be 2, so we conditionally slice.
    mel_clean = mel_clean_full[0] if mel_clean_full.ndim == 3 else mel_clean_full
    mel_corr = mel_corr_full[0] if mel_corr_full.ndim == 3 else mel_corr_full
    mel_recon = mel_recon_full[0] if mel_recon_full.ndim == 3 else mel_recon_full
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    
    im0 = axs[0].imshow(mel_clean, aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title('Clean Reference Spectrogram (Left Channel)')
    axs[0].set_ylabel('Mel Bins')
    
    im1 = axs[1].imshow(mel_corr, aspect='auto', origin='lower', cmap='viridis')
    axs[1].set_title('Corrupted Spectrogram (Left Channel)')
    axs[1].set_ylabel('Mel Bins')
    
    im2 = axs[2].imshow(mel_recon, aspect='auto', origin='lower', cmap='viridis')
    axs[2].set_title('Reconstructed Spectrogram (Left Channel)')
    axs[2].set_ylabel('Mel Bins')
    axs[2].set_xlabel('Frames')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(baseline_metrics: dict, recon_metrics: dict, save_path: str):
    """Plots a grouped bar chart comparing the metrics before and after reconstruction."""
    metrics_keys = list(baseline_metrics.keys())
    
    baseline_vals = [baseline_metrics[k] for k in metrics_keys]
    recon_vals = [recon_metrics[k] for k in metrics_keys]
    
    x = np.arange(len(metrics_keys))
    width = 0.35
    
    # Expanded the figure size to provide more horizontal and vertical rendering space
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x - width/2, baseline_vals, width, label='Corrupted (Baseline)', color='#e74c3c')
    ax.bar(x + width/2, recon_vals, width, label='Reconstructed', color='#2ecc71')
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Objective and Perceptual Metrics Comparison')
    ax.set_xticks(x)
    
    # Rotated the labels by 45 degrees and anchored them to the right edge
    ax.set_xticklabels(metrics_keys, rotation=45, ha='right', rotation_mode='anchor')
    ax.legend()
    
    # Add value annotations
    for i, v in enumerate(baseline_vals):
        ax.text(i - width/2, v, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(recon_vals):
        ax.text(i + width/2, v, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        
    # tight_layout() will automatically adjust the bottom margin to fit the rotated text
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()