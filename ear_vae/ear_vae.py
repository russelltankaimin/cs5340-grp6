"""
File: ear_vae/ear_vae.py
Description: The main wrapper class for the EAR-VAE model. It brings together the 
Oobleck encoder/decoder and the optional continuous transformer to form the complete 
Variational Autoencoder architecture.
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor, no_grad
from .autoencoders import OobleckDecoder, OobleckEncoder
from .transformer import ContinuousTransformer

sample_eps = 1e-6

def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale)
    var = stdev * stdev + sample_eps
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl

class EAR_VAE(nn.Module):
    def __init__(self, model_config: dict = None):
        super().__init__()
        if model_config is None:
            # Default configuration fallback
            model_config = {
                "encoder": {
                    "config": {
                        "in_channels": 2, "channels": 128, "c_mults": [1, 2, 4, 8, 16],
                        "strides": [2, 4, 4, 4, 8], "latent_dim": 128, "use_snake": True
                    }
                },
                "decoder": {
                    "config": {
                        "out_channels": 2, "channels": 128, "c_mults": [1, 2, 4, 8, 16],
                        "strides": [2, 4, 4, 4, 8], "latent_dim": 64, 
                        "use_nearest_upsample": False, "use_snake": True, "final_tanh": False,
                    },
                },
                "latent_dim": 64, "downsampling_ratio": 1024, "io_channels": 2,
            }

        if model_config.get("transformer") is not None:
            self.transformers = ContinuousTransformer(
                dim=model_config["decoder"]["config"]["latent_dim"],
                depth=model_config["transformer"]["depth"],
                **model_config["transformer"].get("config", {}),
            )
        else:
            self.transformers = None

        self.encoder = OobleckEncoder(**model_config["encoder"]["config"])
        self.decoder = OobleckDecoder(**model_config["decoder"]["config"])

    def forward(self, audio) -> Tensor:
        """Forward pass through the entire VAE pipeline."""
        status = self.encoder(audio)
        mean, scale = status.chunk(2, dim=1)
        z, kl = vae_sample(mean, scale)
        
        if self.transformers is not None:
            z = z.permute(0, 2, 1)
            z = self.transformers(z)
            z = z.permute(0, 2, 1)

        x = self.decoder(z)
        return x, kl

    def encode(self, audio, use_sample=True):
        """Encodes audio into the latent space."""
        x = self.encoder(audio)
        mean, scale = x.chunk(2, dim=1)
        if use_sample:
            z, _ = vae_sample(mean, scale)
        else:
            z = mean
        return z

    def decode(self, z):
        """Decodes a latent vector back into audio."""
        if self.transformers is not None:
            z = z.permute(0, 2, 1)
            z = self.transformers(z)
            z = z.permute(0, 2, 1)
            
        x = self.decoder(z)
        return x

    @no_grad()
    def inference(self, audio):
        """End-to-end inference without gradient tracking."""
        z = self.encode(audio)
        recon_audio = self.decode(z)
        return recon_audio