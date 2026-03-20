"""
File: scripts/vae_sample.py
Description: A simple script to run the VAE forward pass (encode -> decode) on an 
audio file to verify the model is loaded correctly and producing valid reconstructions.
"""
import argparse
import json
import torch
import os

from ear_vae.ear_vae import EAR_VAE
from utils.audio_io import load_audio, save_audio

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on: {device}")

    # Load Model Config
    with open(args.vae_config, 'r') as f:
        config = json.load(f)
    target_sr = config.get("sample_rate", 44100)

    # Load Model
    print("Loading model...")
    model = EAR_VAE(model_config=config).to(device)
    model.load_state_dict(torch.load(args.vae_checkpoint, map_location="cpu"))
    model.eval()

    # Load and process audio
    print(f"Loading input audio: {args.input}")
    audio = load_audio(args.input, target_sr=target_sr, device=device)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        reconstructed_audio = model.inference(audio)

    # Output formatting
    input_fname = os.path.splitext(os.path.basename(args.input))[0]
    output_fpath = os.path.join(".", f"{input_fname}_recon.wav")
    
    save_audio(output_fpath, reconstructed_audio, target_sr)
    print(f"Reconstruction saved to: {output_fpath}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Run VAE audio inference.")
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--vae-checkpoint', type=str, required=True)
    p.add_argument('--vae-config', type=str, required=True)
    
    main(p.parse_args())