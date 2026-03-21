import torchaudio
import os
import torch
import argparse
import json
from ear_vae.ear_vae import EAR_VAE

MODEL_PATH = os.path.join(".", "vae_ckpt", "ear_vae_44k.pyt")
CONFIG_PATH = os.path.join(".", "vae_ckpt", "model_config.json")

def main(args):
    input_audio_path = args.input_fpath
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Input audio path: {input_audio_path}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Config path: {CONFIG_PATH}")

    with open(CONFIG_PATH, 'r') as f:
        vae_gan_model_config = json.load(f)

    print("Loading model...")
    model = EAR_VAE(model_config=vae_gan_model_config).to(device)

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully.")

    with torch.no_grad():
        gt_y, sr = torchaudio.load(input_audio_path, backend="ffmpeg")

        if len(gt_y.shape) == 1:
            gt_y = gt_y.unsqueeze(0)

        # Resample if necessary
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100).to(device)
            gt_y = resampler(gt_y)

        gt_y = gt_y.to(device, torch.float32)
        
        # Convert to stereo if mono
        if gt_y.shape[0] == 1:
            gt_y = torch.cat([gt_y, gt_y], dim=0)

        # Add batch dimension
        gt_y = gt_y.unsqueeze(0)

        print(f"Input audio shape (after processing): {gt_y.shape}")
        audio_latent = model.encode(gt_y)
        print(f"Encoded latent shape: {audio_latent.shape}")
        reconstructed_audio = model.decode(audio_latent)

        # get stem name from input audio path
        input_fname = os.path.splitext(os.path.basename(input_audio_path))[0]
        print(f"Input filename: {input_fname}")
        output_filename = f"{input_fname}_recon.wav"

        reconstructed_audio = reconstructed_audio.squeeze(0).cpu()
        output_fpath = os.path.join(".", output_filename)
        torchaudio.save(output_fpath, reconstructed_audio, sample_rate=44100, backend="ffmpeg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run VAE-GAN audio inference.")
    parser.add_argument('--input-fpath', type=str, required=True, help='Path to the input audio file (e.g., "input.wav").')
    
    args = parser.parse_args()
    main(args)
