# CS5340 Group 6 - Bayesian Audio Reconstruction

## Basic Setup

### General Setup
Install the required dependencies for the project:
```bash
pip install -r requirements.txt
```

### VAE Setup and Usage
Reference: https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE

Install the custom dependencies for the VAE model:
```bash
pip install descript-audio-codec
pip install alias-free-torch
```

Note: You might need to install `ffmpeg` separately on your system to handle audio processing.

For example, on MacOS, you can use Homebrew:
```bash
brew install ffmpeg
```

Download the model weight file from [HuggingFace](https://huggingface.co/earlab/EAR_VAE/tree/main/pretrained_weight), named `ear_vae_44k.pyt` and place it in the `vae_ckpt` directory.

To run the VAE inference script, use the following command:
```bash
python vae_sample.py --input-fpath path/to/your/input.wav 
```