# CS5340 Group 6 - Bayesian Audio Reconstruction

This repository contains the codebase for our Bayesian Audio Reconstruction pipeline. The programme restores degraded audio signals by performing gradient descent on the latent space of a frozen, pre-trained audio Variational Autoencoder (EAR-VAE). 

By unifying our forward corruption models into fully differentiable PyTorch modules, we ensure that the mathematical operations used to degrade the evaluation datasets are perfectly inverted during the Bayesian optimisation loop.

## Directory Structure
* **`ear_vae/`**: The core generative model architectures, including the Oobleck autoencoders and continuous transformers.
* **`corruptions/`**: Fully differentiable PyTorch implementations of audio degradations (e.g., FFT masking, Soft Clipping, Gaussian Noise) managed by a central registry.
* **`utils/`**: Shared utilities for audio I/O, region crossfading, and objective loss functions.
* **`scripts/`**: Executable entry points for dataset generation, statistics computation, and the Bayesian reconstruction loop.
* **`vae_ckpt/`**: Directory designated for pre-trained weights and configurations.

```text
cs5340-grp6-main/
├── ear_vae/                  # Core generative model architecture (frozen VAE)
│   ├── __init__.py           # Package initialisation
│   ├── ear_vae.py            # Main wrapper class for the entire VAE pipeline
│   ├── autoencoders.py       # Oobleck-based convolutional encoder and decoder
│   └── transformer.py        # Continuous transformer for latent space modelling
├── corruptions/              # Fully differentiable forward models (degradations)
│   ├── __init__.py           # Package init; auto-registers all corruptions on import
│   ├── registry.py           # Central dictionary for managing corruption functions
│   ├── additive.py           # Differentiable additive noises (Gaussian, Pink, Hum)
│   ├── frequency.py          # Differentiable frequency-domain noises (FFT Mask, EQ)
│   └── nonlinear.py          # Differentiable distortions (Soft Clip, STE Bitcrush)
├── utils/                    # Shared helper functions and logic
│   ├── __init__.py           # Package initialisation
│   ├── audio_io.py           # Standardised audio loading, saving, and resampling
│   ├── regions.py            # Time-segment application with crossfade blending
│   └── losses.py             # Objective functions (L_w, L_colin, L_wav, L_mel)
├── scripts/                  # Executable entry points for the pipeline
│   ├── corrupt_dataset.py    # Generates degraded evaluation datasets
│   ├── run_reconstruction.py # The core Bayesian optimisation gradient descent loop
│   ├── compute_stats.py      # Computes prior latent statistics (mean, std)
│   ├── gpu_job.sh            # Slurm batch script for SoC cluster execution
│   └── vae_sample.py         # Inference test script to verify model health
├── vae_ckpt/                 # Pre-trained model checkpoints
│   ├── ear_vae_44k.pyt       # The pre-trained weights from HuggingFace
│   └── model_config.json     # Hyperparameter configuration for the VAE
├── data/                     # Directory for input/output audio files and JSON stats
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation and setup instructions
```

---

## Basic Setup

### General Setup
Set up your Python environment and install the required dependencies for the project. We recommend using a virtual environment to manage dependencies. 

For macOS and Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required dependencies for the project:
```bash
pip install -r requirements.txt
```

### VAE Setup and Usage
*Reference: [https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE](https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE)*

Install the custom dependencies required for the VAE model:
```bash
pip install descript-audio-codec
pip install alias-free-torch
```

**Note:** You might need to install `ffmpeg` separately on your system to handle audio processing. Remember to restart your terminal or IDE after installation. This is not needed if you are running on the SoC Cluster.

For example, on macOS, you can use Homebrew:
```bash
brew install ffmpeg
```

For Windows, you can install via winget:
```bash
winget install "FFmpeg (Shared)"
```

**Model Weights:** Download the model weight file from [HuggingFace](https://huggingface.co/earlab/EAR_VAE/tree/main/pretrained_weight). It should be named `ear_vae_44k.pyt` and placed in the `vae_ckpt/` directory, alongside `model_config.json`.

To verify the setup, you can run the VAE inference test script:
```bash
python -m scripts.vae_sample --input path/to/your/input.wav --vae-checkpoint vae_ckpt/ear_vae_44k.pyt --vae-config vae_ckpt/model_config.json
```

---

## Experiment Workflow

### 1. Compute Prior Statistics
First, choose an audio file and place it in your `data/` directory. We compute the statistics of a clean audio file to establish the prior distribution used during the Bayesian reconstruction process. 

If you want to use a prior that does not leak information about the original audio, you can compute these statistics from a distinctly different audio file.
```bash
python -m scripts.compute_stats \
    --input data/sample.wav \
    --output latent_stats.json \
    --vae-checkpoint vae_ckpt/ear_vae_44k.pyt \
    --vae-config vae_ckpt/model_config.json
```

### 2. Generate the Corrupted Dataset
Instead of disjointed scripts, use the unified dataset corruptor. This script applies a strictly differentiable corruption to the audio file using the central `CORRUPTION_REGISTRY`. This guarantees the forward model used to degrade the data matches the one the optimiser attempts to invert.
```bash
python -m scripts.corrupt_dataset \
    --input data/sample.wav \
    --output data/corrupted_sample.wav \
    --corruption soft_clip \
    --corruption-kwargs '{"drive": 15.0}'
```

### 3. Run Bayesian Reconstruction
Finally, run the Bayesian reconstruction to obtain the restored audio. The optimiser will traverse the latent space to invert the applied corruption, heavily constrained by the pre-computed prior.
```bash
python -m scripts.run_reconstruction \
    --input data/corrupted_sample.wav \
    --output data/reconstructed.wav \
    --prior-stats latent_stats.json \
    --corruption soft_clip \
    --corruption-kwargs '{"drive": 15.0}' \
    --vae-checkpoint vae_ckpt/ear_vae_44k.pyt \
    --vae-config vae_ckpt/model_config.json \
    --steps 500 --lr 1e-3
```
*Note: Ensure the `--corruption` and `--corruption-kwargs` arguments exactly match the ones used in Step 2 to ensure the reconstruction process works as intended*.

---

## SoC Cluster Execution

If you are running on the SoC Cluster, modify `scripts/gpu_job.sh` to include the correct command to run the reconstruction script with the appropriate parameters. Then, submit the job using:
```bash
sbatch scripts/gpu_job.sh
```