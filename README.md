# CS5340 Group 6 - Bayesian Audio Reconstruction

This repository contains our Bayesian audio reconstruction experiments built on top of `EAR_VAE`, along with local single-file tooling, batch pipeline utilities, and SLURM scripts for cluster runs.

## Repository Layout

```text
cs5340-grp6/
├── pipeline.py
├── README.md
├── requirements.txt
├── requirements_new.txt
├── setup.py
├── corruptors/
│   ├── additive_noise.py
│   ├── audio_corruptor_freq.py
│   ├── audio_corruptor_single.py
│   ├── waveform_sinus_dist.py
│   ├── waveform_soft_clip_dist.py
│   └── waveform_tap_n_flutter.py
├── ear_vae/
│   ├── autoencoders.py
│   ├── ear_vae.py
│   └── transformer.py
├── experiments/
│   ├── exp_v1.py
│   ├── exp_v1_1.py
│   └── exp_v2.py
├── pipelines/
│   ├── pipeline_eval.py
│   └── preprocessing.py
├── scripts/
│   ├── gpu_job.sh
│   ├── gpu_vae_sample.sh
│   ├── slurm-install.sh
│   ├── slurm-pipeline.sh
│   ├── slurm-run.sh
│   ├── pipeline/
│   │   ├── submit_jobs.sh
│   │   └── worker.slurm
│   └── preprocessing/
│       ├── submit_jobs.sh
│       └── worker.slurm
├── utils/
│   ├── audio_io.py
│   ├── compute_stats.py
│   ├── evaluate.py
│   ├── extract.py
│   ├── metrics.py
│   ├── vae_sample.py
│   ├── visualise.py
│   └── waveform_to_mel.py
└── vae_ckpt/
    ├── ear_vae_44k.pyt
    └── model_config.json
```

### What Each Folder Is For

- `corruptors/`: corruption functions and standalone corruption utilities.
- `ear_vae/`: local copy of the EAR_VAE model implementation.
- `experiments/`: reconstruction experiments. `exp_v1.py` is the baseline, `exp_v2.py` adds a latent trajectory prior, and `exp_v1_1.py` is the latest variant with x-space trajectory terms and optional Jacobian support.
- `pipelines/`: main preprocessing and evaluation pipeline entry points.
- `scripts/`: SLURM helper scripts for installation, preprocessing, evaluation, and GPU jobs.
- `utils/`: reusable helpers for stats computation, extraction, evaluation, plotting, and VAE sampling.
- `vae_ckpt/`: required VAE checkpoint and config files.
- `pipeline.py`: legacy root-level wrapper that is not the main documented pipeline.

Other useful directories you will see during normal use:

- `data/`: convenient place for local test audio files.
- `data_full/`: larger dataset-style input directory.
- `output/`: recommended output location for the main preprocessing/evaluation flow.
- `results/`: legacy output location used by the root `pipeline.py` wrapper.

## Virtual Environment Setup

### Requirements

- Python 3.11 or newer
- `ffmpeg` available on your machine

If `ffmpeg` is missing, install it first.

- Windows: `winget install "FFmpeg (Shared)"`
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`

### Create and Activate the Virtual Environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Dependencies

For a normal local setup:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Notes:

- `requirements.txt` is the main project dependency list for local use.
- `requirements_new.txt` is mainly used by the cluster install script in `scripts/slurm-install.sh`.
- The reconstruction code expects both `vae_ckpt/ear_vae_44k.pyt` and `vae_ckpt/model_config.json` to exist.

## Run the Main Pipeline

The main documented workflow uses:

1. `pipelines.preprocessing` to split the source audio into `split_80.wav` and `split_20.wav`, and to compute `latent_stats.json`.
2. `pipelines.pipeline_eval` to chunk the evaluation split, apply one or more corruptions, run reconstruction with `experiments/exp_v1_1.py`, and write metrics to `metrics_report.csv`.

### Step 1: Preprocess the Audio

```bash
python -m pipelines.preprocessing --input data/sample.wav --output output/sample_run --clip-seconds 5
```

This creates:

- `output/sample_run/split_80.wav`
- `output/sample_run/split_20.wav`
- `output/sample_run/latent_stats.json`

### Step 2: Run the Main Evaluation / Reconstruction Pipeline

```bash
python -m pipelines.pipeline_eval --input output/sample_run/split_20.wav --output-dir output/sample_run --prior-stats output/sample_run/latent_stats.json --clip-seconds 5 --steps 2000
```

### Run Only Selected Corruptions

```bash
python -m pipelines.pipeline_eval --input output/sample_run/split_20.wav --output-dir output/sample_run --prior-stats output/sample_run/latent_stats.json --clip-seconds 5 --corruptions soft_clip sinusoidal --steps 2000
```

Supported corruption names in `pipelines.pipeline_eval` are:

- `sinusoidal`
- `soft_clip`
- `soft_clip_rms`
- `tape_wow_flutter`

### Override Corruption Parameters

```bash
python -m pipelines.pipeline_eval --input output/sample_run/split_20.wav --output-dir output/sample_run --prior-stats output/sample_run/latent_stats.json --clip-seconds 5 --corruptions soft_clip --corruption_kwargs '{"soft_clip": {"drive": 20.0}}' --steps 2000
```

### Common Optional Arguments

- `--target-sr 44100`
- `--steps 2000`
- `--lr 1e-3`
- `--log-every 50`
- `--include-jacobian`
- `--lambda-logdet 1e-6`

### Output Structure

The main pipeline writes into the directory passed to `--output-dir`, for example:

```text
output/sample_run/
├── latent_stats.json
├── split_20.wav
├── split_80.wav
├── metrics_report.csv
├── sinusoidal/
│   ├── clip_0_clean.wav
│   ├── clip_0_corrupted.wav
│   └── clip_0_recon.wav
└── soft_clip/
    ├── clip_0_clean.wav
    ├── clip_0_corrupted.wav
    └── clip_0_recon.wav
```

## Test on Individual Audio Files

### Option 1: Quick VAE Smoke Test on One File

This checks that the VAE checkpoint loads and can encode/decode a single audio file.

```bash
python utils/vae_sample.py --input-fpath data/sample.wav
```

This writes `<input_stem>_recon.wav` to the project root.

### Option 2: Full Single-File Reconstruction Run

If you want the full Bayesian reconstruction workflow on one audio file, run the main preprocessing + evaluation pipeline:

```bash
python -m pipelines.preprocessing --input data/sample.wav --output output/sample_run --clip-seconds 5
python -m pipelines.pipeline_eval --input output/sample_run/split_20.wav --output-dir output/sample_run --prior-stats output/sample_run/latent_stats.json --clip-seconds 5 --corruptions sinusoidal --steps 2000
```

This is the recommended way to test a single file end to end. By default, `pipelines.pipeline_eval` uses the corruption registry from `experiments/exp_v1_1.py` and writes a `metrics_report.csv` into the chosen output directory.

## Batch / Cluster Usage

For cluster runs, use the scripts in `scripts/`:

- `scripts/slurm-install.sh`: install dependencies in a cluster environment.
- `scripts/preprocessing/submit_jobs.sh`: submit grouped preprocessing jobs.
- `scripts/pipeline/submit_jobs.sh`: submit grouped evaluation jobs.
- `scripts/slurm-pipeline.sh` and `scripts/slurm-run.sh`: older root-level wrappers that still call `pipeline.py`, not the main documented pipeline.

## Notes

- `pipelines/pipeline_eval.py` is the main documented pipeline and uses `experiments/exp_v1_1.py`.
- `pipeline.py` still exists, but it is treated as a legacy wrapper in this README.
- `corruptors/audio_corruptor_single.py` is a standalone corruption tool for generating corrupted files, but it is separate from the differentiable corruption functions used inside the reconstruction experiments.
