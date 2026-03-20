# CS5340 Group 6 - Bayesian Audio Reconstruction

## Basic Setup

### General Setup
Setup your Python environment and install the required dependencies for the project. We recommend using a virtual environment to manage dependencies. For MacOS and Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

For windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

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

Note: You might need to install `ffmpeg` separately on your system to handle audio processing. Remember to restart your terminal/IDE after installation. This is not needed if you are running on SoC Cluster.

For example, on MacOS, you can use Homebrew:
```bash
brew install ffmpeg
```

For Windows, you can install via winget:
```bash
winget install "FFmpeg (Shared)"
```

Download the model weight file from [HuggingFace](https://huggingface.co/earlab/EAR_VAE/tree/main/pretrained_weight), named `ear_vae_44k.pyt` and place it in the `vae_ckpt` directory.

To run the VAE inference script, use the following command:
```bash
python vae_sample.py --input-fpath path/to/your/input.wav 
```

### Experiment Sample Workflow
1. Choose your favourite audio file and place it in the `data` directory, here we use `data/sample.wav` as an example.

2. We run 
    ```bash
    python compute_stats.py --input-fpath data/sample.wav
    ```
    to compute the statistics of the audio file, which will be used for the Bayesian reconstruction process. If you want to use a prior that does not leak information of the original audio, you can use the statistics computed from a different audio file.

3. We extract a short clip of `T` seconds, here `T=5` seconds, from the original audio file using 
    ```bash
    python extract.py --audio-path data/sample.wav --start-time 96 --duration 5
    ```

4. We then corrupt the extracted clip using 
    ```bash
    python waveform_soft_clip_dist.py
    ```
    You should modify the `audio_path` in the script to point to the extracted clip.

5. Finally, we run 
    ```bash
    python exp_v1.py --input path/to/corrupted/audio.wav --corruption soft_clip
    ```
    to perform the Bayesian reconstruction and obtain the reconstructed audio. 

6. If you are running on the SoC Cluster, modify `scripts/gpu_job.sh` to include the correct command to run the reconstruction script with the appropriate parameters. Then, submit the job using:
    ```bash
    sbatch scripts/gpu_job.sh
    ```

7. Note that you can change the type of corruption, the severity of the corruption, and other parameters in the scripts to experiment with different scenarios. Just make sure that the parameters describing the corruption are passed into the `exp_v1.py` script correctly to ensure the reconstruction process works as intended.