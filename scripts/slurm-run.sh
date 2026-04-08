#!/bin/bash
#SBATCH --job-name=cs5340-all
#SBATCH --partition=gpu            # adjust to your cluster
#SBATCH --gres=gpu:h100-96          # or comment out if CPU only
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# --- Optional: activate your environment / modules ---
VENV="${HOME}/cs5340-venv"
source "${VENV}/bin/activate"

mkdir -p logs

# Paths (adjust as needed)
SCRIPT_DIR="${HOME}/cs5340"
PY_DRIVER="${SCRIPT_DIR}/pipeline.py"

# Variables: Define the base input directory
INPUT_DIR="${SCRIPT_DIR}/data_full/test-clean"
OUTPUT_BASE="${SCRIPT_DIR}/outputs/${SLURM_JOB_ID}"

# Move to the project root
cd "${SCRIPT_DIR}"

# Explicitly add the project root to Python's search path
export PYTHONPATH="${SCRIPT_DIR}"
export LD_LIBRARY_PATH=/home/m/muchen/cs5340-venv/lib/python3.12/site-packages/nvidia/nvrtc/lib:$LD_LIBRARY_PATH

nvidia-smi

echo "Starting sequential processing of all .flac files in ${INPUT_DIR}..."

# Recursively find all .flac files and loop through them
find "${INPUT_DIR}" -type f -name "*.flac" | while read -r input_file; do

    # 1. Strip the base input path to get the relative path (e.g., 61/70968/61-70968-0000.flac)
    relative_path="${input_file#$INPUT_DIR/}"

    # 2. Extract just the directory component (e.g., 61/70968)
    relative_dir=$(dirname "$relative_path")

    # 3. Construct the corresponding nested output directory path
    nested_output_dir="${OUTPUT_BASE}/${relative_dir}"

    # 4. Create the nested output directory structure
    mkdir -p "${nested_output_dir}"

    echo "========================================"
    echo "Processing: ${input_file}"
    echo "Output directory: ${nested_output_dir}"
    echo "========================================"

    # Run the Python script for the current file
    python "${PY_DRIVER}" \
      --audio-path "${input_file}" \
      --output-dir "${nested_output_dir}" \
      --fill-last-clip

done

echo "Pipeline execution completed for all files!"
