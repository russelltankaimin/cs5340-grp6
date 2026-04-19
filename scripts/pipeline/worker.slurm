#!/bin/bash
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Exit immediately if a command exits with a non-zero status
set -e
set -x

# Retrieve positional arguments passed by submit_jobs.sh
PREFIX="$1"
INPUT_PARENT="$2"
CLIP_SECONDS="$3"

# Define the absolute path to your project's root directory
PROJECT_ROOT="${HOME}/cs5340"
EVAL_MODULE="pipelines.pipeline_eval"

# Activate your Python environment here
VENV="${HOME}/cs5340-venv"
source "${VENV}/bin/activate"

cd "$PROJECT_ROOT"

echo "=================================================="
echo "Starting job on node: $HOSTNAME"
echo "Processing all directories starting with: $PREFIX"
echo "Input parent: $INPUT_PARENT"
echo "Project root: $PROJECT_ROOT"
echo "=================================================="

# Iterate over all directories matching the specific starting digit
find "$INPUT_PARENT/${PREFIX}"* -type f -name "latent_stats.json" | while read -r stats_file; do
    
    # Extract the directory containing the stats file
    TARGET_DIR=$(dirname "$stats_file")
    AUDIO_INPUT="$TARGET_DIR/split_20.wav"
    
    # Verify it is a valid directory
    if [ -f "$AUDIO_INPUT" ]; then
        echo ">>> Running Eval for: $TARGET_DIR"
        
        # We set output-dir to the same folder as requested
        python -m $EVAL_MODULE \
            --input "$AUDIO_INPUT" \
            --output-dir "$TARGET_DIR" \
            --prior-stats "$stats_file" \
            --clip-seconds "$CLIP_SECONDS" \
            --steps 2000
            
    else
        echo "!!! Skipping $TARGET_DIR: split_20.wav not found."
    fi
done

echo "=================================================="
echo "Finished all processing for prefix group: $PREFIX"
