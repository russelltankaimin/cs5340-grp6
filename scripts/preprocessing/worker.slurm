#!/bin/bash
#SBATCH --output=logs/preprocess_%x_%j.out
#SBATCH --error=logs/preprocess_%x_%j.err
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
OUTPUT_PARENT="$2"
INPUT_PARENT="$3"
CLIP_SECONDS="$4"

# Define the absolute path to your project's root directory
# (the folder containing both 'utils' and the directory holding 'preprocessing.py')
PROJECT_ROOT="${HOME}/cs5340"
PREPROCESS_MODULE_DIR="pipelines"

# Activate your Python environment here
VENV="${HOME}/cs5340-venv"
source "${VENV}/bin/activate"

echo "=================================================="
echo "Starting job on node: $HOSTNAME"
echo "Processing all directories starting with: $PREFIX"
echo "Input parent: $INPUT_PARENT"
echo "Output parent: $OUTPUT_PARENT"
echo "Project root: $PROJECT_ROOT"
echo "=================================================="

cd "$PROJECT_ROOT"

# Iterate over all directories matching the specific starting digit
for TARGET_DIR in "$INPUT_PARENT"/${PREFIX}*/; do
    
    # Verify it is a valid directory
    if [ -d "$TARGET_DIR" ]; then
        echo ">>> Entering grouped directory: $TARGET_DIR"
        
        # Recursively find all .mp3 files strictly within this subdirectory
        find "$TARGET_DIR" -type f -name "*.mp3" | while read -r filepath; do
            
            # Extract the relative path by stripping the overarching INPUT_PARENT prefix
            rel_path="${filepath#$INPUT_PARENT/}"
            
            # Extract the directory portion of the relative path
            rel_dir=$(dirname "$rel_path")
            
            # Extract the filename without the .mp3 extension
            filename=$(basename "$filepath" .mp3)
            
            # Construct the mirrored target directory
            file_output_dir="$OUTPUT_PARENT/$rel_dir/$filename"
            
            # Create the nested directory structure
            mkdir -p "$file_output_dir"
            
            echo "Processing: $rel_path"
            
            # Execute the Python script
            python -m ${PREPROCESS_MODULE_DIR}.preprocessing \
                --input "$filepath" \
                --output "$file_output_dir" \
                --clip-seconds "$CLIP_SECONDS"
                
        done
    fi
done

echo "=================================================="
echo "Finished all processing for prefix group: $PREFIX"
