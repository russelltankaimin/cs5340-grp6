#!/bin/bash

# Define the absolute paths and parameters
SCRIPT_DIR="${HOME}/cs5340"
INPUT_PARENT="${SCRIPT_DIR}/output"
SLURM_JOB="${SCRIPT_DIR}/scripts/pipeline/worker.slurm"
CLIP_SECONDS=5.0

# Ensure the logs directory exists
mkdir -p logs

echo "Scanning for directories and grouping jobs by starting digit (1-9)..."

# Iterate strictly through digits 1 to 9
for digit in {1..9}; do
    # Check if any subdirectories exist that start with the current digit
    # We direct stderr to /dev/null to silently ignore non-matches
    if ls -d "$INPUT_PARENT"/${digit}*/ >/dev/null 2>&1; then
        echo "Submitting SLURM job for all subdirectories starting with prefix: ${digit}"
        
        # Submit the worker script, passing the digit prefix as the first argument
        sbatch \
            --job-name="eval_group_${digit}" \
            "$SLURM_JOB" \
            "$digit" \
            "$INPUT_PARENT" \
            "$CLIP_SECONDS"
    fi
done

echo "All grouped jobs have been submitted to the queue."
