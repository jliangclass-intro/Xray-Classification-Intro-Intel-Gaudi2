#!/usr/bin/env bash

# SLURM script to submit 9 training jobs
# Datasets: chestxray14, chexpert, mimic
# Backbones: convnext_base, resnet50, swin_base
# Training config: batch_size=256, optimizer=adamw, lr=1e-4
#
# Usage: ./submit-all-slurm.sh
# Note: Uses 1 GPU per job (for educational purposes, keeping it simple)

set -e

# Always use 1 GPU by default
# (Using 2 or more GPUs is automatically supported by Lightning in train_v2_lightning.py)
NUM_GPUS=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# SLURM job configuration
SLURM_OPTS=(
    --job-name=xray-train
    --time=24:00:00
    --nodes=1
    --ntasks-per-node=1
    --cpus-per-task=8
    --mem=64G
    --gres=gpu:a100:${NUM_GPUS}
 

# Function to submit a single job
submit_job() {
    local command=$1
    
    # Extract dataset and backbone from command for job name
    local dataset=$(echo "$command" | grep -oP 'dataset=\K\w+')
    local backbone=$(echo "$command" | grep -oP 'model=\K\w+')
    
    echo "Submitting job: ${command}"
    
    sbatch "${SLURM_OPTS[@]}" \
        --job-name="${dataset}-${backbone}" \
        --wrap="bash -c '${command}'"
}

# Submit all 9 jobs
echo "=========================================="
echo "Submitting 9 training jobs (1 GPU per job)"
echo "=========================================="

# Dataset: chestxray14
submit_job "bash run.sh dataset=chestxray14 model=convnext_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4"
submit_job "bash run.sh dataset=chestxray14 model=resnet50 train.batch_size=128 train.optimizer=adamw train.lr=1e-4"
submit_job "bash run.sh dataset=chestxray14 model=swin_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4"

# Dataset: chexpert
submit_job "bash run.sh dataset=chexpert model=convnext_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4"
submit_job "bash run.sh dataset=chexpert model=resnet50 train.batch_size=128 train.optimizer=adamw train.lr=1e-4"
submit_job "bash run.sh dataset=chexpert model=swin_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4"

# Dataset: mimic
submit_job "bash run.sh dataset=mimic model=convnext_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4"
submit_job "bash run.sh dataset=mimic model=resnet50 train.batch_size=128 train.optimizer=adamw train.lr=1e-4"
submit_job "bash run.sh dataset=mimic model=swin_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4"

echo "=========================================="
echo "All 9 jobs submitted!"
echo "Check status with: squeue -u \$USER"
echo "=========================================="
