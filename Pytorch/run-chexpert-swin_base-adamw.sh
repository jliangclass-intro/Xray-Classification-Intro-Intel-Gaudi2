#!/usr/bin/env bash
#SBATCH --job-name=chexpert-swin-adamw
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

# For SOL Cluster, check if we are already in environemnt: $CONDA_PREFIX
if [ -z "$CONDA_PREFIX" ]; then
    echo "Activating conda environment..."
    module load mamba gcc-13.2.0-gcc-12.1.0
    source activate pytorch-gpu-2.3.1-cuda-12.1
fi

python src/train_v1_classic.py dataset=chexpert model=swin_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4
