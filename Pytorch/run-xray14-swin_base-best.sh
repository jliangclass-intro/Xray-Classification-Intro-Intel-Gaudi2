#!/usr/bin/env bash
#SBATCH --job-name=xray14-convnext-adam
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

cd "$(dirname "$0")" || exit 1
source setup_env.sh

# for Nvidia a100, use batch size = 128
# for Nvidia v100, use batch size = 64

DATASET="chestxray14"
MODEL="swin_base"
OPTIMIZER="adamw"
LR="2e-4"
BATCH_SIZE=64

OUTPUT_DIR="$SCRATCH/outputs-Xray-Classification-Intro/$CLUSTER"
mkdir -p $OUTPUT_DIR

python ${MY_PYTHON_CODE:-src/train_v2_lightning.py} dataset=$DATASET model=$MODEL train.batch_size=$BATCH_SIZE train.optimizer=$OPTIMIZER train.lr=$LR scheduler=reduce_lr_on_plateau hydra.run.dir=$OUTPUT_DIR "$@"
