# Specification: Run Scripts

## `run.sh`
- Calls `python src/train_v2_lightning.py` with passthrough of all arguments
- Handles conda environment activation and module loading
- Example usage:
  - `./run.sh dataset=chestxray14 model=convnext_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4`
  - `./run.sh dataset=chestxray14 model=resnet50 train.batch_size=128 train.optimizer=adamw train.lr=1e-4`
  - `./run.sh dataset=chestxray14 model=swin_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4`

## `submit-all-slurm.sh`
- SLURM batch submission script that submits 9 training jobs
- Datasets: `chestxray14`, `chexpert`, `mimic`
- Backbones: `convnext_base`, `resnet50`, `swin_base`
- Training config: `batch_size=128`, `optimizer=adamw`, `lr=1e-4`
- Each job runs independently with unique job name: `{dataset}-{backbone}`
- The `submit_job()` function accepts a complete `bash run.sh` command string and wraps it with `sbatch --wrap`
- Example: `submit_job "bash run.sh dataset=chestxray14 model=convnext_base train.batch_size=128 train.optimizer=adamw train.lr=1e-4"`
- Usage: `./submit-all-slurm.sh`
- Check job status: `squeue -u $USER`
