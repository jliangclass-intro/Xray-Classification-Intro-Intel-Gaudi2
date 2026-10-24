# Specification: Environment & Run Scripts

## `setup_env.sh`
- Automatically detects the execution environment (Slurm `sol` or `phoenix`, or local workstation) and activates the appropriate Conda/venv environment:
  - **ASU SOL (`gaudi*` hostnames)**: Loads `mamba` and activates `gaudi-pytorch-vllm` (Intel Gaudi 2 SynapseAI PyTorch).
  - **ASU SOL (A100 GPU nodes)**: Loads `mamba gcc-13.2.0-gcc-12.1.0` and activates `pytorch-gpu-2.3.1-cuda-12.1`.
  - **ASU Phoenix (V100 GPU nodes)**: Loads `mamba gcc-12.3.0-vw` and activates `pytorch-gpu-2.3.1-cuda-12.1`.
  - **Local Workstation**: Activates `$HOME/.venvs/torch-cuda/bin/activate` if present.

## `run.sh`
- Sources `setup_env.sh` and runs `python ${MY_PYTHON_CODE:-src/train_v2_lightning.py} "$@"` with passthrough of all Hydra CLI overrides.
- Example usage:
  - `./run.sh dataset=chestxray14 model=resnet50 train.batch_size=64 train.optimizer=adam train.lr=2e-4`
  - `MY_PYTHON_CODE=src/train_v1_classic.py ./run.sh dataset=chestxray14 model=resnet50`

## Dataset & Backbone SLURM Scripts (`run-*-best.sh` & `run-*-adamw.sh`)
- Pre-configured SLURM batch scripts for all 3 datasets (`xray14`, `chexpert`, `mimic`) and 3 backbones (`resnet50`, `convnext-base`, `swin_base`):
  - `run-xray14-resnet50-best.sh`, `run-xray14-convnext-base-best.sh`, `run-xray14-swin_base-best.sh`
  - `run-chexpert-resnet50-best.sh`, `run-chexpert-convnext-base-best.sh`, `run-chexpert-swin_base-best.sh`
  - `run-mimic-resnet50-best.sh`, `run-mimic-convnext-base-best.sh`, `run-mimic-swin_base-best.sh`
- Each script sources `setup_env.sh`, sets `OUTPUT_DIR="/scratch/$USER/outputs-Xray-Classification-Intro/$CLUSTER"`, and invokes `${MY_PYTHON_CODE:-src/train_v2_lightning.py}` while forwarding any extra CLI overrides (`"$@"`).
- See `sbatch-examples.md` and `INTEL_GAUDI2_GUIDE.md` for `sbatch` submission examples on SOL (Gaudi 2 & A100) and Phoenix (V100).

