# Specification: Dependencies

## Core Requirements (`requirements.txt`)
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `timm>=0.9.0`
- `pandas>=2.0.0`
- `Pillow>=10.0.0`
- `numpy>=1.24.0`
- `hydra-core>=1.3.0`
- `omegaconf>=2.3.0`
- `torchmetrics>=1.0.0`
- `lightning>=2.6`
- `tqdm`
- `pyyaml`

## Hardware-Specific Packages (Optional / Cluster-Provided)
- **Intel Gaudi 2 (`hpu` on ASU SOL `gaudi-pytorch-vllm` conda env)**:
  - `habana_frameworks.torch` (SynapseAI PyTorch bridge, pre-installed in `gaudi-pytorch-vllm`)
  - `lightning-habana` (optional Lightning HPU plugin for `src/train_v2_lightning.py`; falls back gracefully if absent)
- **NVIDIA CUDA (`cuda` on ASU SOL A100 / Phoenix V100 `pytorch-gpu-2.3.1-cuda-12.1` conda env)**:
  - Standard CUDA-enabled `torch` and `torchvision` (no Gaudi packages required).
