# X-ray Classification Intro

Multi-label chest X-ray classification framework (ChestX-ray14, CheXpert, MIMIC-CXR) using PyTorch and `timm`.

> **Hardware Focus & Compatibility (`Intel Gaudi 2` + `NVIDIA CUDA`):**
> This repository is primarily geared toward running and benchmarking on **Intel Gaudi 2 (Habana HPU)** accelerators (such as the `gaudi*` nodes on the ASU SOL cluster), while maintaining **full dual compatibility with NVIDIA CUDA GPUs** (e.g., A100 on SOL, V100 on Phoenix, or local CUDA workstations) and CPU fallback. The environment scripts (`setup_env.sh`) and training code (`src/utils.py`, `src/train_v1_classic.py`, `src/train_v2_lightning.py`) automatically detect and configure the active hardware backend (`cuda`, `hpu`, or `cpu`).

## Documentation

- **[Intel Gaudi 2 (HPU) Guide (`Pytorch/INTEL_GAUDI2_GUIDE.md`)](./Pytorch/INTEL_GAUDI2_GUIDE.md)**: Primary guide for running on Intel Gaudi 2, explaining why standard CUDA-oriented PyTorch scripts underperform on Gaudi 2 and what is needed (`htcore.mark_step()`, FP32 loss reduction, CPU metric evaluation, and `lightning-habana`) to achieve CUDA-equivalent AUROC.
- **[PyTorch Classification Guide (`Pytorch/README.md`)](./Pytorch/README.md)**: Project overview, configuration, SLURM scripts, and environment setup on ASU SOL (Gaudi 2 & A100) and PHXRC (V100) clusters.
- **[SLURM `sbatch` Examples (`Pytorch/sbatch-examples.md`)](./Pytorch/sbatch-examples.md)**: Example `sbatch` commands for SOL and Phoenix.

## References
- https://github.com/jlianglab/BenchmarkTransformers
- https://github.com/MR-HosseinzadehTaher/BenchmarkTransferLearning/
