# X-ray Classification Intro

Multi-label chest X-ray classification framework (ChestX-ray14, CheXpert, MIMIC-CXR) using PyTorch and `timm`.

## Documentation

- **[PyTorch Classification Guide (`Pytorch/README.md`)](./Pytorch/README.md)**: Project overview, configuration, SLURM scripts, and environment setup on ASU SOL and PHXRC clusters.
- **[Intel Gaudi 2 (HPU) Guide (`Pytorch/INTEL_GAUDI2_GUIDE.md`)](./Pytorch/INTEL_GAUDI2_GUIDE.md)**: Why standard CUDA-oriented PyTorch scripts underperform on Intel Gaudi 2 and what is needed (`htcore.mark_step()`, FP32 loss reduction, CPU metric evaluation, and `lightning-habana`) to achieve CUDA-equivalent AUROC on Gaudi 2.
- **[SLURM `sbatch` Examples (`Pytorch/sbatch-examples.md`)](./Pytorch/sbatch-examples.md)**: Example `sbatch` commands for SOL (A100) and Phoenix (V100).

## References
- https://github.com/jlianglab/BenchmarkTransformers
- https://github.com/MR-HosseinzadehTaher/BenchmarkTransferLearning/
