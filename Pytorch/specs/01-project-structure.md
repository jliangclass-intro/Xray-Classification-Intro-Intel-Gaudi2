# Specification: Project Structure

```
Pytorch/
  configs/
    config.yaml
    dataset/
      chestxray14.yaml
      chexpert.yaml
      mimic.yaml
    logger/
      aim.yaml
      wandb.yaml
    model/
      resnet50.yaml
      convnext_base.yaml
      swin_base.yaml
    optimizer/
      adam.yaml
      adamw.yaml
      sgd.yaml
    scheduler/
      reduce_lr_on_plateau.yaml
      cosine_annealing.yaml
      none.yaml
    train/
      default.yaml
    validation/
      default.yaml
      nocrop.yaml
      center_crop.yaml
      ten_crop.yaml
      three_validation.yaml
  data/
    chexpert/
    mimic_cxr/
    xray14/
  src/
    train_v1_classic.py
    train_v2_lightning.py
    dataset.py
    utils.py
    evaluate.py
  INTEL_GAUDI2_GUIDE.md
  README.md
  sbatch-examples.md
  setup_env.sh
  run.sh
  run-xray14-*.sh
  run-chexpert-*.sh
  run-mimic-*.sh
  requirements.txt
  specs/
```

## Rules
- `configs/config.yaml` defines Hydra defaults (`dataset`, `model`, `train`, `validation`, `scheduler`, `logger`) and output directory layout.
- Dataset, model, train, optimizer, scheduler, and validation configs are selectable via Hydra CLI overrides.
- `src/train_v2_lightning.py` - PyTorch Lightning training script (default entrypoint in `run-*.sh` and `run.sh`).
- `src/train_v1_classic.py` - Classic PyTorch training script (selectable via `MY_PYTHON_CODE=src/train_v1_classic.py`).
- `INTEL_GAUDI2_GUIDE.md` - Troubleshooting and best-practices guide for Intel Gaudi 2 (HPU) and comparison with NVIDIA CUDA.

