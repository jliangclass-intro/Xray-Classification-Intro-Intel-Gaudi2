# Specification: PyTorch Chest X-ray Classification (Hydra)

## Goal
Provide a minimal, PyTorch training pipeline for multi-label chest X-ray classification using timm backbones and Hydra configs.

## Key features
- CSV-based dataset with multi-label targets
- torchvision transforms, no MONAI
- Hydra-based config groups for dataset/model/train/logger
- Masked BCE loss (ignore labels == -1)
- Torchmetrics AUROC/AUPR computed via `update()`/`compute()`
- Save val + test predictions to CSV (predictions only, no ground truth) and model checkpoints (`.pt` files)
- Batch-level logging (first 20 batches, then every 10th)
- Two training scripts: classic PyTorch (`train_v1_classic.py`) and PyTorch Lightning (`train_v2_lightning.py`)
- Output analysis script (`analyze_output.py`) to find best validation epoch and calculate test metrics

## Inputs/Outputs
- Inputs: CSVs and image folders
- Outputs: All files are saved under `outputs/{dataset_name}/{run_name}/` directory:
  - Model checkpoints: `outputs/{dataset_name}/{run_name}/checkpoints/best_model.pt` and `outputs/{dataset_name}/{run_name}/checkpoints/last_model.pt` (state_dict only, `.pt` format)
  - Prediction CSVs: `outputs/{dataset_name}/{run_name}/val/preds_epochXXX_iterYYYYYY.csv` and `outputs/{dataset_name}/{run_name}/test/preds_epochXXX_iterYYYYYY.csv` (predictions only, no ground truth)
  - Training log: `outputs/{dataset_name}/{run_name}/training.log`
