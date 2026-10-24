# Specification: PyTorch Chest X-ray Classification (Hydra)

## Goal
Provide a clean, reproducible PyTorch training and evaluation pipeline for multi-label chest X-ray classification using `timm` backbones and Hydra configs, **primarily geared toward Intel Gaudi 2 (Habana HPU)** accelerators while **fully supporting NVIDIA CUDA GPUs** (A100, V100, etc.) and CPU fallback.

## Key features
- **Dual Hardware Support (Intel Gaudi 2 & NVIDIA CUDA)**:
  - Automatic accelerator detection (`hpu`, `cuda`, `cpu`) via `get_best_accelerator()` in `src/utils.py` and cluster environment selection via `setup_env.sh` (`gaudi-pytorch-vllm` on SOL Gaudi 2 nodes, `pytorch-gpu-2.3.1-cuda-12.1` on SOL A100 and Phoenix V100 nodes).
  - Gaudi 2 Lazy Mode graph step management (`hpu_mark_step()`), FP32 loss reduction inside `MaskedBCEWithLogitsLoss`, CPU metric computation on HPU for exact numerical parity, and `lightning-habana` integration.
- CSV-based dataset (`CSVDataset`) with multi-label targets and multi-path cluster root resolution (`resolve_data_root()`).
- `torchvision` transforms supporting `nocrop`, `center_crop`, and `ten_crop` evaluation strategies.
- Hydra-based config groups for `dataset`, `model`, `train`, `optimizer`, `scheduler`, `validation`, and `logger`.
- Masked BCE loss (`MaskedBCEWithLogitsLoss`, ignores labels `== -1`) computed in `float32`.
- `torchmetrics` AUROC/AUPR (`MultilabelAUROC`, `MultilabelAveragePrecision`) with per-label and subset (`top5`, `top12`, `top14`) reporting.
- Save validation and test predictions to CSV (predictions only, no ground truth), per-evaluation AUROC history (`auroc_list.csv`), and model checkpoints (`.pt` state_dict files).
- Batch-level logging (first 20 batches, then every 10th).
- Two training scripts:
  - PyTorch Lightning (`src/train_v2_lightning.py`, default in `run-*.sh`)
  - Classic PyTorch (`src/train_v1_classic.py`, selectable via `MY_PYTHON_CODE=src/train_v1_classic.py`)

## Inputs/Outputs
- **Inputs**: Annotation CSVs (`data/xray14/`, `data/chexpert/`, `data/mimic_cxr/`) and chest X-ray image directories.
- **Outputs**: All run artifacts are saved under `{hydra.run.dir}/{run_name}/`:
  - Model checkpoints: `checkpoints/best_model.pt`, `checkpoints/epoch_XXX.pt` (or `epoch_X.pt`), and `checkpoints/last_model.pt` (CPU `state_dict`, `.pt` format)
  - AUROC summary CSV: `auroc_list.csv` (records validation and test AUROC at every evaluation step)
  - Prediction CSVs: `val/preds_epochXXX_iterYYYYYY.csv` and `test/preds_epochXXX_iterYYYYYY.csv` (plus `val_<strategy>/` and `test_<strategy>/` when multi-crop validation is enabled)
  - Training log: `training.log`

