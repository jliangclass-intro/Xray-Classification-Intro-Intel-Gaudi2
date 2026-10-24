# Specification: Evaluation

## Implementation
- Evaluation functions are in `src/evaluate.py`:
  - `evaluate()`: Main evaluation function that runs model inference (supporting 4D single-crop and 5D multi-crop inputs, `bfloat16` autocast, and `hpu_mark_step()` on Intel Gaudi 2) and computes metrics.
  - `calculate_multilabel_metrics()`: Computes macro `mean_auroc` and `mean_aupr` in `float32`.
  - `calculate_per_label_auroc()`: Computes per-class AUROC values.
  - `calculate_subset_metrics()`: Computes metrics for configured label subsets (`test_labels`).
  - `save_predictions()`: Saves per-sample predicted probabilities to CSV.
  - `save_val_auroc()`: Appends validation and test AUROC metrics to `auroc_list.csv`.

## Metrics & Device Placement
- Uses `torchmetrics`:
  - `MultilabelAUROC(average="macro" | "none", ignore_index=-1)`
  - `MultilabelAveragePrecision(average="macro", ignore_index=-1)`
- **Hardware Parity (`_resolve_metric_device`)**:
  - On **Intel Gaudi 2 (`hpu`)**, `torchmetrics` computation is automatically routed to `"cpu"` (`float32` / `int64`) so 64-bit integer casting, dynamic masking (`target != -1`), and sorting do not suffer from HPU TPC precision/fallback issues.
  - On **NVIDIA CUDA (`cuda`)**, `torchmetrics` runs natively on GPU.

## Label Subsets (`test_labels`)
- If `test_labels` is defined in the dataset config, calculates metrics for each specified subset (`top5`, `top12`, `top6`, `top14`, etc.):
  - `auroc_{subset_name}` and `aupr_{subset_name}` for both validation and test splits.

## Multi-Crop Evaluation Strategies (`train_v2_lightning.py`)
- Supports evaluating multiple crop strategies per validation step (configured via `validation.strategies`, e.g., `["nocrop", "center_crop", "ten_crop"]`).
- In multi-GPU/multi-HPU DDP mode, gathers predictions across ranks and deduplicates DDP padding by sample path before computing global metrics on all ranks.

## Saved Evaluation Files
- **Prediction CSVs (`save_predictions`)**:
  - `val/preds_epochXXX_iterYYYYYY.csv` and `test/preds_epochXXX_iterYYYYYY.csv` (plus `val_<strategy>/` and `test_<strategy>/` for `center_crop` / `ten_crop`).
  - Columns: `Path`, `__key__` (if present), and `pred_<label>` (rounded to 4 decimal places; no ground-truth columns).
- **AUROC Summary (`save_val_auroc`)**:
  - `auroc_list.csv` in the run output directory, recording `epoch`, `iter`, `val_mean_auroc`, `test_mean_auroc`, and subset/crop AUROC columns at each evaluation step.

