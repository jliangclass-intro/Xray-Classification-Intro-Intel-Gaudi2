# Specification: Evaluation

## Implementation
- Evaluation functions are in `src/evaluate.py`:
  - `evaluate()`: Main evaluation function that runs model inference and computes metrics
  - `calculate_subset_metrics()`: Calculate metrics for label subsets
  - `multilabel_metrics()`: Calculate multilabel metrics from probabilities and targets

## Metrics
- Use torchmetrics:
  - `MultilabelAUROC`
  - `MultilabelAveragePrecision`
- Call `update()` on each batch and `compute()` at end.
- Use `ignore_index=-1` to skip uncertain labels.
- Report `mean_auroc` and `mean_aupr` (macro) for all labels.

## Label Subsets (test_labels)
- If `test_labels` is defined in the dataset config, calculate metrics for specified label subsets.
- `test_labels` is a dictionary mapping subset names to lists of label names.
- Common subsets: `top5`, `top12`, `top6`, `top14`, etc.
- For each subset, calculate:
  - `{split}_auroc_{subset_name}` (e.g., `val_auroc_top5`, `test_auroc_top12`)
  - `{split}_aupr_{subset_name}` (e.g., `val_aupr_top5`, `test_aupr_top12`)
- Metrics are calculated for both validation and test sets.
- Example config:
  ```yaml
  test_labels:
    top5:
      - Atelectasis
      - Cardiomegaly
      - Consolidation
      - Edema
      - Pleural Effusion
    top12:
      - Atelectasis
      - Cardiomegaly
      - Consolidation
      - Edema
      - Enlarged Cardiomediastinum
      - Fracture
      - Lung Lesion
      - Lung Opacity
      - Pleural Effusion
      - Pleural Other
      - Pneumonia
      - Pneumothorax
  ```

## Predictions
- Save both validation and test predictions at each evaluation step.
- Saved in split-specific subdirectories: `val/` and `test/`
- Filename format:
  - `val/preds_epochXXX_iterYYYYYY.csv`
  - `test/preds_epochXXX_iterYYYYYY.csv`
- CSV columns (predictions only, no ground truth):
  - `Path`: original image path from dataset CSV (before image_root is applied)
  - `__key__`: original `__key__` column value from dataset CSV (if present in original dataset)
  - `pred_<label>`: prediction probability for each label (rounded to 4 decimal places)
- Implementation: `save_predictions()` function in `src/evaluate.py` (shared by both `train_v1_classic.py` and `train_v2_lightning.py`)
