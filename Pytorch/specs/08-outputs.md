# Specification: Outputs

## Output locations
- Hydra run directory: `outputs/${dataset.name}/` (organized by dataset name)
- Training output directory: `outputs/{dataset_name}/{run_name}/`
  - `dataset_name`: dataset name (e.g., `ChestXray14`, `CheXpert`, `MIMIC`)
  - `run_name` format: `{dataset_name}_{backbone}_{optimizer}_{lr}_{date}_{time}` (e.g., `ChestXray14_resnet50_adamw_0_0001_20241201_143022`)
    - `backbone`: model backbone name (e.g., `resnet50`, `convnext_base`)
    - `optimizer`: optimizer type (e.g., `adamw`, `sgd`)
    - `lr`: learning rate with dots replaced by underscores (e.g., `0_0001` for `0.0001`, `1_00e-4` for `1e-4`)
    - `date` format: `YYYYMMDD` (e.g., `20241201`)
    - `time` format: `HHMMSS` (e.g., `143022`)
  - Example: `outputs/ChestXray14/ChestXray14_resnet50_adamw_0_0001_20241201_143022/`

## Files (all under `outputs/{dataset_name}/{run_name}/`)
- `training.log` - training log file (contains all INFO level logs from training)
- `.hydra/` - Hydra configuration snapshot directory (contains config.yaml and other Hydra files)
- `checkpoints/best_model.pt` - saved when metric improves (state_dict only, .pt format)
- `checkpoints/last_model.pt` - saved at end of training (state_dict only, .pt format)
- `val/preds_epochXXX_iterYYYYYY.csv` - validation predictions (probabilities rounded to 4 decimal places)
  - Columns: `Path`, `__key__` (if exists in original dataset), `pred_<label>` for each label (no ground truth columns)
  - `Path`: original image path from dataset CSV (before image_root is applied)
  - `__key__`: original `__key__` column value from dataset CSV (if present)
- `test/preds_epochXXX_iterYYYYYY.csv` - test predictions (probabilities rounded to 4 decimal places)
  - Columns: `Path`, `__key__` (if exists in original dataset), `pred_<label>` for each label (no ground truth columns)
  - `Path`: original image path from dataset CSV (before image_root is applied)
  - `__key__`: original `__key__` column value from dataset CSV (if present)

## Complete file paths
All files are located under the `{run_name}` directory:
- `outputs/{dataset_name}/{run_name}/training.log`
- `outputs/{dataset_name}/{run_name}/checkpoints/best_model.pt`
- `outputs/{dataset_name}/{run_name}/checkpoints/last_model.pt`
- `outputs/{dataset_name}/{run_name}/val/preds_epochXXX_iterYYYYYY.csv`
- `outputs/{dataset_name}/{run_name}/test/preds_epochXXX_iterYYYYYY.csv`
