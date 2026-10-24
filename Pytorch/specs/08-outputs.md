# Specification: Outputs

## Output locations
- Base Hydra run directory (`hydra.run.dir`):
  - Default in `run-*-best.sh`: `/scratch/$USER/outputs-Xray-Classification-Intro/$CLUSTER`
  - Default in `configs/config.yaml`: `output-cuda-$(date +%Y%m%d)-$(date +%H%M%S)` (can be overridden with `hydra.run.dir=...`)
- Run subdirectory: `{hydra.run.dir}/{run_name}/`
  - `run_name` format: `${dataset.name}_${model.backbone}_${train.optimizer}_${format_lr:${train.lr}}_${oc.env:RUN_TIMESTAMP,run}` (or overridden via `RUN_ID` / `run_name=...`)

## Files (all under `{hydra.run.dir}/{run_name}/`)
- `training.log` — training log file containing all `INFO`-level configuration, batch progress, and evaluation logs.
- `.hydra/` — Hydra configuration snapshot directory (`config.yaml`, `hydra.yaml`, `overrides.yaml`).
- `auroc_list.csv` — CSV table tracking `epoch`, `iter`, `val_mean_auroc`, `test_mean_auroc`, and subset/crop AUROC scores across all evaluation steps.
- `checkpoints/best_model.pt` — saved whenever the monitored validation metric improves (CPU `state_dict`, `.pt` format).
- `checkpoints/epoch_XXX.pt` (or `epoch_X.pt` in `train_v1_classic.py`) — saved at the end of each epoch.
- `checkpoints/last_model.pt` — saved at the end of training in `train_v1_classic.py`.
- `val/preds_epochXXX_iterYYYYYY.csv` (and `val_center_crop/`, `val_ten_crop/` when multi-crop validation is enabled) — validation predictions rounded to 4 decimal places (`Path`, optional `__key__`, and `pred_<label>` columns).
- `test/preds_epochXXX_iterYYYYYY.csv` (and `test_center_crop/`, `test_ten_crop/` when multi-crop validation is enabled) — test predictions rounded to 4 decimal places (`Path`, optional `__key__`, and `pred_<label>` columns).

