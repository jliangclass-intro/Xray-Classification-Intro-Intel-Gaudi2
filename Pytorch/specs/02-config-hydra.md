# Specification: Hydra Configuration

## Root config (`configs/config.yaml`)
- Must include defaults:
  - dataset: `chestxray14`
  - model: `resnet50`
  - train: `default`
- Hydra run directory: `outputs/${dataset.name}` (organized by dataset name)
- Use `hydra.job.chdir: true` so outputs are local to run dir.
- Hydra's internal logging is disabled via `override hydra/hydra_logging: none` and `override hydra/job_logging: none` to prevent automatic log file creation (like `train_v2_lightning.log`) - training scripts create their own `training.log` file in the `{run_name}` directory.
- `run_name` is constructed in `config.yaml` using Hydra resolvers: `${dataset.name}_${model.backbone}_${train.optimizer}_${format_lr:${train.lr}}_${now:%Y%m%d}_${now:%H%M%S}`
  - Uses custom `format_lr` resolver (registered in training scripts) to format learning rate (replaces `.` with `_`)
  - Uses `${now:%Y%m%d}` for date and `${now:%H%M%S}` for time
- The `.hydra` directory is automatically placed in `outputs/{dataset_name}/{run_name}/.hydra` via `hydra.output_subdir: ${run_name}/.hydra` since `run_name` is known at config load time.

## Dataset config fields
- `name`: dataset name
- `train_ann`, `val_ann`, `test_ann`: paths to annotation CSV files
- `train_image_root`, `val_image_root`, `test_image_root`: root folders
- `image_path_key`: column name in CSV (default `Path`)
- `image_path_append`: optional suffix
- `labels`: list of label names
- `num_classes`: int

## Model config fields
- `backbone`: full timm model name, e.g. `resnet50`, `densenet121`, `efficientnet_b0`, `convnext_base`, `swin_base_patch4_window7_224` (config file: `swin_base.yaml`)
- `pretrained`: bool
- `image_size`: int
- `normalize`: `imagenet` or `none`

## Train config fields
- `batch_size`, `num_workers`, `epochs`
- `lr`, `weight_decay`, `optimizer`, `momentum`
- `eval_every`
- `seed`
- `output_dir` (relative to Hydra run dir)
- `save_predictions`, `save_best_model`
- `monitor`, `monitor_mode`, `patience` (default: 10, early stopping patience in epochs)