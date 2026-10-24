# Specification: Hydra Configuration

## Root config (`configs/config.yaml`)
- Includes default config groups:
  - `dataset`: `chestxray14`
  - `model`: `resnet50`
  - `train`: `default`
  - `validation`: `three_validation`
  - `scheduler`: `reduce_lr_on_plateau`
  - `logger`: `aim`
- Uses `hydra.job.chdir: true` so outputs are written relative to `hydra.run.dir`.
- Hydra's internal logging is disabled via `override hydra/hydra_logging: none` and `override hydra/job_logging: none` so training scripts manage their own `training.log` file inside `{run_name}/`.
- `run_name` is constructed in `config.yaml` using Hydra resolvers:
  `${dataset.name}_${model.backbone}_${train.optimizer}_${format_lr:${train.lr}}_${oc.env:RUN_TIMESTAMP,run}`
  - Uses custom `format_lr` resolver (registered with `replace=True` in training scripts) to replace `.` with `_`.
  - Can also be overridden via `RUN_ID` environment variable or `run_name=...` CLI argument.
- `.hydra` configuration snapshots are saved in `{hydra.run.dir}/{run_name}/.hydra` via `hydra.output_subdir: ${run_name}/.hydra`.
- `debug: false` — when overridden with `debug=true`, runs 1 training batch, runs evaluation once, and exits cleanly.

## Dataset config fields (`configs/dataset/*.yaml`)
- `name`: dataset name (`ChestXray14`, `CheXpert`, `MIMIC`)
- `train_ann`, `val_ann`, `test_ann`: paths to annotation CSV files
- `train_image_root`, `val_image_root`, `test_image_root`: single root folder path or a list of candidate cluster root paths (resolved to the first existing directory via `resolve_data_root()`)
- `image_path_key`: column name in CSV (default `Path`)
- `image_path_append`: optional suffix
- `labels`: list of label names
- `num_classes`: int
- `test_labels`: optional dict mapping subset names (e.g., `top5`, `top12`, `top14`) to lists of label names

## Model config fields (`configs/model/*.yaml`)
- `backbone`: full `timm` model name, e.g., `resnet50`, `convnext_base`, `swin_base_patch4_window7_224` (config file: `swin_base.yaml`)
- `pretrained`: bool (`true`)
- `image_size`: int (`224`)
- `normalize`: `imagenet` or `none`

## Train config fields (`configs/train/default.yaml`)
- `batch_size` (default `128`), `num_workers` (`4`), `epochs` (`100`)
- `lr` (`1e-4`), `weight_decay` (`1e-4`), `optimizer` (`adamw`, `adam`, or `sgd`), `momentum` (`0.9`)
- `eval_every_iter` (`1000`): run validation and test evaluation every N training iterations
- `seed` (`42`)
- `output_dir` (`artifacts`)
- `save_predictions` (`true`), `save_best_model` (`true`)
- `monitor` (`mean_auroc_no_crop` or `mean_auroc`), `monitor_mode` (`max`), `patience` (`15` evaluation steps)

## Validation & Scheduler config groups
- `configs/validation/*.yaml`: defines `strategies` list (`["nocrop"]`, `["center_crop"]`, `["ten_crop"]`, or all three in `three_validation.yaml`).
- `configs/scheduler/*.yaml`: configures `reduce_lr_on_plateau` (`mode`, `factor`, `patience`, `threshold`, `cooldown`, `min_lr`, `warmup_epochs`), `cosine_annealing` (`T_max`, `eta_min`), or `none`.
