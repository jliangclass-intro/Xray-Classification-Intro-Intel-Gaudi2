# Specification: Training

## Hardware & Mixed Precision (`Intel Gaudi 2` & `NVIDIA CUDA`)
- **Device Detection (`src/utils.py`)**:
  - Automatically imports `habana_frameworks.torch.core as htcore` when available and selects `'cuda'`, `'hpu'`, or `'cpu'`.
- **Mixed Precision**:
  - **Intel Gaudi 2 (`hpu`)**: Uses `bfloat16` mixed precision (`bf16-mixed`).
  - **NVIDIA Ampere+ GPUs (e.g., A100 on SOL, compute capability $\ge 8.0$)**: Uses `bfloat16` mixed precision (`bf16-mixed`).
  - **NVIDIA Pre-Ampere GPUs (e.g., V100 on Phoenix, compute capability $< 8.0$)**: Automatically falls back to `float16` (`16-mixed` AMP with `GradScaler`).
- **Intel Gaudi 2 Lazy Mode Step Handling**:
  - Calls `hpu_mark_step()` (`htcore.mark_step()`) after `loss.backward()` and `optimizer.step()` in `src/train_v1_classic.py` and in `on_before_optimizer_step` / `optimizer_step` in `src/train_v2_lightning.py`.
  - Uses `lightning-habana` (`HPUAccelerator`, `SingleHPUStrategy` / `HPUParallelStrategy`, `HPUPrecisionPlugin`) when available in `src/train_v2_lightning.py`.

## Loss (`MaskedBCEWithLogitsLoss`)
- Binary cross-entropy with logits that masks out uncertain/missing labels (`targets == -1`).
- Explicitly upcasts `logits` and `targets` to `float32` before computing `F.binary_cross_entropy_with_logits` and `loss.sum() / denom` so `bfloat16` sum reduction over `batch_size * num_classes` never loses mantissa precision on Intel Gaudi 2 or Ampere+ GPUs.

## Optimizers & Schedulers
- **Optimizers**: `adamw`, `adam`, or `sgd` (with `lr`, `weight_decay`, and `momentum`).
- **Schedulers**:
  - `reduce_lr_on_plateau` (with optional `LinearLR` warmup via `SequentialLR`)
  - `cosine_annealing` (`CosineAnnealingLR`)
  - `none`

## Training Loop & Checkpointing
- Evaluates every `eval_every_iter` iterations (default `1000`).
- Logs every batch for the first 20 batches, then every 10th batch (includes rank, dataset name, optimizer, epoch, global step, batch index, batch time, data time, learning rates, loss, elapsed time, and ETA).
- All `INFO` logs are written to `training.log` inside the run output directory.
- Model weights are copied to CPU via `_state_dict_cpu()` before `torch.save()` to avoid HPU/Gaudi serialization errors:
  - `checkpoints/best_model.pt` — saved whenever the monitored validation metric improves.
  - `checkpoints/epoch_XXX.pt` (Lightning) or `checkpoints/epoch_X.pt` (Classic) — saved at the end of each epoch.
  - `checkpoints/last_model.pt` — saved at the end of training in `train_v1_classic.py`.

## Early Stopping
- Monitors the validation metric specified by `train.monitor` (e.g., `mean_auroc_no_crop` or `mean_auroc`) with mode `train.monitor_mode` (`max` or `min`).
- Stops training if the monitored metric does not improve for `train.patience` consecutive evaluation checks (default `15`).

