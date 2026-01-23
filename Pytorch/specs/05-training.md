# Specification: Training

## Loss
- Use BCEWithLogits with masking for labels == -1.

## Optimizers
- `adamw` or `sgd`
- Weight decay and momentum supported.

## Loop
- Standard epoch loop with tqdm progress bar.
- Log every batch for first 20 batches, then every 10th batch.
- Log format includes: rank, dataset name, optimizer, epoch, global step, batch index, batch timing, data timing, learning rates, loss, elapsed time, ETA.
- All INFO level logs are saved to `training.log` file under `outputs/{dataset_name}/{run_name}/` directory.
- Save `checkpoints/last_model.pt` at end of training under `outputs/{dataset_name}/{run_name}/checkpoints/` directory.
- Save `checkpoints/best_model.pt` when monitored metric improves (based on `monitor` and `monitor_mode`) under `outputs/{dataset_name}/{run_name}/checkpoints/` directory.

## Early Stopping
- Early stopping based on `patience` (default: 10 epochs).
- Monitors the validation metric specified by `monitor` (default: `mean_auroc`).
- Stops training if the monitored metric does not improve for `patience` consecutive evaluation epochs.
- Improvement is determined by `monitor_mode`: `max` (higher is better) or `min` (lower is better).
- When early stopping triggers, logs the best metric value and the epoch at which it was achieved.
- Training always saves `checkpoints/last_model.pt` even if early stopping occurs.
