# Specification: Dataset

## CSV format
- Must contain a column with image paths (default `Path`).
- Optionally contains a `__key__` column for unique sample identification.
- Must contain one column per label with values:
  - `1` positive
  - `0` negative
  - `-1` or NaN = ignore (masked out in loss and metrics)

## Dataset class (`src/dataset.py`)
- Reads CSV using `pandas`.
- Resolves candidate cluster root directories (`resolve_data_root()` in `src/train_v2_lightning.py`) so configs work across `/scratch/...` and `/data/jliang12/...` without manual editing.
- Converts relative paths to full paths using `make_path(image_root, p, image_path_append)`.
- Loads image with `PIL.Image`, converts to RGB.
- Applies `torchvision` transforms:
  - Training: `Resize((image_size, image_size))`, `RandomHorizontalFlip()`, `RandomRotation(10)`, `ToTensor()`, `Normalize(...)`
  - Evaluation (`nocrop`, `center_crop`, `ten_crop`): configurable via `validation.strategies`
- Uses `pin_memory=(device_type == "cuda")` on `DataLoader` instances (enabled on NVIDIA CUDA GPUs, disabled on Intel Gaudi 2 HPU and CPU).
- Returns 5-tuple: `(image_tensor, label_tensor, full_image_path, original_path, original_key)`
  - `image_tensor`: transformed image tensor (4D `(C, H, W)` or 5D `(n_crops, C, H, W)` for `ten_crop`)
  - `label_tensor`: float32 label tensor with values `1.0` (positive), `0.0` (negative), `-1.0` (ignore)
  - `full_image_path`: resolved path after `image_root` and `image_path_append` are applied
  - `original_path`: original path value from CSV (before `image_root` is applied)
  - `original_key`: value from `__key__` column in CSV (if present, otherwise `None`)
- Validation: random path existence check (`_validate_paths`) at initialization to fail fast if dataset paths are misconfigured.

