# PyTorch Chest X-ray Classification (No MONAI)

A **simple, student-friendly** multi-label chest X-ray training example using **PyTorch + timm** for backbones (no MONAI).

## What’s included
- CSV-based dataset loader (multi-label)
- torchvision transforms
- timm backbones (resnet50, densenet121, efficientnet_b0, convnext_base, swin_base_patch4_window7_224)
- masked BCE loss to ignore uncertain labels (-1)
- AUROC/AUPR evaluation (macro mean)
- prediction CSV export

## Setup

```bash
pip install -r requirements.txt
```

## Configure (Hydra)
Edit [configs/config.yaml](configs/config.yaml) and the dataset file under [configs/dataset/](configs/dataset/) to set:
- `train_ann`, `val_ann`, `test_ann` (paths to annotation CSV files)
- `train_image_root`, `val_image_root`, `test_image_root`
- label list in `labels`

## Run

```bash
./run.sh
```

Outputs (model + predictions) are saved under the Hydra run folder, e.g. `outputs/2026-01-14/14-05-00/`, then under `artifacts/<run_id>/`.

## Hydra overrides (examples)

```bash
python src/train.py model=resnet50 train.epochs=20 train.batch_size=16
```

## Notes for students
- The dataset expects CSV columns for each label (0/1 or -1 for “ignore”).
- Update `image_path_key` if your CSV uses a different column name.
- To try another backbone, update `model.backbone` in the config (use full timm model names).
