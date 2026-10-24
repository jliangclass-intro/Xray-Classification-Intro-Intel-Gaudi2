# PyTorch Chest X-ray Classification

A **simple and clean** multi-label chest X-ray classification framework using **PyTorch + timm** for training and evaluating deep learning models on medical imaging datasets.

## Overview

This project provides a complete training pipeline for multi-label chest X-ray classification tasks. It supports multiple datasets (ChestX-ray14, CheXpert, MIMIC-CXR), various model architectures (ResNet, ConvNeXt, Swin Transformer), and includes comprehensive evaluation metrics with per-label AUROC reporting.

### Key Features

- **Multi-label Classification**: Handles multiple pathology labels per image with support for uncertain labels (-1)
- **Multiple Datasets**: Pre-configured support for ChestX-ray14, CheXpert, and MIMIC-CXR
- **Flexible Model Architectures**: Easy integration with any timm model (ResNet50, ConvNeXt Base, Swin Base, and more)
- **Comprehensive Evaluation**: 
  - Mean AUROC and AUPR metrics
  - Per-label AUROC display after each epoch
  - Subset metrics for specific label groups (e.g., top5, top12)
- **Hydra Configuration**: Flexible configuration management with easy overrides
- **SLURM Support**: Ready-to-use bash scripts for cluster submission
- **Prediction Export**: CSV exports of predictions for further analysis

## Project Structure

```
Pytorch/
├── src/
│   ├── train_v2_lightning.py  # Main PyTorch Lightning training script
│   ├── train_v1_classic.py    # Classic PyTorch training script
│   ├── dataset.py             # CSV-based dataset loader
│   ├── evaluate.py            # Evaluation metrics and AUROC calculation
│   └── utils.py               # Utility functions
├── configs/
│   ├── config.yaml            # Main configuration file
│   ├── dataset/               # Dataset-specific configs
│   │   ├── chestxray14.yaml
│   │   ├── chexpert.yaml
│   │   └── mimic.yaml
│   ├── model/                 # Model architecture configs
│   │   ├── resnet50.yaml
│   │   ├── convnext_base.yaml
│   │   └── swin_base.yaml
│   └── train/                 # Training hyperparameters
├── run-*.sh                   # SLURM submission scripts
├── INTEL_GAUDI2_GUIDE.md      # Guide for running & debugging on Intel Gaudi 2 HPU
└── outputs/                   # Training outputs (created during training)
```

## Supported Datasets

- **ChestX-ray14**: 14 common thoracic pathology labels
- **CheXpert**: Large-scale chest X-ray dataset with uncertainty labels
- **MIMIC-CXR**: MIMIC Chest X-ray dataset

## Model Architectures

Pre-configured support for:
- **ResNet50**: Classic CNN architecture
- **ConvNeXt Base**: Modern CNN with improved design
- **Swin Base**: Vision Transformer with hierarchical structure

Additional timm models can be easily added by updating the config.

## Training Workflow

1. **Data Loading**: CSV-based annotation files with image paths and multi-label targets
2. **Data Augmentation**: Random horizontal flip and rotation during training
3. **Model Training**: 
   - Masked BCE loss (ignores uncertain labels marked as -1)
   - AdamW or SGD optimizer
   - Learning rate scheduling support
4. **Evaluation**: 
   - Validation and test set evaluation after each epoch
   - Mean AUROC and per-label AUROC displayed
   - Best model checkpointing based on validation metrics
5. **Output**: 
   - Model checkpoints (best and last)
   - Prediction CSVs for validation and test sets
   - AUROC tracking CSV (`auroc_list.csv`)
   - Training logs

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

The project uses Hydra for configuration management. Main configuration files:

- **`configs/config.yaml`**: Main config with dataset, model, and training settings
- **`configs/dataset/*.yaml`**: Dataset-specific paths and labels
- **`configs/model/*.yaml`**: Model architecture settings
- **`configs/train/*.yaml`**: Training hyperparameters

### Key Configuration Options

- Dataset paths: `train_ann`, `val_ann`, `test_ann` (CSV annotation files)
- Image roots: `train_image_root`, `val_image_root`, `test_image_root`
- Labels: List of pathology labels in `labels`
- Model: Backbone architecture in `model.backbone`
- Training: Batch size, learning rate, epochs, optimizer in `train.*`

## Running Training

### Using SLURM Scripts (Recommended)

Pre-configured scripts for each dataset and model combination. These scripts inherently set safe defaults (`48:00:00`, `8` CPUs, `64G` RAM, `1` A100 GPU) and automatically `cd` into the correct directory to avoid Slurm initial path confusion.

```bash
# ChestX-ray14
./run-xray14-resnet50-best.sh
./run-xray14-convnext-base-best.sh
./run-xray14-swin_base-best.sh

# CheXpert
./run-chexpert-resnet50-best.sh
./run-chexpert-convnext-base-best.sh
./run-chexpert-swin_base-best.sh

# MIMIC-CXR
./run-mimic-resnet50-best.sh
./run-mimic-convnext-base-best.sh
./run-mimic-swin_base-best.sh
```

### Modify configuration
```bash
# ./run-xray14-convnext-base-best.sh dataset.dataset_root=$XRAY_DATASET
# ./run-xray14-convnext-base-best.sh debug=True dataset.dataset_root=$XRAY_DATASET train.batch_size=128
```

### Intel Gaudi 2 HPU
See **[`INTEL_GAUDI2_GUIDE.md`](./INTEL_GAUDI2_GUIDE.md)** for full instructions on running on Intel Gaudi 2 (`gaudi*` nodes on SOL) and required HPU adjustments (`htcore.mark_step()`, FP32 loss reduction, CPU metric evaluation, and `lightning-habana`) to match CUDA training accuracy:
```bash
HABANA_VISIBLE_DEVICES=0 MY_PYTHON_CODE=src/train_v1_classic.py ./run-xray14-resnet50-best.sh train.batch_size=64
```

### Direct Python Execution

```bash
python src/train_v2_lightning.py dataset=chestxray14 model=resnet50 train.batch_size=128 train.optimizer=adamw train.lr=1e-4
```

### Hydra Overrides

```bash
python src/train_v2_lightning.py \
    dataset=chexpert \
    model=convnext_base \
    train.epochs=20 \
    train.batch_size=64 \
    train.lr=5e-5
```

## Output Structure

Training outputs are saved under `outputs/{dataset_name}/{run_name}/`:

```
outputs/
└── {dataset_name}/
    └── {run_name}/
        ├── checkpoints/
        │   ├── best_model.pt      # Best model based on validation metric
        │   └── last_model.pt      # Final model checkpoint
        ├── val/
        │   └── preds_epoch*.csv   # Validation predictions
        ├── test/
        │   └── preds_epoch*.csv   # Test predictions
        ├── auroc_list.csv         # AUROC metrics per epoch
        ├── training.log           # Training log file
        └── .hydra/                # Hydra configuration backup
```

## Evaluation Metrics

After each epoch, the training script displays:

- **Mean AUROC**: Average AUROC across all labels
- **Per-label AUROC**: Individual AUROC for each pathology label
- **Subset Metrics**: Optional metrics for label subsets (e.g., top5, top12) if configured

Example output:
```
================================================================================
Epoch 1 Evaluation Results
================================================================================
Validation Mean AUROC: 0.8234
Test Mean AUROC: 0.8156

Validation Per-Label AUROC:
  Atelectasis: 0.7890
  Cardiomegaly: 0.8123
  Consolidation: 0.7654
  ...
================================================================================
```

## ASU SOL cluster

To prepare the environment on SOL cluster:

```bash
module load mamba gcc-12.1.0-gcc-11.2.0
source activate pytorch-gpu-2.3.1-cuda-12.1
pip install --user --upgrade \
    arrow pyarrow pandas lightning timm hydra-core
```

## ASU PHXRC Cluster

### To use minimal environment:

```bash
module load mamba gcc-12.3.0-vw
source activate pytorch-gpu-2.3.1-cuda-12.1
pip install --user --upgrade \
    arrow pyarrow pandas lightning timm hydra-core
```

### To create full environment:

```bash
module load mamba
mamba create -n torch-cuda-mamba python=3.13
source activate torch-cuda-mamba
mamba install torch torchvision torchmetrics transformers jupyter matplotlib scikit-image scikit-learn timm einops opencv-python pycocotools
```

## Notes for Students

- **CSV Format**: Dataset CSVs should have columns for each label with values:
  - `0`: Negative (no pathology)
  - `1`: Positive (pathology present)
  - `-1`: Uncertain/ignore (excluded from loss calculation)
- **Image Path Column**: Update `image_path_key` in dataset config if your CSV uses a different column name (default: "Path")
- **Custom Backbones**: To use other timm models, update `model.backbone` in the config with the full timm model name
- **Early Stopping**: Configured via `train.patience` (default: 10 epochs)
- **Best Model**: Saved based on `train.monitor` metric (default: `mean_auroc`)

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- PyTorch
- timm (model architectures)
- torchmetrics or sklearn (evaluation metrics)
- hydra-core (configuration management)
- pandas, numpy
