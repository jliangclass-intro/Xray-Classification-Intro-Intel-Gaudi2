# Specification: Dependencies

Required:
- torch
- torchvision
- pandas
- numpy
- pyyaml
- tqdm
- pillow
- hydra-core
- omegaconf
- torchmetrics
- timm
- psutil

Optional:
- lightning - for PyTorch Lightning training script (`train_v2_lightning.py`, training continues with warning if not installed)
- ipywidgets - for interactive Jupyter notebook widgets (required for visualize_training.ipynb)