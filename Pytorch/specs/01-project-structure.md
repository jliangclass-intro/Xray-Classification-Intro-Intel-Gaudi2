# Specification: Project Structure

```
Pytorch/
  configs/
    config.yaml
    dataset/
      chestxray14.yaml
      chexpert.yaml
      mimic.yaml
      vindr_cxr.yaml
    model/
      resnet50.yaml
      convnext_base.yaml
      swin_base.yaml
    train/
      default.yaml
  src/
    train_v1_classic.py
    train_v2_lightning.py
    dataset.py
    utils.py
    evaluate.py
    analyze_output.py
    visualize_training.ipynb
  run.sh
  submit-all-slurm.sh
  requirements.txt
  specs/
```

## Rules
- `configs/config.yaml` defines Hydra defaults and run directory.
- Dataset/model/train configs are selectable by Hydra group name.
- `src/train_v1_classic.py` - Classic PyTorch training script (entrypoint)
- `src/train_v2_lightning.py` - PyTorch Lightning training script (entrypoint)
