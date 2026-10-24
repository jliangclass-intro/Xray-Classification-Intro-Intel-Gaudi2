import os
import socket
import time
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig

from dataset import CSVDataset
from utils import ensure_dir, get_best_accelerator, seed_everything, hpu_mark_step
from evaluate import evaluate, save_predictions, save_val_auroc

# Register custom resolver for formatting learning rate
def format_lr(lr: float) -> str:
    """Format learning rate by replacing dots with underscores."""
    return str(lr).replace(".", "_")

OmegaConf.register_new_resolver("format_lr", format_lr, replace=True)


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _state_dict_cpu(model: nn.Module) -> dict:
    """Copy model state dict to CPU. Avoids HPU/Gaudi serialization errors on torch.save."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def _pick_free_port() -> int:
    """Return a free port for DDP rendezvous to avoid EADDRINUSE when multiple jobs run."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss that ignores targets == -1.
    
    Students can modify this class to implement custom loss functions.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 so bfloat16 sum reduction over (batch_size * num_classes)
        # does not lose mantissa precision on Intel Gaudi 2 (HPU) or Ampere+ GPUs.
        logits = logits.float()
        targets = targets.float()
        mask = targets != -1
        targets_clamped = torch.where(mask, targets, torch.zeros_like(targets))
        pos_w = self.pos_weight.float() if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets_clamped,
            pos_weight=pos_w,
            reduction="none",
        )
        loss = loss * mask.float()
        denom = mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def should_log(batch_idx: int) -> bool:
    return batch_idx < 20 or (batch_idx + 1) % 10 == 0


def get_lrs(optimizer: optim.Optimizer) -> Tuple[float, float]:
    if len(optimizer.param_groups) >= 2:
        return optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
    lr = optimizer.param_groups[0]["lr"]
    return lr, lr


def build_transforms(image_size: int, normalize: str, train: bool = True, crop_strategy: str = "nocrop"):
    t = []
    
    if train:
        t.append(transforms.Resize((image_size, image_size)))
        t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.RandomRotation(10))
        t.append(transforms.ToTensor())
    else:
        if crop_strategy == "center_crop":
            t.append(transforms.Resize(int(image_size * 1.14)))
            t.append(transforms.CenterCrop(image_size))
            t.append(transforms.ToTensor())
        elif crop_strategy == "ten_crop":
            t.append(transforms.Resize(int(image_size * 1.14)))
            t.append(transforms.TenCrop(image_size))
            t.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        else: # nocrop
            t.append(transforms.Resize((image_size, image_size)))
            t.append(transforms.ToTensor())

    if normalize == "imagenet":
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not train and crop_strategy == "ten_crop":
            t.append(transforms.Lambda(lambda crops: torch.stack([normalize_transform(crop) for crop in crops])))
        else:
            t.append(normalize_transform)
    else:
        logger.warning(f"No normalization applied - {normalize}")

    return transforms.Compose(t)


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()
    model_name = backbone

    assert pretrained, "Only allow pretrained models"

    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        raise ValueError(f"Unsupported backbone: {backbone} (timm name: {model_name})") from exc


def abs_or_none(path: str):
    if path is None:
        return None
    return to_absolute_path(path)


def resolve_data_root(path_or_list) -> Optional[str]:
    """Resolve a path or list of paths to the first existing directory."""
    if path_or_list is None:
        return None
    
    # Handle list of paths
    if isinstance(path_or_list, (list, tuple)) or OmegaConf.is_list(path_or_list):
        paths = OmegaConf.to_container(path_or_list, resolve=True) if OmegaConf.is_list(path_or_list) else path_or_list
        for p in paths:
            abs_p = to_absolute_path(p)
            if os.path.exists(abs_p):
                logger.info(f"Found existing data root: {abs_p}")
                return abs_p
        
        # If none exist, return the first one (converted to absolute)
        # This preserves behavior for when the user provides a path that needs to be created or checked later
        fallback = to_absolute_path(paths[0])
        logger.warning(f"None of the provided data roots exist. Falling back to first option: {fallback}")
        return fallback
        
    # Handle single path
    return abs_or_none(path_or_list)


class XRayClassificationModule(L.LightningModule):
    """Simple Lightning module for X-ray classification.
    
    Students can easily modify:
    - loss_fn: Override configure_loss() method
    - optimizer: Override configure_optimizers() method
    - training_step: Override for custom training logic
    - validation_step: Override for custom validation logic
    """
    
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        label_names: List[str],
        label_subsets: Optional[Dict[str, List[int]]],
        optimizer: str = "adamw",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        pretrained: bool = True,
        output_dir: str = "outputs",
        save_predictions: bool = True,
        dataset_name: str = "dataset",
        epochs: int = 10,
        scheduler_cfg: Optional[DictConfig] = None,
        monitor: str = "mean_auroc",
        strategies: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["label_subsets", "scheduler_cfg"])
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.label_names = label_names
        self.label_subsets = label_subsets
        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.pretrained = pretrained
        self.output_dir = output_dir
        self.save_predictions = save_predictions
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.scheduler_cfg = scheduler_cfg
        self.monitor = monitor
        self.strategies = strategies if strategies is not None else ["nocrop"]
        
        # Build model
        self.model = build_model(backbone, num_classes, pretrained)
        
        # Loss function - students can modify this
        self.loss_fn = self.configure_loss()
        
        # Track best metric for early stopping
        self.best_metric = -float("inf")
        self.best_epoch = 0
        
        # Tracking for batch logging
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss_meter = AverageMeter()
        self.start_time = time.time()
        self.data_time_start = None
        
    def configure_loss(self) -> nn.Module:
        """Configure loss function. Students can override this method to use custom loss."""
        return MaskedBCEWithLogitsLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step. Students can override for custom training logic."""
        # Track data loading time
        if self.data_time_start is not None:
            self.data_time.update(time.time() - self.data_time_start)
        batch_start = time.time()
        
        images, labels, _, _, _ = batch
        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        
        # Track batch processing time
        batch_size = images.size(0)
        self.batch_time.update(time.time() - batch_start)
        self.loss_meter.update(loss.item(), batch_size)
        
        # Log loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Set data_time_start for next batch
        self.data_time_start = time.time()
        
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """Ensure HPU Lazy Mode flushes the backward graph before optimizer.step()."""
        if getattr(self.device, "type", None) == "hpu":
            hpu_mark_step()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Run optimizer step and flush HPU Lazy Mode graph afterwards."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure)
        if getattr(self.device, "type", None) == "hpu":
            hpu_mark_step()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log training progress every 10th batch."""
        if should_log(batch_idx):
            current_epoch = self.current_epoch + 1
            global_step = self.global_step
            iters_per_epoch = self.trainer.num_training_batches
            optimizer = self.optimizers()
            lr_head, lr_back = get_lrs(optimizer)
            rank = 0
            
            elapsed = time.time() - self.start_time
            total_iters = max(1, iters_per_epoch * self.epochs)
            eta = (elapsed / max(1, global_step)) * (total_iters - global_step)
            
            logger.info(
                f"Rank {rank} {self.dataset_name} {self.optimizer_name} E={current_epoch}/{self.epochs} "
                f"g_step={global_step} [B {batch_idx}/{iters_per_epoch}] "
                f"BT={self.batch_time.val:.2f}({self.batch_time.avg:.2f}) "
                f"DT={self.data_time.val:.2f}({self.data_time.avg:.2f}) "
                f"LR={lr_head:.2e}/{lr_back:.2e} Loss={self.loss_meter.val:.4f}({self.loss_meter.avg:.4f}) "
                f"Time {seconds_to_hms(elapsed)}/{seconds_to_hms(eta)}"
            )
    
    def on_train_epoch_start(self):
        """Reset meters at start of each epoch."""
        self.batch_time.reset()
        self.data_time.reset()
        self.loss_meter.reset()
        self.data_time_start = time.time()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step - just pass through, evaluation happens in on_validation_epoch_end."""
        pass
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Test step - just pass through, evaluation happens in on_test_epoch_end."""
        pass
    
    def on_validation_epoch_end(self):
        """Evaluate on validation and test sets at end of validation epoch."""
        if self.trainer.sanity_checking:
            return
            
        # Dataloaders are provided as Lists
        val_loaders = []
        if self.trainer.val_dataloaders:
            if isinstance(self.trainer.val_dataloaders, (list, tuple)):
                val_loaders = list(self.trainer.val_dataloaders)
            else:
                val_loaders = [self.trainer.val_dataloaders]
                
        # To avoid saving the auroc list on every loader, we will accumulate metrics
        all_val_metrics = {}
        
        for loader_idx, val_loader in enumerate(val_loaders):
            strategy_name = self.strategies[loader_idx] if hasattr(self, "strategies") and loader_idx < len(self.strategies) else "nocrop"
            
            val_metrics_local, val_probs, val_targets, val_paths, val_keys = evaluate(
                self.model, val_loader, self.device, self.label_names, self.label_subsets
            )
            
            world_size = self.trainer.world_size if self.trainer.world_size > 0 else 1
            if world_size > 1 and torch.distributed.is_initialized():
                probs_t = torch.tensor(val_probs, device=self.device)
                targets_t = torch.tensor(val_targets, device=self.device)
                
                gathered_probs = self.all_gather(probs_t)
                gathered_targets = self.all_gather(targets_t)
                
                val_probs = gathered_probs.view(-1, gathered_probs.size(-1)).cpu().numpy()
                val_targets = gathered_targets.view(-1, gathered_targets.size(-1)).cpu().numpy()
                
                paths_list = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(paths_list, val_paths)
                val_paths = [item for sublist in paths_list for item in sublist]
                
                if val_keys is not None:
                    keys_list = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(keys_list, val_keys)
                    val_keys = [item for sublist in keys_list for item in sublist]
            
                # Remove DDP padding uniquely by path
                unique_paths = []
                unique_indices = []
                seen = set()
                for i, p in enumerate(val_paths):
                    if p not in seen:
                        seen.add(p)
                        unique_paths.append(p)
                        unique_indices.append(i)
                
                val_paths = unique_paths
                val_probs = val_probs[unique_indices]
                val_targets = val_targets[unique_indices]
                if val_keys is not None:
                    val_keys = [val_keys[i] for i in unique_indices]
                
                # Recompute global metrics on the correctly unpadded full set
                # We do this on all ranks so they can safely self.log identically
                from evaluate import calculate_multilabel_metrics, calculate_per_label_auroc, calculate_subset_metrics
                mean_auroc, mean_aupr = calculate_multilabel_metrics(val_probs, val_targets, self.device)
                val_metrics = {
                    "mean_auroc": mean_auroc,
                    "mean_aupr": mean_aupr,
                    "per_label_auroc": calculate_per_label_auroc(val_probs, val_targets, self.device),
                }
                if self.label_subsets is not None:
                    for subset_name, label_indices in self.label_subsets.items():
                        subset_auroc, subset_aupr = calculate_subset_metrics(val_probs, val_targets, label_indices, self.device)
                        val_metrics[f"auroc_{subset_name}"] = subset_auroc
                        val_metrics[f"aupr_{subset_name}"] = subset_aupr
            else:
                val_metrics = val_metrics_local
                
            suffix = f"_{strategy_name}" if strategy_name != "nocrop" else ""
            for k, v in val_metrics.items():
                if k == "mean_auroc":
                    all_val_metrics[f"mean_auroc{suffix}"] = v
                elif k.startswith("auroc_"):
                    subset = k.replace("auroc_", "")
                    all_val_metrics[f"auroc_{subset}{suffix}"] = v
                else:
                    all_val_metrics[f"{k}{suffix}"] = v
                
            # Log validation metrics globally across active shards
            for key, value in val_metrics.items():
                if isinstance(value, list):
                    if key == "per_label_auroc" and value:
                        mean_val = float(np.nanmean([x for x in value if isinstance(x, (int, float))] or [float("nan")]))
                        self.log(f"val_{key}_{strategy_name}_mean", mean_val, on_epoch=True, prog_bar=True, sync_dist=True)
                    continue
                if not (isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1)):
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.item()
                
                if strategy_name == "nocrop":
                    self.log(f"val_{key}", value, on_epoch=True, prog_bar=True, sync_dist=True)
                else:
                    self.log(f"val_{key}_{strategy_name}", value, on_epoch=True, prog_bar=True, sync_dist=True)
                
                # Dynamic aliasing: if the metric is exactly 'mean_auroc' and the user specifically requested 
                # a customized alias like 'mean_auroc_no_crop' in config, we log it so EarlyStopping can find it natively.
                if key == "mean_auroc" and self.monitor != "mean_auroc":
                    if strategy_name in self.monitor or (strategy_name == "nocrop" and "no_crop" in self.monitor):
                        self.log(f"val_{self.monitor}", value, on_epoch=True, prog_bar=False, sync_dist=True)
            
            # Save predictions manually exclusively on zero rank
            if self.save_predictions and self.trainer.is_global_zero:
                suffix = f"_{strategy_name}" if strategy_name != "nocrop" else ""
                save_predictions(
                    self.output_dir, 
                    f"val{suffix}", 
                    self.current_epoch + 1, 
                    self.global_step, 
                    val_paths, 
                    val_probs, 
                    val_targets, 
                    self.label_names,
                    keys=val_keys
                )
        
        # Always evaluate and save test set predictions when validation predictions are saved
        test_loaders = getattr(self, "test_loaders", getattr(self, "test_dataloaders", []))
        if not test_loaders:
            test_loader = getattr(self, "test_loader", None)
            if test_loader is not None:
                test_loaders = [test_loader]
        
        all_test_metrics = {}
        for loader_idx, test_loader in enumerate(test_loaders):
            strategy_name = self.strategies[loader_idx] if hasattr(self, "strategies") and loader_idx < len(self.strategies) else "nocrop"
            
            test_metrics_local, test_probs, test_targets, test_paths, test_keys = evaluate(
                self.model, test_loader, self.device, self.label_names, self.label_subsets
            )
            
            world_size = self.trainer.world_size if self.trainer.world_size > 0 else 1
            if world_size > 1 and torch.distributed.is_initialized():
                probs_t = torch.tensor(test_probs, device=self.device)
                targets_t = torch.tensor(test_targets, device=self.device)
                
                gathered_probs = self.all_gather(probs_t)
                gathered_targets = self.all_gather(targets_t)
                
                test_probs = gathered_probs.view(-1, gathered_probs.size(-1)).cpu().numpy()
                test_targets = gathered_targets.view(-1, gathered_targets.size(-1)).cpu().numpy()
                
                paths_list = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(paths_list, test_paths)
                test_paths = [item for sublist in paths_list for item in sublist]
                
                if test_keys is not None:
                    keys_list = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(keys_list, test_keys)
                    test_keys = [item for sublist in keys_list for item in sublist]
            
                unique_paths = []
                unique_indices = []
                seen = set()
                for i, p in enumerate(test_paths):
                    if p not in seen:
                        seen.add(p)
                        unique_paths.append(p)
                        unique_indices.append(i)
                
                test_paths = unique_paths
                test_probs = test_probs[unique_indices]
                test_targets = test_targets[unique_indices]
                if test_keys is not None:
                    test_keys = [test_keys[i] for i in unique_indices]
                
                from evaluate import calculate_multilabel_metrics, calculate_per_label_auroc, calculate_subset_metrics
                mean_auroc, mean_aupr = calculate_multilabel_metrics(test_probs, test_targets, self.device)
                test_metrics = {
                    "mean_auroc": mean_auroc,
                    "mean_aupr": mean_aupr,
                    "per_label_auroc": calculate_per_label_auroc(test_probs, test_targets, self.device),
                }
                if self.label_subsets is not None:
                    for subset_name, label_indices in self.label_subsets.items():
                        subset_auroc, subset_aupr = calculate_subset_metrics(test_probs, test_targets, label_indices, self.device)
                        test_metrics[f"auroc_{subset_name}"] = subset_auroc
                        test_metrics[f"aupr_{subset_name}"] = subset_aupr
            else:
                test_metrics = test_metrics_local
                
            suffix = f"_{strategy_name}" if strategy_name != "nocrop" else ""
            for k, v in test_metrics.items():
                if k == "mean_auroc":
                    all_test_metrics[f"mean_auroc{suffix}"] = v
                elif k.startswith("auroc_"):
                    subset = k.replace("auroc_", "")
                    all_test_metrics[f"auroc_{subset}{suffix}"] = v
                else:
                    all_test_metrics[f"{k}{suffix}"] = v
            
            # Log test metrics
            for key, value in test_metrics.items():
                if isinstance(value, list):
                    if key == "per_label_auroc" and value:
                        mean_val = float(np.nanmean([x for x in value if isinstance(x, (int, float))] or [float("nan")]))
                        self.log(f"test_{key}_{strategy_name}_mean", mean_val, on_epoch=True, sync_dist=True)
                    continue
                if not (isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1)):
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if strategy_name == "nocrop":
                    self.log(f"test_{key}", value, on_epoch=True, sync_dist=True)
                else:
                    self.log(f"test_{key}_{strategy_name}", value, on_epoch=True, sync_dist=True)
            
            if self.save_predictions and self.trainer.is_global_zero:
                suffix = f"_{strategy_name}" if strategy_name != "nocrop" else ""
                save_predictions(
                    self.output_dir, 
                    f"test{suffix}", 
                    self.current_epoch + 1, 
                    self.global_step, 
                    test_paths, 
                    test_probs, 
                    test_targets, 
                    self.label_names,
                    keys=test_keys
                )
        
        # Save auroc_list.csv after validation exactly once
        if bool(all_val_metrics) and self.trainer.is_global_zero:
            save_val_auroc(
                self.output_dir,
                self.current_epoch + 1,
                self.global_step,
                all_val_metrics,
                test_metrics=all_test_metrics if bool(all_test_metrics) else None,
                label_subsets=self.label_subsets,
            )
    
    def on_test_epoch_end(self):
        """Evaluate on test set at end of test epoch."""
        test_loaders = getattr(self, "test_loaders", getattr(self, "test_dataloaders", []))
        if not test_loaders:
            test_loader = getattr(self, "test_loader", None)
            if test_loader is not None:
                test_loaders = [test_loader]
        
        for loader_idx, test_loader in enumerate(test_loaders):
            strategy_name = self.strategies[loader_idx] if hasattr(self, "strategies") and loader_idx < len(self.strategies) else "nocrop"
            
            test_metrics_local, test_probs, test_targets, test_paths, test_keys = evaluate(
                self.model, test_loader, self.device, self.label_names, self.label_subsets
            )
            
            world_size = self.trainer.world_size if self.trainer.world_size > 0 else 1
            if world_size > 1 and torch.distributed.is_initialized():
                probs_t = torch.tensor(test_probs, device=self.device)
                targets_t = torch.tensor(test_targets, device=self.device)
                
                gathered_probs = self.all_gather(probs_t)
                gathered_targets = self.all_gather(targets_t)
                
                test_probs = gathered_probs.view(-1, gathered_probs.size(-1)).cpu().numpy()
                test_targets = gathered_targets.view(-1, gathered_targets.size(-1)).cpu().numpy()
                
                paths_list = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(paths_list, test_paths)
                test_paths = [item for sublist in paths_list for item in sublist]
                
                if test_keys is not None:
                    keys_list = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(keys_list, test_keys)
                    test_keys = [item for sublist in keys_list for item in sublist]
            
                unique_paths = []
                unique_indices = []
                seen = set()
                for i, p in enumerate(test_paths):
                    if p not in seen:
                        seen.add(p)
                        unique_paths.append(p)
                        unique_indices.append(i)
                
                test_paths = unique_paths
                test_probs = test_probs[unique_indices]
                test_targets = test_targets[unique_indices]
                if test_keys is not None:
                    test_keys = [test_keys[i] for i in unique_indices]
                
                from evaluate import calculate_multilabel_metrics, calculate_per_label_auroc, calculate_subset_metrics
                mean_auroc, mean_aupr = calculate_multilabel_metrics(test_probs, test_targets, self.device)
                test_metrics = {
                    "mean_auroc": mean_auroc,
                    "mean_aupr": mean_aupr,
                    "per_label_auroc": calculate_per_label_auroc(test_probs, test_targets, self.device),
                }
                if self.label_subsets is not None:
                    for subset_name, label_indices in self.label_subsets.items():
                        subset_auroc, subset_aupr = calculate_subset_metrics(test_probs, test_targets, label_indices, self.device)
                        test_metrics[f"auroc_{subset_name}"] = subset_auroc
                        test_metrics[f"aupr_{subset_name}"] = subset_aupr
            else:
                test_metrics = test_metrics_local
            
            # Log test metrics
            for key, value in test_metrics.items():
                if isinstance(value, list):
                    if key == "per_label_auroc" and value:
                        mean_val = float(np.nanmean([x for x in value if isinstance(x, (int, float))] or [float("nan")]))
                        self.log(f"test_{key}_{strategy_name}_mean", mean_val, on_epoch=True, sync_dist=True)
                    continue
                if not (isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1)):
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if strategy_name == "nocrop":
                    self.log(f"test_{key}", value, on_epoch=True, sync_dist=True)
                else:
                    self.log(f"test_{key}_{strategy_name}", value, on_epoch=True, sync_dist=True)
            
            if self.save_predictions and self.trainer.is_global_zero:
                suffix = f"_{strategy_name}" if strategy_name != "nocrop" else ""
                save_predictions(
                    self.output_dir, 
                    f"test{suffix}", 
                    self.current_epoch + 1, 
                    self.global_step, 
                    test_paths, 
                    test_probs, 
                    test_targets, 
                    self.label_names,
                    keys=test_keys
                )
    
    def configure_optimizers(self):
        """Configure optimizer. Students can override this method to use custom optimizer."""
        if self.optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        if self.scheduler_cfg is None or self.scheduler_cfg.get("name") == "none":
            return optimizer
        
        scheduler_name = self.scheduler_cfg.name.lower()
        if scheduler_name == "reduce_lr_on_plateau":
            # For ReduceLROnPlateau, we need to return a dict with monitor
            monitor_metric = "mean_auroc"
            if self.scheduler_cfg.get("monitor"):
                monitor_metric = self.scheduler_cfg.monitor

            # version 1            
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer,
            #     mode=self.scheduler_cfg.get("mode", "max"),
            #     factor=self.scheduler_cfg.get("factor", 0.1),
            #     patience=self.scheduler_cfg.get("patience", 10),
            #     threshold=self.scheduler_cfg.get("threshold", 0.0001),
            #     min_lr=float(self.scheduler_cfg.get("min_lr", 1e-6)),
            # )

            # version 2: 
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_cfg.get("mode", "min"),
                factor=self.scheduler_cfg.get("factor", 0.2),
                patience=self.scheduler_cfg.get("patience", 2),
                threshold=self.scheduler_cfg.get("threshold", 0.0001),
                cooldown=self.scheduler_cfg.get("cooldown", 1),
                min_lr=float(self.scheduler_cfg.get("min_lr", 1e-7)),
                verbose=self.scheduler_cfg.get("verbose", True)
            )
            
            warmup_epochs = self.scheduler_cfg.get("warmup_epochs", 0)
            if warmup_epochs > 0:
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_epochs
                )
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_epochs]
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": f"val_{monitor_metric}",
                },
            }
        elif scheduler_name == "cosine_annealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_cfg.get("T_max", self.epochs),
                eta_min=float(self.scheduler_cfg.get("eta_min", 1e-5)),
            )
            return [optimizer], [scheduler]
        else:
            logger.warning(f"Unsupported scheduler: {scheduler_name}, using none")
            return optimizer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    train_cfg = cfg.train

    # Get dataset name
    dataset_name = dataset_cfg.get("name", "dataset")
    
    seed_everything(int(train_cfg.get("seed", 42)))

    label_names = dataset_cfg["labels"]
    num_classes = len(label_names)
    cfg_num_classes = dataset_cfg.get("num_classes")
    if cfg_num_classes is not None and int(cfg_num_classes) != num_classes:
        raise ValueError(
            f"Config num_classes ({cfg_num_classes}) != len(labels) ({num_classes})."
        )
    
    # Extract test_labels subsets if defined and convert label names to indices
    label_subsets = None
    if "test_labels" in dataset_cfg and dataset_cfg.test_labels is not None:
        label_subsets = {}
        test_labels_cfg = dataset_cfg.test_labels
        # Create a mapping from label name to index
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        
        # Process each subset (top5, top12, top6, top14, etc.)
        for subset_name, subset_labels in test_labels_cfg.items():
            if isinstance(subset_labels, list):
                # Convert label names to indices
                subset_indices = []
                for label_name in subset_labels:
                    if label_name in label_to_idx:
                        subset_indices.append(label_to_idx[label_name])
                    else:
                        logger.warning(f"Label '{label_name}' in {subset_name} not found in label_names, skipping")
                if subset_indices:
                    label_subsets[subset_name] = subset_indices
                    logger.info(f"Found {subset_name} with {len(subset_indices)} labels: {subset_labels}")

    # Get optimizer, lr, and backbone for later use
    opt_type = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg.get("lr", 1e-4))
    backbone = model_cfg.get("backbone", "model")
    
    # Use the same output directory Hydra created (where .hydra lives) so we have one folder, not two
    # output_subdir is ${run_name}/.hydra, so the run dir is runtime.output_dir/run_name
    run_name = cfg.get("run_name", "run_0")
    env_run_id = os.environ.get("RUN_ID")
    if env_run_id:
        run_name = env_run_id
        cfg.run_name = run_name
    try:
        hydra_cfg = HydraConfig.get()
        run_root = hydra_cfg.runtime.output_dir
        output_dir = os.path.join(run_root, run_name)
    except Exception:
        output_dir = run_name
        run_root = os.getcwd()
    # Hydra already created run_name when it wrote .hydra; ensure it exists for non-rank0
    rank = int(os.environ.get("RANK", "0"))
    is_rank0 = rank == 0
    if is_rank0:
        ensure_dir(output_dir)
    else:
        for _ in range(30):
            if os.path.isdir(output_dir):
                break
            time.sleep(0.2)
        if not os.path.isdir(output_dir):
            ensure_dir(output_dir)
    output_dir_abs = os.path.abspath(output_dir)
    logger.info(f"Process PID={os.getpid()} rank={rank} output_dir={output_dir} (absolute: {output_dir_abs})")

    # Create checkpoints subdirectory (rank 0 only for creation)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if is_rank0:
        ensure_dir(checkpoints_dir)

    # Setup file logging to output directory (rank 0 only so one training.log)
    if is_rank0:
        log_file = os.path.join(output_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

    # Load datasets
    train_csv = to_absolute_path(dataset_cfg.train_ann)
    val_csv = to_absolute_path(dataset_cfg.val_ann)
    test_csv = to_absolute_path(dataset_cfg.test_ann)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_tf = build_transforms(model_cfg["image_size"], model_cfg.get("normalize", "imagenet"), train=True)
    
    train_ds = CSVDataset(
        train_df,
        image_key=dataset_cfg.get("image_path_key", "Path"),
        label_names=label_names,
        image_root=resolve_data_root(dataset_cfg.get("train_image_root")),
        image_append=dataset_cfg.get("image_path_append", ""),
        transform=train_tf,
    )
    
    device_type = get_best_accelerator()
    use_pin_memory = device_type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=use_pin_memory,
    )
    
    # Load evaluation strategies
    val_cfg = cfg.get("validation", {})
    strategies = val_cfg.get("strategies", ["nocrop"])
    if not isinstance(strategies, (list, tuple)):
        from omegaconf.listconfig import ListConfig
        if isinstance(strategies, ListConfig):
            strategies = list(strategies)
        else:
            strategies = [strategies]
            
    val_loaders = []
    test_loaders = []
    
    for strategy in strategies:
        eval_tf = build_transforms(model_cfg["image_size"], model_cfg.get("normalize", "imagenet"), train=False, crop_strategy=strategy)
        
        val_ds = CSVDataset(
            val_df,
            image_key=dataset_cfg.get("image_path_key", "Path"),
            label_names=label_names,
            image_root=resolve_data_root(dataset_cfg.get("val_image_root")),
            image_append=dataset_cfg.get("image_path_append", ""),
            transform=eval_tf,
        )
        test_ds = CSVDataset(
            test_df,
            image_key=dataset_cfg.get("image_path_key", "Path"),
            label_names=label_names,
            image_root=resolve_data_root(dataset_cfg.get("test_image_root")),
            image_append=dataset_cfg.get("image_path_append", ""),
            transform=eval_tf,
        )
        
        val_loaders.append(torch.utils.data.DataLoader(
            val_ds,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=use_pin_memory,
        ))
        test_loaders.append(torch.utils.data.DataLoader(
            test_ds,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=use_pin_memory,
        ))

    # Create Lightning module
    debug = bool(OmegaConf.select(cfg, "debug", default=False))
    epochs = int(train_cfg.get("epochs", 10))
    if debug:
        epochs = 1
    model = XRayClassificationModule(
        backbone=backbone,
        num_classes=num_classes,
        label_names=label_names,
        label_subsets=label_subsets,
        optimizer=opt_type,
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        momentum=float(train_cfg.get("momentum", 0.9)),
        pretrained=model_cfg.get("pretrained", True),
        output_dir=output_dir,
        save_predictions=train_cfg.get("save_predictions", True),
        dataset_name=dataset_name,
        epochs=epochs,
        scheduler_cfg=cfg.get("scheduler"),
        monitor=train_cfg.get("monitor", "mean_auroc"),
        strategies=strategies,
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback - save as .pt files directly
    monitor_name = train_cfg.get("monitor", "mean_auroc")
    monitor_mode = train_cfg.get("monitor_mode", "max")
    
    # Track best and last models manually to save as .pt
    class SaveModelCallback(L.Callback):
        def __init__(self, checkpoints_dir, monitor_name, monitor_mode):
            super().__init__()
            self.checkpoints_dir = checkpoints_dir
            self.monitor_name = monitor_name
            self.monitor_mode = monitor_mode
            self.best_metric = -float("inf") if monitor_mode == "max" else float("inf")
            self.best_model_state = None
        
        def on_validation_epoch_end(self, trainer, pl_module):
            # Get the monitored metric value
            metric_value = trainer.callback_metrics.get(f"val_{self.monitor_name}")
            if metric_value is None:
                return
            
            metric_value = float(metric_value)
            is_better = (metric_value > self.best_metric) if self.monitor_mode == "max" else (metric_value < self.best_metric)
            
            if is_better:
                self.best_metric = metric_value
                # Save best model state_dict (CPU copy avoids HPU/Gaudi save errors)
                self.best_model_state = _state_dict_cpu(pl_module.model)
                torch.save(self.best_model_state, os.path.join(self.checkpoints_dir, "best_model.pt"))
                logger.info(f"Saved best model: val_{self.monitor_name} = {metric_value:.4f}")
        
        def on_train_epoch_end(self, trainer, pl_module):
            # Save model of this epoch
            epoch = trainer.current_epoch + 1
            torch.save(_state_dict_cpu(pl_module.model), os.path.join(self.checkpoints_dir, f"epoch_{epoch:03d}.pt"))
            logger.info(f"Saved model for epoch {epoch}")
        
        # def on_train_end(self, trainer, pl_module):
        #     # Save last model state_dict
        #     torch.save(_state_dict_cpu(pl_module.model), os.path.join(self.checkpoints_dir, f"last_model_{epoch:03d}.pt"))
        #     logger.info("Saved last model")
    
    callbacks.append(SaveModelCallback(checkpoints_dir, monitor_name, monitor_mode))
    
    # Early stopping callback
    patience = int(train_cfg.get("patience", 10))
    early_stop_callback = EarlyStopping(
        monitor=f"val_{monitor_name}",
        mode=monitor_mode,
        patience=patience,
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    
    # Setup trainer
    device_type = get_best_accelerator()
    eval_every_iter = int(train_cfg.get("eval_every_iter", 1000))
    
    # Determine accelerator, devices, and strategy (Lightning: gpu=cuda, hpu=Habana, cpu=cpu)
    trainer_strategy = "ddp"
    trainer_plugins = None
    if device_type == "cuda":
        accelerator = "gpu"
        devices = "auto"
        # Print CUDA device model and memory at the beginning
        if torch.cuda.is_available():
            logger.info("=" * 80)
            logger.info("CUDA DEVICES (to be used)")
            logger.info("=" * 80)
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024**3)
                logger.info(f"  GPU {i}: {name} (total memory: {total_gb:.2f} GB)")
            logger.info("=" * 80)
    elif device_type == "hpu":
        _hpu_count = getattr(torch.hpu, "device_count", None)
        n_hpu = (_hpu_count() if callable(_hpu_count) else _hpu_count) if _hpu_count is not None else 0
        try:
            from lightning_habana.pytorch.accelerator import HPUAccelerator
            from lightning_habana.pytorch.strategies import HPUParallelStrategy, SingleHPUStrategy
            from lightning_habana.pytorch.plugins import HPUPrecisionPlugin
            accelerator = HPUAccelerator()
            devices = n_hpu if n_hpu > 0 else 1
            trainer_strategy = HPUParallelStrategy() if devices > 1 else SingleHPUStrategy()
            trainer_plugins = [HPUPrecisionPlugin(precision="bf16-mixed")]
            logger.info("Using lightning_habana HPUAccelerator and HPU strategy.")
        except ImportError:
            accelerator = "hpu"
            devices = "auto" if n_hpu else 1
            trainer_strategy = "auto"
            logger.warning(
                "lightning_habana is not installed; falling back to Lightning accelerator='hpu', strategy='auto'. "
                "For single-HPU runs without lightning_habana, MY_PYTHON_CODE=src/train_v1_classic.py is recommended."
            )
    else:
        accelerator = "cpu"
        devices = 1
        trainer_strategy = "auto"
    
    # Mixed precision: prefer bf16 when supported (Ampere+), else fp16 (AMP)
    precision = "32-true"
    use_amp = device_type != "cpu"
    
    bf16_supported = False
    if device_type == "cuda" and torch.cuda.is_available():
        bf16_supported = torch.cuda.is_bf16_supported()
        if bf16_supported:
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                logger.info(
                    f"CUDA architecture (compute capability {major}.x) "
                    "reports bfloat16 support, but it may cause engine errors "
                    "on non-Ampere+ GPUs. Falling back to float16 (AMP)."
                )
                bf16_supported = False
    elif device_type == "hpu":
        bf16_supported = True
        
    use_bf16 = use_amp and bf16_supported
    if use_bf16:
        precision = "bf16-mixed"
    elif use_amp:
        precision = "16-mixed"
    
    # Avoid EADDRINUSE: on single-node runs use a free port so each job gets a unique port.
    # When WORLD_SIZE > 1 (multi-node) the launcher sets MASTER_PORT for all ranks; do not override.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        os.environ["MASTER_PORT"] = str(_pick_free_port())
        logger.info(f"DDP: using MASTER_PORT={os.environ['MASTER_PORT']}")
    
    trainer_kw = dict(
        max_epochs=epochs,
        val_check_interval=eval_every_iter,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        accelerator=accelerator if device_type == "hpu" else "auto",
        devices=devices,
        strategy=trainer_strategy,
        enable_progress_bar=True,
        logger=False,  # Disable default Lightning logger
        enable_checkpointing=False,  # Disable default checkpointing, we use our own
    )
    if trainer_plugins is not None:
        trainer_kw["plugins"] = trainer_plugins
    else:
        trainer_kw["precision"] = precision
    if debug:
        trainer_kw["limit_train_batches"] = 1
        logger.info("Debug: limit_train_batches=1, max_epochs=1; will run 1 batch, then evaluation, then exit.")
    trainer = L.Trainer(**trainer_kw)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Process PID: {os.getpid()} (one process per SLURM job)")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Backbone: {backbone}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch Size: {train_cfg.get('batch_size', 64)}")
    logger.info(f"Optimizer: {opt_type}")
    logger.info(f"Learning Rate: {lr}")
    logger.info(f"Monitor: {monitor_name} ({monitor_mode})")
    logger.info(f"Patience: {patience}")
    logger.info(f"Precision (AMP): {precision}")
    logger.info(f"Debug (1 batch + eval + exit): {debug}")
    if device_type == "cuda" and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        logger.info(f"Accelerator: {accelerator} ({n_gpus} GPU(s))")
        if world_size > 1:
            logger.info(f"Distributed: WORLD_SIZE={world_size} (all GPUs train one model together via DDP)")
    elif device_type == "hpu":
        _dc = getattr(torch.hpu, "device_count", None)
        n_hpu = (_dc() if callable(_dc) else _dc) if _dc is not None else 0
        logger.info(f"Accelerator: {accelerator} ({n_hpu} HPU(s))")
    else:
        logger.info(f"Accelerator: {accelerator}")
    logger.info("=" * 80)
    
    # Store test loader for access in callbacks
    model.test_loaders = test_loaders
    
    # Train
    trainer.fit(model, train_loader, val_dataloaders=val_loaders)
    
    if debug:
        logger.info("Debug: evaluation done, exiting.")
        return
    
    # Test on best model - load from .pt file
    best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        # Load to CPU first, then Lightning will move to correct device
        best_state_dict = torch.load(best_model_path, map_location="cpu")
        model.model.load_state_dict(best_state_dict)
    else:
        logger.warning(f"Best model not found at {best_model_path}, using current model state")
    
    trainer.test(model, test_loaders)


if __name__ == "__main__":
    main()
