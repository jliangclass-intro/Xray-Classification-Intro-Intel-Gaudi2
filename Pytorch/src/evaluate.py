"""Evaluation functions for model assessment."""
import os
import logging
import contextlib
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import ensure_dir, hpu_mark_step

from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision

logger = logging.getLogger(__name__)


def _resolve_metric_device(device: Union[str, torch.device]) -> Union[str, torch.device]:
    """Return CPU when device is HPU so torchmetrics int64/sorting/ignore_index ops do not degrade.

    Args:
        device: Requested device string or ``torch.device``.

    Returns:
        ``'cpu'`` if the requested device is an Intel Gaudi HPU, otherwise ``device`` unchanged.
    """
    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    if dev_type == "hpu":
        return "cpu"
    return device


def calculate_multilabel_metrics(
    probs: np.ndarray, targets: np.ndarray, device: Union[str, torch.device] = "cpu"
) -> Tuple[float, float]:
    """
    Calculate multilabel metrics using torchmetrics.
    
    Args:
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes), -1 for ignored labels
        device: Device to run metrics on (automatically routed to CPU on HPU)
        
    Returns:
        Tuple of (mean_auroc, mean_aupr)
    """
    num_classes = probs.shape[1]
    metric_device = _resolve_metric_device(device)
    
    # Convert to torch tensors in float32 / int64 for exact metric computation
    probs_tensor = torch.from_numpy(probs).to(metric_device, dtype=torch.float32)
    targets_tensor = torch.from_numpy(targets).to(metric_device, dtype=torch.long)
    
    # Initialize metrics
    auroc_metric = MultilabelAUROC(
        num_labels=num_classes,
        average="macro",
        ignore_index=-1,
    ).to(metric_device)
    aupr_metric = MultilabelAveragePrecision(
        num_labels=num_classes,
        average="macro",
        ignore_index=-1,
    ).to(metric_device)
    
    # Update metrics
    auroc_metric.update(probs_tensor, targets_tensor)
    aupr_metric.update(probs_tensor, targets_tensor)
    
    # Compute metrics
    mean_auroc = float(auroc_metric.compute().cpu().item())
    mean_aupr = float(aupr_metric.compute().cpu().item())
    
    return mean_auroc, mean_aupr


def calculate_per_label_auroc(
    probs: np.ndarray, targets: np.ndarray, device: Union[str, torch.device] = "cpu"
) -> List[float]:
    """
    Calculate AUROC for each individual label using torchmetrics.
    
    Args:
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes), -1 for ignored labels
        device: Device to run metrics on (automatically routed to CPU on HPU)
        
    Returns:
        List of AUROC values for each label (nan if cannot be calculated)
    """
    num_classes = probs.shape[1]
    per_label_aurocs = []
    metric_device = _resolve_metric_device(device)
    
    probs_tensor = torch.from_numpy(probs).to(metric_device, dtype=torch.float32)
    targets_tensor = torch.from_numpy(targets).to(metric_device, dtype=torch.long)
    
    auroc_metric = MultilabelAUROC(
        num_labels=num_classes,
        average="none",
        ignore_index=-1,
    ).to(metric_device)
    
    try:
        auroc_metric.update(probs_tensor, targets_tensor)
        auroc_per_label = auroc_metric.compute().cpu().numpy()
        per_label_aurocs = [float(auroc) if not np.isnan(auroc) else float("nan") for auroc in auroc_per_label]
    except Exception as e:
        logger.warning(f"Failed to calculate per-label auroc: {e}")
        per_label_aurocs = [float("nan")] * num_classes

    return per_label_aurocs


def multilabel_metrics(
    probs: np.ndarray, targets: np.ndarray, label_names: List[str], device: Union[str, torch.device] = "cpu"
) -> Dict[str, float]:
    """
    Calculate multilabel metrics (wrapper for backward compatibility).
    
    Args:
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes), -1 for ignored labels
        label_names: List of label names
        device: Device to run metrics on
        
    Returns:
        Dictionary with mean_auroc and mean_aupr
    """
    mean_auroc, mean_aupr = calculate_multilabel_metrics(probs, targets, device)
    return {
        "mean_auroc": mean_auroc,
        "mean_aupr": mean_aupr,
    }


def calculate_subset_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    label_indices: List[int],
    device: Union[str, torch.device],
) -> Tuple[float, float]:
    """
    Calculate mean AUROC and mean AUPR for a subset of labels.
    
    Uses torchmetrics if available, otherwise falls back to sklearn.
    
    Args:
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes), -1 for ignored labels
        label_indices: List of label indices to include in the subset
        device: Device to run metrics on (for torchmetrics)
        
    Returns:
        Tuple of (mean_auroc, mean_aupr)
    """
    if len(label_indices) == 0:
        return float("nan"), float("nan")
    
    # Extract subset of probabilities and targets
    subset_probs = probs[:, label_indices]
    subset_targets = targets[:, label_indices]
    
    # Use the modular metrics calculation function
    return calculate_multilabel_metrics(subset_probs, subset_targets, device)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    label_names: List[str],
    label_subsets: Optional[Dict[str, List[int]]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[str], Optional[List[str]]]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader for the dataset
        device: Device to run evaluation on
        label_names: List of label names
        label_subsets: Optional dictionary mapping subset names to label indices
        
    Returns:
        Tuple of (metrics dict, probabilities array, targets array, paths list, keys list or None)
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_original_paths = []
    all_original_keys = []

    # Get the actual number of classes from the model output
    # This ensures we use the model's output size, not the config
    loader_iter = iter(loader)
    first_batch = next(loader_iter)
    images, labels, paths, original_paths, original_keys = first_batch
    images = images.to(device)
    labels = labels.to(device)
    
    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    if dev_type == "hpu":
        eval_autocast = torch.amp.autocast(device_type="hpu", dtype=torch.bfloat16)
    else:
        eval_autocast = contextlib.nullcontext()

    with torch.no_grad(), eval_autocast:
        # Check if 5D tensor (B, n_crops, C, H, W)
        n_crops = 1
        if images.dim() == 5:
            bs, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w)
            
        logits = model(images)
        num_classes = logits.shape[1]
        
        # Validate that label_names matches the model output
        if len(label_names) != num_classes:
            raise ValueError(
                f"Mismatch: model outputs {num_classes} classes, but label_names has {len(label_names)} labels. "
                f"Model output shape: {logits.shape}, label_names: {label_names[:5]}..."
            )

    with torch.no_grad(), eval_autocast:
        # Process first batch
        probs = torch.sigmoid(logits.float())
        if n_crops > 1:
            probs = probs.view(bs, n_crops, -1).mean(dim=1)
        if dev_type == "hpu":
            hpu_mark_step()
            
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        all_original_paths.extend(original_paths)
        if original_keys is not None:
            all_original_keys.extend(original_keys)

        # Process remaining batches
        for images, labels, _, original_paths, original_keys in loader_iter:
            images = images.to(device)
            labels = labels.to(device)

            n_crops = 1
            if images.dim() == 5:
                bs, n_crops, c, h, w = images.size()
                images = images.view(-1, c, h, w)

            logits = model(images)
            probs = torch.sigmoid(logits.float())

            if n_crops > 1:
                probs = probs.view(bs, n_crops, -1).mean(dim=1)
            if dev_type == "hpu":
                hpu_mark_step()

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_original_paths.extend(original_paths)
            if original_keys is not None:
                all_original_keys.extend(original_keys)

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_classes))
    targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, num_classes))

    # Calculate metrics using modular function (with automatic fallback)
    mean_auroc, mean_aupr = calculate_multilabel_metrics(probs, targets, device)
    
    # Calculate per-label AUROC
    per_label_aurocs = calculate_per_label_auroc(probs, targets, device)
    
    metrics = {
        "mean_auroc": mean_auroc,
        "mean_aupr": mean_aupr,
        "per_label_auroc": per_label_aurocs,  # List of AUROC for each label
    }
    
    # Calculate metrics for label subsets if provided
    if label_subsets is not None:
        for subset_name, label_indices in label_subsets.items():
            subset_auroc, subset_aupr = calculate_subset_metrics(probs, targets, label_indices, device)
            metrics[f"auroc_{subset_name}"] = subset_auroc
            metrics[f"aupr_{subset_name}"] = subset_aupr
    
    # Return original_keys only if we collected any
    return metrics, probs, targets, all_original_paths, all_original_keys if all_original_keys else None


def save_predictions(
    output_dir: str,
    split: str,
    epoch: int,
    iteration: int,
    paths: List[str],
    probs: np.ndarray,
    targets: np.ndarray,
    label_names: List[str],
    keys: Optional[List[str]] = None,
) -> None:
    """
    Save predictions to CSV file (only predictions, no ground truth).
    
    Args:
        output_dir: Output directory
        split: Split name (val or test)
        epoch: Current epoch number
        iteration: Current global iteration
        paths: List of original image paths from CSV (Path column)
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes) - not saved to CSV
        label_names: List of label names
        keys: Optional list of __key__ values from CSV (if exists)
    """
    if len(paths) == 0:
        return
    # Round probabilities to 4 decimal places
    probs_rounded = np.round(probs, 4)
    data = {"Path": paths}
    # Add __key__ column if provided
    if keys is not None and len(keys) > 0:
        data["__key__"] = keys
    for i, lbl in enumerate(label_names):
        data[f"pred_{lbl}"] = probs_rounded[:, i]
    df = pd.DataFrame(data)
    # Create split-specific subdirectory (val/ or test/)
    split_dir = os.path.join(output_dir, split)
    ensure_dir(split_dir)
    out_path = os.path.join(split_dir, f"preds_epoch{epoch:03d}_iter{iteration:06d}.csv")
    df.to_csv(out_path, index=False)


def save_val_auroc(
    output_dir: str,
    epoch: int,
    iteration: int,
    val_metrics: Dict[str, float],
    test_metrics: Optional[Dict[str, float]] = None,
    label_subsets: Optional[Dict[str, List[int]]] = None,
    is_best: bool = False,
) -> None:
    """
    Save validation and test AUROC metrics to CSV file (auroc_list.csv).
    
    Records val mean auroc and corresponding test auroc for every epoch.
    
    Args:
        output_dir: Output directory
        epoch: Current epoch number
        iteration: Current global iteration
        val_metrics: Validation metrics dictionary (must contain mean_auroc and optionally auroc_<subset>)
        test_metrics: Optional test metrics dictionary
        label_subsets: Optional dictionary mapping subset names to label indices (for column ordering)
        is_best: Whether this is the best validation epoch (not used, kept for API compatibility)
    """
    csv_path = os.path.join(output_dir, "auroc_list.csv")
    
    row_data = {
        "epoch": epoch,
        "iter": iteration,
    }
    
    # Process validation metrics
    for key, val in val_metrics.items():
        if key.startswith("mean_aupr") or key.startswith("per_label_auroc"):
            continue
            
        if key == "mean_auroc":
            row_name = "val_mean_auroc"
        elif key.startswith("mean_auroc_"):
            row_name = f"val_{key}"
        elif key.startswith("auroc_"):
            subset_name = key.replace("auroc_", "")
            row_name = f"val_auroc_{subset_name}"
        else:
            row_name = f"val_{key}"
            
        row_data[row_name] = round(val, 4) if not np.isnan(val) else val
        
    # Process test metrics
    if test_metrics is not None:
        for key, val in test_metrics.items():
            if key.startswith("mean_aupr") or key.startswith("per_label_auroc"):
                continue
                
            if key == "mean_auroc":
                row_name = "test_mean_auroc"
            elif key.startswith("mean_auroc_"):
                row_name = f"test_{key}"
            elif key.startswith("auroc_"):
                subset_name = key.replace("auroc_", "")
                row_name = f"test_{subset_name}_mean_auroc"
            else:
                row_name = f"test_{key}"
                
            row_data[row_name] = round(val, 4) if not np.isnan(val) else val
            
    # Create DataFrame and append to CSV
    df = pd.DataFrame([row_data])
    
    # If file exists and has content, append; otherwise create new
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            existing_df = pd.read_csv(csv_path)
            # Combine and write directly as requested
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
        except pd.errors.EmptyDataError:
            df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)
