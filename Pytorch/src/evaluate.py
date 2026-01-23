"""Evaluation functions for model assessment."""
import os
import logging
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import ensure_dir

# Try to import torchmetrics, fallback to sklearn if not available
try:
    from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if not TORCHMETRICS_AVAILABLE and not SKLEARN_AVAILABLE:
    logging.warning("Neither torchmetrics nor sklearn available. Metrics calculation will fail.")

logger = logging.getLogger(__name__)


def _calculate_metrics_torchmetrics(
    probs: np.ndarray, targets: np.ndarray, device: Union[str, torch.device]
) -> Tuple[float, float]:
    """
    Calculate multilabel metrics using torchmetrics.
    
    Args:
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes), -1 for ignored labels
        device: Device to run metrics on
        
    Returns:
        Tuple of (mean_auroc, mean_aupr)
    """
    num_classes = probs.shape[1]
    
    # Convert to torch tensors
    probs_tensor = torch.from_numpy(probs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)
    
    # Initialize metrics
    auroc_metric = MultilabelAUROC(
        num_labels=num_classes,
        average="macro",
        ignore_index=-1,
    ).to(device)
    aupr_metric = MultilabelAveragePrecision(
        num_labels=num_classes,
        average="macro",
        ignore_index=-1,
    ).to(device)
    
    # Update metrics
    auroc_metric.update(probs_tensor, targets_tensor.long())
    aupr_metric.update(probs_tensor, targets_tensor.long())
    
    # Compute metrics
    mean_auroc = float(auroc_metric.compute().cpu().item())
    mean_aupr = float(aupr_metric.compute().cpu().item())
    
    return mean_auroc, mean_aupr


def _calculate_metrics_sklearn(
    probs: np.ndarray, targets: np.ndarray, device: Union[str, torch.device]
) -> Tuple[float, float]:
    """
    Calculate multilabel metrics using sklearn (fallback).
    
    Args:
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes), -1 for ignored labels
        device: Device parameter (ignored for sklearn, kept for API consistency)
        
    Returns:
        Tuple of (mean_auroc, mean_aupr)
    """
    num_classes = probs.shape[1]
    aurocs = []
    auprs = []
    
    for c in range(num_classes):
        # Mask out ignored labels (-1)
        mask = targets[:, c] != -1
        if mask.sum() == 0:
            continue
        
        y_true = targets[mask, c]
        y_score = probs[mask, c]
        
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            continue
        
        try:
            auroc = roc_auc_score(y_true, y_score)
            aurocs.append(auroc)
        except (ValueError, Exception):
            pass
        
        try:
            aupr = average_precision_score(y_true, y_score)
            auprs.append(aupr)
        except (ValueError, Exception):
            pass
    
    mean_auroc = float(np.nanmean(aurocs)) if aurocs else float("nan")
    mean_aupr = float(np.nanmean(auprs)) if auprs else float("nan")
    
    return mean_auroc, mean_aupr


def calculate_multilabel_metrics(
    probs: np.ndarray, targets: np.ndarray, device: Union[str, torch.device] = "cpu"
) -> Tuple[float, float]:
    """
    Calculate multilabel metrics with automatic fallback.
    
    Uses torchmetrics if available, otherwise falls back to sklearn.
    
    Args:
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes), -1 for ignored labels
        device: Device to run metrics on (for torchmetrics)
        
    Returns:
        Tuple of (mean_auroc, mean_aupr)
    """
    if TORCHMETRICS_AVAILABLE:
        return _calculate_metrics_torchmetrics(probs, targets, device)
    elif SKLEARN_AVAILABLE:
        logger.debug("Using sklearn fallback for metrics calculation")
        return _calculate_metrics_sklearn(probs, targets, device)
    else:
        raise ImportError("Neither torchmetrics nor sklearn is available for metrics calculation.")


def calculate_per_label_auroc(
    probs: np.ndarray, targets: np.ndarray, device: Union[str, torch.device] = "cpu"
) -> List[float]:
    """
    Calculate AUROC for each individual label.
    
    Args:
        probs: Probability predictions, shape (N, num_classes)
        targets: Ground truth labels, shape (N, num_classes), -1 for ignored labels
        device: Device to run metrics on (for torchmetrics)
        
    Returns:
        List of AUROC values for each label (nan if cannot be calculated)
    """
    num_classes = probs.shape[1]
    per_label_aurocs = []
    
    if TORCHMETRICS_AVAILABLE:
        # Use torchmetrics for per-label calculation
        probs_tensor = torch.from_numpy(probs).to(device)
        targets_tensor = torch.from_numpy(targets).to(device)
        
        # Use MultilabelAUROC with average="none" to get per-label metrics
        auroc_metric = MultilabelAUROC(
            num_labels=num_classes,
            average="none",  # Return per-label, not averaged
            ignore_index=-1,
        ).to(device)
        
        try:
            auroc_metric.update(probs_tensor, targets_tensor.long())
            auroc_per_label = auroc_metric.compute().cpu().numpy()
            per_label_aurocs = [float(auroc) if not np.isnan(auroc) else float("nan") for auroc in auroc_per_label]
        except (ValueError, Exception) as e:
            # Fallback to sklearn if torchmetrics fails
            logger.debug(f"torchmetrics per-label calculation failed: {e}, falling back to sklearn")
            for c in range(num_classes):
                mask = targets[:, c] != -1
                if mask.sum() == 0:
                    per_label_aurocs.append(float("nan"))
                    continue
                
                y_true = targets[mask, c]
                y_score = probs[mask, c]
                
                if len(np.unique(y_true)) < 2:
                    per_label_aurocs.append(float("nan"))
                    continue
                
                try:
                    auroc = roc_auc_score(y_true, y_score)
                    per_label_aurocs.append(float(auroc))
                except (ValueError, Exception):
                    per_label_aurocs.append(float("nan"))
    
    elif SKLEARN_AVAILABLE:
        # Use sklearn for per-label calculation
        for c in range(num_classes):
            # Mask out ignored labels (-1)
            mask = targets[:, c] != -1
            if mask.sum() == 0:
                per_label_aurocs.append(float("nan"))
                continue
            
            y_true = targets[mask, c]
            y_score = probs[mask, c]
            
            # Skip if only one class present
            if len(np.unique(y_true)) < 2:
                per_label_aurocs.append(float("nan"))
                continue
            
            try:
                auroc = roc_auc_score(y_true, y_score)
                per_label_aurocs.append(float(auroc))
            except (ValueError, Exception):
                per_label_aurocs.append(float("nan"))
    else:
        raise ImportError("Neither torchmetrics nor sklearn is available for metrics calculation.")
    
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
    
    with torch.no_grad():
        logits = model(images)
        num_classes = logits.shape[1]
        
        # Validate that label_names matches the model output
        if len(label_names) != num_classes:
            raise ValueError(
                f"Mismatch: model outputs {num_classes} classes, but label_names has {len(label_names)} labels. "
                f"Model output shape: {logits.shape}, label_names: {label_names[:5]}..."
            )

    with torch.no_grad():
        # Process first batch
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        all_original_paths.extend(original_paths)
        if original_keys is not None:
            all_original_keys.extend(original_keys)

        # Process remaining batches
        for images, labels, _, original_paths, original_keys in loader_iter:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

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
    
    # Build row data - round all AUROC values to 4 decimal places
    row_data = {
        "epoch": epoch,
        "iter": iteration,
        "val_mean_auroc": round(val_metrics.get("mean_auroc", float("nan")), 4),
    }
    
    # Add validation subset AUROCs (if defined in label_subsets or in metrics)
    if label_subsets is not None:
        for subset_name in sorted(label_subsets.keys()):
            key = f"auroc_{subset_name}"
            val = val_metrics.get(key, float("nan"))
            row_data[f"val_auroc_{subset_name}"] = round(val, 4) if not np.isnan(val) else val
    else:
        # Extract subset metrics from val_metrics keys
        for key in sorted(val_metrics.keys()):
            if key.startswith("auroc_") and key != "mean_auroc":
                # Extract subset name (e.g., "top5" from "auroc_top5")
                subset_name = key.replace("auroc_", "")
                val = val_metrics.get(key, float("nan"))
                row_data[f"val_auroc_{subset_name}"] = round(val, 4) if not np.isnan(val) else val
    
    # Add test metrics for every epoch (not just best)
    if test_metrics is not None:
        test_val = test_metrics.get("mean_auroc", float("nan"))
        row_data["test_mean_auroc"] = round(test_val, 4) if not np.isnan(test_val) else test_val
        
        # Add test subset AUROCs
        if label_subsets is not None:
            for subset_name in sorted(label_subsets.keys()):
                key = f"auroc_{subset_name}"
                val = test_metrics.get(key, float("nan"))
                row_data[f"test_auroc_{subset_name}"] = round(val, 4) if not np.isnan(val) else val
        else:
            # Extract subset metrics from test_metrics keys
            for key in sorted(test_metrics.keys()):
                if key.startswith("auroc_") and key != "mean_auroc":
                    # Extract subset name (e.g., "top5" from "auroc_top5")
                    subset_name = key.replace("auroc_", "")
                    val = test_metrics.get(key, float("nan"))
                    row_data[f"test_auroc_{subset_name}"] = round(val, 4) if not np.isnan(val) else val
    else:
        # Add empty test columns when test_metrics not provided
        row_data["test_mean_auroc"] = float("nan")
        # Add empty test subset columns
        if label_subsets is not None:
            for subset_name in sorted(label_subsets.keys()):
                row_data[f"test_auroc_{subset_name}"] = float("nan")
        else:
            # Try to infer from val_metrics what test columns might be needed
            for key in sorted(val_metrics.keys()):
                if key.startswith("auroc_") and key != "mean_auroc":
                    # Extract subset name (e.g., "top5" from "auroc_top5")
                    subset_name = key.replace("auroc_", "")
                    row_data[f"test_auroc_{subset_name}"] = float("nan")
    
    # Create DataFrame and append to CSV
    df = pd.DataFrame([row_data])
    
    # If file exists, append; otherwise create new
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        # Combine and remove duplicates (keep last if same epoch/iter)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        # Remove duplicates based on epoch and iter, keeping the last one
        combined_df = combined_df.drop_duplicates(subset=["epoch", "iter"], keep="last")
        combined_df = combined_df.sort_values(by=["epoch", "iter"])
        combined_df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)
