"""CSV-based multi-label chest X-ray dataset loader."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def make_path(root: Optional[str], p: str, append: str = "") -> str:
    """Construct a full image file path from an optional root directory and relative path.

    Args:
        root: Base directory containing the image files (or None if ``p`` is already relative/absolute).
        p: Relative or absolute image path read from the CSV column.
        append: Optional suffix (e.g., ``'.jpg'``) to append to ``p`` before resolving.

    Returns:
        Resolved file path string.
    """
    if pd.isna(p):
        return p
    s = str(p) + append
    path = Path(s)
    if path.is_absolute():
        return str(path)
    if root is None:
        return str(path)
    return str(Path(root) / path)


class CSVDataset(Dataset):
    """PyTorch Dataset for loading chest X-ray images and multi-label targets from a CSV DataFrame.

    Missing or uncertain labels (NaN or -1 in the CSV) are encoded as ``-1.0`` so that
    ``MaskedBCEWithLogitsLoss`` and ``torchmetrics`` (``ignore_index=-1``) can mask them out.

    Args:
        df: pandas DataFrame containing image paths and label columns.
        image_key: Column name in ``df`` containing the image file paths.
        label_names: List of column names in ``df`` corresponding to the target disease labels.
        image_root: Root directory prepended to relative paths in ``image_key``.
        image_append: Optional file extension/suffix appended to each path in ``image_key``.
        transform: Optional ``torchvision.transforms`` pipeline applied to each PIL image.
        validate_paths: Whether to check that a random subset of image files exists on disk at init.
        validate_samples: Number of random rows to check when ``validate_paths=True``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_key: str,
        label_names: List[str],
        image_root: Optional[str] = None,
        image_append: str = "",
        transform=None,
        validate_paths: bool = True,
        validate_samples: int = 5,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_key = image_key
        self.label_names = label_names
        self.image_root = image_root
        self.image_append = image_append
        self.transform = transform

        # Store original paths before modification (for saving predictions)
        # Store both __key__ (if exists) and the original Path value
        if "__key__" in self.df.columns:
            self.original_keys = self.df["__key__"].astype(str).tolist()
        else:
            self.original_keys = None
        # Always store the original Path value (before image_root is applied)
        self.original_paths = self.df[image_key].astype(str).tolist()

        self.df[image_key] = self.df[image_key].apply(
            lambda p: make_path(self.image_root, p, self.image_append)
        )

        if validate_paths and len(self.df) > 0:
            self._validate_paths(validate_samples)

    def _validate_paths(self, n_samples: int = 5) -> None:
        """Verify that a random sample of resolved image paths exists on disk.

        Args:
            n_samples: Number of random rows from the DataFrame to check.

        Raises:
            FileNotFoundError: If any sampled image path is NaN or does not exist on disk.
        """
        n_samples = min(n_samples, len(self.df))
        indices = np.random.choice(len(self.df), n_samples, replace=False)
        missing = []
        for idx in indices:
            path = self.df.iloc[idx][self.image_key]
            if pd.isna(path) or not Path(path).exists():
                missing.append(path)
        if missing:
            raise FileNotFoundError(f"Missing image paths. Example: {missing[0]}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str, Optional[str]]:
        """Load and transform a single image and its multi-label target vector.

        Args:
            idx: Sample index in ``[0, len(self) - 1]``.

        Returns:
            Tuple of ``(image_tensor, labels_tensor, resolved_path, original_path, original_key)``.
        """
        row = self.df.iloc[idx]
        image_path = row[self.image_key]
        if pd.isna(image_path):
            raise FileNotFoundError("Image path is NaN")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        labels = []
        for lbl in self.label_names:
            v = row.get(lbl, float("nan"))
            labels.append(-1.0 if pd.isna(v) else float(v))
        labels = torch.tensor(labels, dtype=torch.float32)

        # Get original path and key from stored values
        original_path = self.original_paths[idx]
        original_key = self.original_keys[idx] if self.original_keys is not None else None
        
        return image, labels, image_path, original_path, original_key
