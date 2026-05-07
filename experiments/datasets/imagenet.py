"""ImageNet-1k validation split — un-gated HuggingFace mirror.

:class:`ImagenetValHFDataset` auto-downloads ~830 MB of parquet files
from ``evanarlian/imagenet_1k_resized_256`` on first call. Returns
``(image, class_idx)`` pairs keyed by ImageNet-1k class index 0..999.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from ._legacy import _ParquetImageDataset, load_imagenet_val_hf


class ImagenetValHFDataset(_ParquetImageDataset):
    """HuggingFace mirror backend (un-gated, auto-downloaded)."""

    def __new__(
        cls,
        root: Path,
        *,
        n_per_class: Optional[int] = None,
        classes: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        seed: int = 0,
        log: Callable = print,
    ):
        return load_imagenet_val_hf(
            root=root, n_per_class=n_per_class, classes=classes,
            transform=transform, seed=seed, log=log,
        )


__all__ = [
    "ImagenetValHFDataset",
    "load_imagenet_val_hf",
]
