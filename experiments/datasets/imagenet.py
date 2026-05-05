"""ImageNet-1k validation split — two backends.

* :class:`ImagenetValDataset` — disk-tree backend. Expects
  ``<root>/imagenet_val/<wnid>/<image>.JPEG`` populated manually
  (image-net.org is gated; see :func:`load_imagenet_val` docstring
  for setup options).
* :class:`ImagenetValHFDataset` — un-gated HuggingFace mirror
  ``evanarlian/imagenet_1k_resized_256``. Auto-downloads ~830 MB of
  parquet files on first call.

Both expose the same return contract — ``(image, class_idx)`` pairs
keyed by ImageNet-1k class index 0..999.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from ._legacy import (
    CuratedDataset,
    IMAGENET_SYNSETS_PATH,
    _ParquetImageDataset,
    load_imagenet_val,
    load_imagenet_val_hf,
)


class ImagenetValDataset(CuratedDataset):
    """Disk-tree backend (gated source — manual setup required).

    See :func:`load_imagenet_val` for download instructions.
    """

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
        return load_imagenet_val(
            root=root, n_per_class=n_per_class, classes=classes,
            transform=transform, seed=seed, log=log,
        )


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
    "ImagenetValDataset",
    "ImagenetValHFDataset",
    "load_imagenet_val",
    "load_imagenet_val_hf",
    "IMAGENET_SYNSETS_PATH",
]
