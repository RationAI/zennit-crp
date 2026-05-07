"""Imagenette — 10-class ImageNet subset for fast smoke tests.

fast.ai's compact ImageNet variant: 10 visually-distinct classes
(tench, English springer, cassette player, chainsaw, church,
French horn, garbage truck, gas pump, golf ball, parachute), ~98 MB
total, classes mapped back to their original ImageNet-1k indices so
a probe head trained on Imagenette can in-principle co-exist with
ImageNet-1k logits.

Usage::

    from datasets.imagenette import ImagenetteDataset
    ds = ImagenetteDataset(root='data', split='val', transform=tfm)

The class is a thin wrapper around :func:`load_imagenette` (in
``_legacy.py``) — kept here for the per-file API contract noted in
``README.md``. Full implementation still lives in ``_legacy.py`` so we
preserve back-compat for older code that imports ``load_imagenette``
directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from ._legacy import (
    CuratedDataset,
    IMAGENETTE_CLASS_NAMES,
    IMAGENETTE_TO_IMAGENET,
    IMAGENETTE_URL,
    load_imagenette,
)


class ImagenetteDataset(CuratedDataset):
    """Class-API wrapper around :func:`load_imagenette`.

    Parameters identical to :func:`load_imagenette`. On construction it
    auto-downloads + extracts the tarball if missing.
    """

    def __new__(
        cls,
        root: Path,
        *,
        split: str = "val",
        n_per_class: Optional[int] = None,
        classes: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        seed: int = 0,
        log: Callable = print,
    ):
        # Delegate to the existing functional loader; CuratedDataset
        # is a dataclass-like, the loader returns one fully populated.
        return load_imagenette(
            root=root, split=split, n_per_class=n_per_class,
            classes=classes, transform=transform, seed=seed, log=log,
        )


__all__ = [
    "ImagenetteDataset",
    "load_imagenette",
    "IMAGENETTE_URL",
    "IMAGENETTE_TO_IMAGENET",
    "IMAGENETTE_CLASS_NAMES",
]
