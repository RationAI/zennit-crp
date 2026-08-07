"""Imagenette — 10-class ImageNet subset for fast smoke tests, self-contained.

fast.ai's compact ImageNet variant: 10 visually-distinct classes (tench,
English springer, cassette player, chainsaw, church, French horn, garbage
truck, gas pump, golf ball, parachute), ~98 MB total. Classes are mapped
back to their original ImageNet-1k indices so a probe head trained on
Imagenette can in-principle co-exist with ImageNet-1k logits.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image

from .base import ImageClassDataset, download_file

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

# Imagenette WordNet IDs → ImageNet-1k class indices.
IMAGENETTE_TO_IMAGENET: dict[str, int] = {
    "n01440764":   0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

IMAGENETTE_CLASS_NAMES: dict[int, str] = {
    0: "tench", 217: "English springer", 482: "cassette player",
    491: "chain saw", 497: "church", 566: "French horn",
    569: "garbage truck", 571: "gas pump", 574: "golf ball",
    701: "parachute",
}


class ImagenetteDataset(ImageClassDataset):
    """Imagenette train/val split, optionally subsampled.

    Auto-downloads + extracts the ~98 MB tarball into ``root`` on first
    construction; subsequent constructions are cache hits.

    Parameters
    ----------
    root : Path
        Directory under which the tarball and its extracted tree live.
    split : {"train", "val"}
        ``"val"`` (~390 imgs/class) is the canonical benchmark choice.
    n_per_class : int | None
        Sample at most this many images per class (deterministic given ``seed``).
    classes : Sequence[int] | None
        Restrict to these ImageNet-1k indices (of the 10 available).
    transform : Callable | None
        Applied to each PIL image at ``__getitem__``.
    """

    name = "imagenette"

    def __init__(
        self,
        root: Path,
        *,
        split: str = "val",
        n_per_class: Optional[int] = None,
        classes: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        seed: int = 0,
        log: Callable = print,
    ):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "imagenette2-160.tgz"
        extracted = root / "imagenette2-160"

        download_file(IMAGENETTE_URL, archive, log=log)
        if not extracted.exists():
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(extracted.parent)

        split_dir = extracted / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"imagenette {split}/ not found at {split_dir}. "
                f"Re-extract from {archive} or pick split ∈ {{'train','val'}}.")

        items = [(p, cls_idx)
                 for wnid, cls_idx in IMAGENETTE_TO_IMAGENET.items()
                 for p in sorted((split_dir / wnid).glob("*.JPEG"))]
        items = self.filter_classes(items, classes)
        items = self.subsample_per_class(items, n_per_class, seed)

        self.root = extracted
        self.items = items
        self.transform = transform

    def _decode(self, source: Path) -> Image.Image:
        return Image.open(source).convert("RGB")


__all__ = [
    "ImagenetteDataset",
    "IMAGENETTE_URL",
    "IMAGENETTE_TO_IMAGENET",
    "IMAGENETTE_CLASS_NAMES",
]
