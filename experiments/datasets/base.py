"""Shared base for the per-dataset loaders.

:class:`ImageClassDataset` = a list of ``(source, class_idx)`` items plus a
subclass-provided ``_decode(source) -> PIL.Image``. The base supplies the
``torch.utils.data.Dataset`` interface, the transform hook, the
``class_indices`` / ``num_classes`` properties, and the shared class-filter /
per-class-subsampling helpers. ``source`` is whatever the subclass decodes:
an image path, a JPEG bytes blob, or an index into an in-memory array.
"""
from __future__ import annotations

import random
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


class ImageClassDataset(Dataset):
    """List-of-``(source, class_idx)`` image classification dataset."""

    items: List[Tuple[object, int]]
    transform: Optional[Callable]

    def _decode(self, source) -> Image.Image:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        source, cls = self.items[i]
        img = self._decode(source)
        if self.transform is not None:
            img = self.transform(img)
        return img, cls

    @property
    def class_indices(self) -> list[int]:
        return sorted({int(c) for _, c in self.items})

    @property
    def num_classes(self) -> int:
        return len(self.class_indices)

    # ── shared item-list helpers ─────────────────────────────────────────────

    @staticmethod
    def filter_classes(items: List[Tuple[object, int]],
                       classes: Optional[Sequence[int]]) -> List[Tuple[object, int]]:
        """Restrict to the given class indices (``None`` = keep all)."""
        if classes is None:
            return items
        keep = {int(c) for c in classes}
        return [(s, c) for s, c in items if int(c) in keep]

    def subsample(self, n_per_class: Optional[int], seed: int = 0) -> "ImageClassDataset":
        """In-place per-class subsampling of an already-constructed dataset
        (same algorithm and determinism as :meth:`subsample_per_class`).
        Returns ``self`` for chaining. Experiments that need a per-class pool
        apply this AFTER construction — dataset classes serve the full data."""
        self.items = self.subsample_per_class(self.items, n_per_class, seed)
        return self

    @staticmethod
    def subsample_per_class(items: List[Tuple[object, int]],
                            n_per_class: Optional[int],
                            seed: int = 0) -> List[Tuple[object, int]]:
        """At most ``n_per_class`` items per class, deterministic given ``seed``
        (``None`` = keep all)."""
        if n_per_class is None:
            return items
        rng = random.Random(seed)
        per_class: dict[int, list] = {}
        for s, c in items:
            per_class.setdefault(int(c), []).append(s)
        sampled: List[Tuple[object, int]] = []
        for c, sources in per_class.items():
            rng.shuffle(sources)
            sampled.extend((s, c) for s in sources[:n_per_class])
        rng.shuffle(sampled)
        return sampled


def download_file(url: str, dest: Path, *, log: Callable = print) -> None:
    """Fetch ``url`` to ``dest`` unless it already exists."""
    if dest.exists():
        log(f"  exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  fetching {url}")
    urllib.request.urlretrieve(url, dest)


__all__ = ["ImageClassDataset", "download_file"]
