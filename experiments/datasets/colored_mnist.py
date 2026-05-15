"""ColoredMNIST — MNIST digits tinted with a per-class colour, with the
correlation between colour and digit controllable.

Reference distribution: https://ieee-dataport.org/documents/colored-mnist
(Nam et al. 2020, "Learning from Failure"). The IEEE-dataport zip is
auth-gated, so this loader **reproduces the dataset programmatically**
from torchvision MNIST instead, matching the same scheme:

* ten digits ↔ ten distinct hues, evenly spaced in HSV;
* train split: a digit gets its "own" hue with probability
  ``correlation_ratio`` (the bias-aligned majority), else a uniformly
  random hue (the bias-conflicting minority);
* test split: every digit gets a uniformly random hue, breaking the
  shortcut so a model that uses colour alone scores at chance.

A model that learns digit shape (rather than colour) keeps high accuracy
on the test split. A colour-shortcut model collapses. That gap is the
whole point of the dataset.

Usage::

    from datasets.colored_mnist import ColoredMNISTDataset
    ds = ColoredMNISTDataset(root='data', split='train',
                             correlation_ratio=0.99)
    img, label = ds[0]                 # (PIL.Image RGB 28×28, int 0-9)

Auto-downloads the underlying MNIST via torchvision on first call (~12
MB). The colourisation is computed in-process with a fixed seed so the
sample-to-colour mapping is reproducible across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


def _hsv_palette(n: int) -> np.ndarray:
    """``n`` evenly-spaced hues in HSV → RGB uint8, fully saturated."""
    import colorsys
    cols = [colorsys.hsv_to_rgb(i / n, 1.0, 1.0) for i in range(n)]
    return (np.array(cols) * 255).astype(np.uint8)


@dataclass
class ColoredMNISTDataset(Dataset):
    """Class-API loader for ColoredMNIST.

    Parameters
    ----------
    root : str | Path
        Project data root (e.g. ``"data"``). MNIST will live under
        ``<root>/colored_mnist/MNIST/`` after the torchvision download.
    split : {"train", "test"}
        ``"train"`` uses the MNIST train split with correlated colours.
        ``"test"`` uses the MNIST test split with uniformly random
        colours (correlation broken — measures shape-vs-shortcut).
    correlation_ratio : float
        For ``split='train'``, probability that a sample receives its
        digit's "own" colour. Default 0.99 matches the strongly-biased
        regime in the LfF paper. Ignored for ``split='test'``.
    transform : callable, optional
        Applied to each PIL image at ``__getitem__``. Pass the model's
        preprocessing pipeline (resize 28→224, ToTensor, etc.).
    auto_download : bool
        If True (default), download MNIST if missing; else raise.
    seed : int
        PRNG seed for the per-sample colour assignment.
    """

    root: Path
    split: str = "train"
    correlation_ratio: float = 0.99
    transform: Optional[Callable] = None
    auto_download: bool = True
    seed: int = 0
    log: Callable[[str], None] = field(default=print, repr=False)

    images: np.ndarray = field(init=False, repr=False)
    labels: np.ndarray = field(init=False, repr=False)
    colors: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test'; got {self.split!r}")
        if not 0.0 <= self.correlation_ratio <= 1.0:
            raise ValueError(
                f"correlation_ratio must be in [0, 1]; got {self.correlation_ratio}"
            )

        cache_root = self.root / "colored_mnist"
        cache_root.mkdir(parents=True, exist_ok=True)

        self.log(f"colored_mnist: setting up under {cache_root}")
        mnist = MNIST(
            root=str(cache_root),
            train=(self.split == "train"),
            download=self.auto_download,
        )
        # MNIST.data is uint8 (N, 28, 28); MNIST.targets is int64 (N,).
        self.images = mnist.data.numpy()
        self.labels = mnist.targets.numpy().astype(np.int64)

        # Deterministic per-sample colour assignment.
        palette = _hsv_palette(10)
        rng = np.random.default_rng(self.seed)
        n = len(self.labels)
        if self.split == "train":
            keep_own = rng.random(n) < self.correlation_ratio
            random_colors = rng.integers(0, 10, size=n)
            color_idx = np.where(keep_own, self.labels, random_colors)
        else:
            color_idx = rng.integers(0, 10, size=n)
        self.colors = palette[color_idx]  # (N, 3) uint8

        n_aligned = int(((color_idx == self.labels).sum()) if self.split == "train" else 0)
        self.log(
            f"  colored_mnist split='{self.split}': {n} samples, "
            f"correlation={self.correlation_ratio if self.split == 'train' else 'broken'}"
            f"{f', aligned={n_aligned}/{n}' if self.split == 'train' else ''}"
        )

    # ── torch.utils.data.Dataset interface ──────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        # Tint the grayscale digit with its assigned colour: each pixel
        # (intensity ∈ [0, 1]) is multiplied by the RGB triple. The
        # background stays black; the digit takes the colour.
        gray = self.images[i].astype(np.float32) / 255.0  # (28, 28)
        rgb = (gray[..., None] * self.colors[i]).astype(np.uint8)  # (28, 28, 3)
        img = Image.fromarray(rgb, mode="RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

    # ── CuratedDataset-compatible properties ────────────────────────────────

    @property
    def class_indices(self) -> list[int]:
        return list(range(10))

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def name(self) -> str:
        return "colored_mnist"
