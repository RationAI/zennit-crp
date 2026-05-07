"""dsprites — 2D shapes dataset for disentanglement / XAI baselines.

Reference: Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a
Constrained Variational Framework," ICLR 2017. Original dataset:
https://github.com/google-deepmind/dsprites-dataset.

737 280 grayscale 64×64 images of three shapes (square / ellipse / heart)
at 6 scales, 40 orientations, 32 × 32 grid positions. Single object per
image, white-on-black background. The original dataset ships as a single
~26 MB ``.npz`` file with 6 latent factors per image:

* ``label_color``     (1 value: white)
* ``label_shape``     (3 values: square, ellipse, heart)
* ``label_scale``     (6 values: 0.5 … 1.0)
* ``label_orientation`` (40 values: 0 … 2π)
* ``label_posX``      (32 values)
* ``label_posY``      (32 values)

For classification you pick a target factor (``shape`` is the most
common, 3 classes; ``scale`` for 6, etc.); this loader exposes that
choice via the ``target`` argument.

Why this dataset for XAI: latent factors are *known per image*, so an
explanation method can be evaluated against ground truth — does the
attribution localise the shape pixel block, or pick up a confounder?

Usage::

    from datasets.dsprites import DSpritesDataset
    ds = DSpritesDataset(root='data', target='shape')   # 3-class
    img, label = ds[0]                                   # (PIL.Image RGB, int)

Auto-downloads the upstream ``.npz`` file from the deepmind GitHub
release on first use.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Official upstream npz from the deepmind/dsprites-dataset release.
DSPRITES_URL = (
    "https://github.com/google-deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)

# Latent factor names in the order they appear in the npz arrays.
LATENT_NAMES = (
    "color", "shape", "scale", "orientation", "posX", "posY",
)
LATENT_SIZES = {
    "color": 1, "shape": 3, "scale": 6,
    "orientation": 40, "posX": 32, "posY": 32,
}


def _download(url: str, dest: Path, *, log=print) -> None:
    if dest.is_file():
        log(f"  exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  downloading {url}")
    log(f"            → {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        # dsprites .npz is small (~26MB); a single read is fine.
        f.write(resp.read())
    tmp.rename(dest)
    log(f"  download complete: {dest.stat().st_size / 1e6:.1f} MB")


@dataclass
class DSpritesDataset(Dataset):
    """Class-API loader for the dsprites dataset.

    Parameters
    ----------
    root : str | Path
        Project data root (e.g. ``"data"``). The dataset will live under
        ``<root>/dsprites/dsprites.npz`` after download.
    target : {"shape", "scale", "orientation", "posX", "posY"}
        Which latent factor to use as the classification label.
        ``"shape"`` (3 classes — square / ellipse / heart) is the most
        common XAI testbed.
    transform : callable, optional
        Applied to each PIL image at ``__getitem__``. Pass the timm /
        torchvision preprocessing pipeline for a model-ready tensor.
    image_mode : {"RGB", "L"}
        Output image mode. ``"RGB"`` (default) tiles the grayscale
        channel 3× so the image is compatible with ImageNet-pretrained
        backbones (DINOv3 etc.). ``"L"`` returns single-channel.
    upsample_to : int | None
        If set, upsample each 64×64 image to ``upsample_to × upsample_to``
        via nearest-neighbour. Useful for ImageNet-trained backbones
        which expect 224 / 256 / 384. Default None (no upsampling — let
        the user's ``transform`` handle resizing).
    auto_download : bool
        If True (default), download if missing. If False, raise.
    download_url : str
        Override the source URL (default: deepmind GitHub).
    n_per_class : int, optional
        Sample at most ``n_per_class`` images per target-label class
        (deterministic given ``seed``). Useful for fast smoke-tests.
    seed : int
        PRNG seed.
    log : callable
        Where to send progress lines.
    """

    root: Path
    target: str = "shape"
    transform: Optional[Callable] = None
    image_mode: str = "RGB"
    upsample_to: Optional[int] = None
    auto_download: bool = True
    download_url: str = DSPRITES_URL
    n_per_class: Optional[int] = None
    seed: int = 0
    log: Callable[[str], None] = field(default=print, repr=False)

    # Set in __post_init__:
    npz_path: Path = field(init=False)
    images: np.ndarray = field(init=False, repr=False)
    labels: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if self.target not in LATENT_NAMES or self.target == "color":
            raise ValueError(
                f"target must be one of {tuple(n for n in LATENT_NAMES if n != 'color')}; "
                f"got {self.target!r}"
            )
        if self.image_mode not in ("RGB", "L"):
            raise ValueError(f"image_mode must be 'RGB' or 'L'; got {self.image_mode!r}")

        cache_root = self.root / "dsprites"
        self.npz_path = cache_root / "dsprites.npz"

        if not self.npz_path.is_file():
            if not self.auto_download:
                raise FileNotFoundError(
                    f"dsprites not found at {self.npz_path} and auto_download=False. "
                    f"Download from {self.download_url} manually."
                )
            self.log(f"dsprites: setting up under {cache_root}")
            _download(self.download_url, self.npz_path, log=self.log)

        self.log(f"  loading {self.npz_path.name}")
        with np.load(self.npz_path, allow_pickle=True, encoding="latin1") as data:
            # ``imgs`` shape: (737280, 64, 64) uint8, ``latents_classes`` shape
            # (737280, 6) — column index matches LATENT_NAMES order.
            self.images = data["imgs"]
            latents = data["latents_classes"]
        target_col = LATENT_NAMES.index(self.target)
        self.labels = latents[:, target_col].astype(np.int64)

        # Optional per-class subsampling.
        if self.n_per_class is not None:
            rng = np.random.default_rng(self.seed)
            keep_indices: List[int] = []
            for c in np.unique(self.labels):
                cls_idx = np.where(self.labels == c)[0]
                rng.shuffle(cls_idx)
                keep_indices.extend(cls_idx[:self.n_per_class].tolist())
            keep_indices = sorted(keep_indices)
            self.images = self.images[keep_indices]
            self.labels = self.labels[keep_indices]

        self.log(
            f"  dsprites target='{self.target}': {len(self.images)} images, "
            f"{LATENT_SIZES[self.target]} classes"
        )

    # ── torch.utils.data.Dataset interface ──────────────────────────────────

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        # The npz stores 0/1 binary masks per pixel; convert to a 0-255 uint8
        # PIL image for transform compatibility.
        arr = (self.images[i] * 255).astype(np.uint8)  # (64, 64)
        img = Image.fromarray(arr, mode="L")
        if self.upsample_to is not None and self.upsample_to != arr.shape[0]:
            img = img.resize((self.upsample_to, self.upsample_to), Image.NEAREST)
        if self.image_mode == "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

    # ── CuratedDataset-compatible properties ────────────────────────────────

    @property
    def class_indices(self) -> list[int]:
        return sorted(set(int(c) for c in np.unique(self.labels)))

    @property
    def num_classes(self) -> int:
        return LATENT_SIZES[self.target]

    @property
    def name(self) -> str:
        return f"dsprites_{self.target}"
