"""Legacy single-file dataset entry points (Imagenette + ImageNet-1k val
HF mirror). Both backends auto-download on first call.

* ``"imagenette"`` — fast.ai's 10-class ImageNet subset (~98 MB),
  classes mapped back to their ImageNet-1k indices. Ideal for quick
  dev / CI smoke tests / laptop runs.

* ``"imagenet_val_hf"`` — full ImageNet-1k validation split (50K
  images, 1000 classes, ~830 MB) via the un-gated HuggingFace mirror
  ``evanarlian/imagenet_1k_resized_256``.

Both backends return a :class:`CuratedDataset` (a
``torch.utils.data.Dataset`` subclass) yielding
``(PIL.Image, target_class)`` pairs and exposing ``items``,
``class_indices``, ``num_classes``.
"""

from __future__ import annotations

import io
import random
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


# ── shared dataset object ─────────────────────────────────────────────────────


@dataclass
class CuratedDataset(Dataset):
    """A list of ``(image_path, imagenet_class_idx)`` pairs with an optional
    transform applied at ``__getitem__`` time. Behaves as a
    ``torch.utils.data.Dataset``.

    Used downstream by :class:`crp.visualization.FeatureVisualization` and by
    the milestone sweep drivers.
    """

    name: str
    """Backend name — ``"imagenette"`` or ``"imagenet_val"``."""

    root: Path
    """Filesystem root the items are anchored under."""

    items: list[tuple[Path, int]]
    """One ``(absolute_image_path, imagenet_1k_class_idx)`` tuple per sample."""

    transform: Optional[Callable] = None
    """Applied to each PIL image before returning."""

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, cls = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, cls

    @property
    def class_indices(self) -> list[int]:
        return sorted({cls for _, cls in self.items})

    @property
    def num_classes(self) -> int:
        return len(self.class_indices)


# ── Imagenette ────────────────────────────────────────────────────────────────


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


def _download(url: str, dest: Path, *, log=print) -> None:
    if dest.exists():
        log(f"  exists: {dest}")
        return
    log(f"  fetching {url}")
    urllib.request.urlretrieve(url, dest)


def _extract(archive: Path, target_dir: Path, *, log=print) -> None:
    if target_dir.exists():
        log(f"  extracted: {target_dir}")
        return
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(target_dir.parent)


def load_imagenette(
    root: Path,
    *,
    split: str = "val",
    n_per_class: Optional[int] = None,
    classes: Optional[Sequence[int]] = None,
    transform: Optional[Callable] = None,
    seed: int = 0,
    log=print,
) -> CuratedDataset:
    """Imagenette train/val split, optionally subsampled.

    Auto-downloads + extracts the ~98 MB tarball into ``root`` on first call;
    subsequent calls are cache hits.

    Parameters
    ----------
    root : Path
        Directory under which ``imagenette2-160.tgz`` and its extracted tree
        live. Created if missing.
    split : {"train", "val"}
        Which subset to walk. ``"val"`` (~390 imgs/class) is the default and
        the canonical choice for benchmarks.
    n_per_class : int | None
        If not None, sample at most ``n_per_class`` images per class
        (deterministic given ``seed``).
    classes : Sequence[int] | None
        Restrict to these ImageNet-1k indices. ``None`` keeps all 10
        Imagenette classes.
    transform : Callable | None
        Applied to each PIL image at ``__getitem__``. Pass the timm
        preprocessing pipeline for a model-ready tensor.
    seed : int
        PRNG seed for ``n_per_class`` sampling.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "imagenette2-160.tgz"
    extracted = root / "imagenette2-160"

    _download(IMAGENETTE_URL, archive, log=log)
    _extract(archive, extracted, log=log)

    split_dir = extracted / split
    if not split_dir.is_dir():
        raise SystemExit(
            f"imagenette {split}/ not found at {split_dir}. "
            f"Re-extract from {archive} or pick split ∈ {{'train','val'}}."
        )

    rng = random.Random(seed)
    items: list[tuple[Path, int]] = []
    for wnid, cls_idx in IMAGENETTE_TO_IMAGENET.items():
        if classes is not None and cls_idx not in set(classes):
            continue
        cls_dir = split_dir / wnid
        if not cls_dir.is_dir():
            continue
        candidates = sorted(cls_dir.glob("*.JPEG"))
        if n_per_class is not None and n_per_class < len(candidates):
            candidates = rng.sample(candidates, k=n_per_class)
        items.extend((p, cls_idx) for p in candidates)

    return CuratedDataset(
        name="imagenette", root=extracted, items=items, transform=transform
    )


# ── unified entry point ──────────────────────────────────────────────────────


@dataclass
class _ParquetImageDataset(Dataset):
    """In-memory list of ``(image_bytes, imagenet_class)`` pairs from
    HuggingFace parquet files. Decodes JPEG/PNG on the fly.

    Behaves like :class:`CuratedDataset` (same ``items``/``num_classes``
    interface where applicable) but ``__getitem__`` reads from the
    in-memory bytes blob, not from disk per call. Used by
    :func:`load_imagenet_val_hf`."""

    name: str
    rows: List[Tuple[bytes, int]] = field(default_factory=list)
    transform: Optional[Callable] = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        b, cls = self.rows[i]
        img = Image.open(io.BytesIO(b)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, cls

    @property
    def class_indices(self) -> list[int]:
        return sorted({c for _, c in self.rows})

    @property
    def num_classes(self) -> int:
        return len(self.class_indices)


def load_imagenet_val_hf(
    root: Path,
    *,
    n_per_class: Optional[int] = None,
    classes: Optional[Sequence[int]] = None,
    transform: Optional[Callable] = None,
    seed: int = 0,
    log=print,
) -> _ParquetImageDataset:
    """ImageNet-1k validation split via the un-gated HuggingFace mirror
    ``evanarlian/imagenet_1k_resized_256`` (50K images at 256x256, 1000
    classes, ~830 MB).

    Auto-downloads on first call into
    ``<root>/imagenet_val_hf/cache/`` (HF Hub format). Subsequent calls
    are cache hits.

    Drop-in replacement for :func:`load_imagenet_val` (the disk-tree
    backend) that does not require manual setup. Same return contract,
    same parameters. Mapping is identity (label = ImageNet-1k class
    index 0..999).
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "load_imagenet_val_hf requires pyarrow + huggingface_hub. "
            "Install with `uv add pyarrow huggingface_hub`."
        ) from e

    cache_dir = (Path(root) / "imagenet_val_hf" / "cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"  loading ImageNet-1k val from HF mirror (cache: {cache_dir})")

    repo = "evanarlian/imagenet_1k_resized_256"
    parquet_files = [
        "data/val-00000-of-00002-b5248be478d25e41.parquet",
        "data/val-00001-of-00002-85f3d9c8fa1edb63.parquet",
    ]
    rows: List[Tuple[bytes, int]] = []
    for f in parquet_files:
        local = hf_hub_download(
            repo_id=repo, filename=f, repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        log(f"  reading {Path(local).name}")
        table = pq.read_table(local, columns=["image", "label"])
        # The 'image' column is a struct {bytes, path}; we only want bytes.
        img_bytes = table["image"].combine_chunks().field("bytes").to_pylist()
        labels = table["label"].to_pylist()
        rows.extend(zip(img_bytes, labels))

    log(f"  total: {len(rows)} samples, {len(set(c for _, c in rows))} classes")

    # Optional class restriction.
    if classes is not None:
        keep = set(int(c) for c in classes)
        rows = [(b, c) for b, c in rows if c in keep]
        log(f"  after class filter ({len(keep)} classes): {len(rows)} samples")

    # Optional per-class subsampling.
    if n_per_class is not None:
        rng = random.Random(seed)
        per_class: dict[int, list[bytes]] = {}
        for b, c in rows:
            per_class.setdefault(c, []).append(b)
        sampled: List[Tuple[bytes, int]] = []
        for c, blobs in per_class.items():
            rng.shuffle(blobs)
            sampled.extend((b, c) for b in blobs[:n_per_class])
        rng.shuffle(sampled)
        rows = sampled
        log(f"  after n_per_class={n_per_class}: {len(rows)} samples")

    return _ParquetImageDataset(name="imagenet_val_hf", rows=rows, transform=transform)


DATASETS: dict[str, Callable[..., CuratedDataset]] = {
    "imagenette":      load_imagenette,
    "imagenet_val_hf": load_imagenet_val_hf,
}


def _default_data_dir() -> Path:
    """``<repo>/data`` — walk up from this file until ``pyproject.toml`` is
    found."""
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "pyproject.toml").is_file():
            return p / "data"
        p = p.parent
    raise RuntimeError("repo root with pyproject.toml not found above this module")


def load(
    name: str,
    *,
    root: Optional[Path] = None,
    **kwargs,
) -> CuratedDataset:
    """Load a dataset by name. ``root`` defaults to ``<repo>/data``."""
    if name not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; choose from {sorted(DATASETS)}"
        )
    if root is None:
        root = _default_data_dir()
    return DATASETS[name](root=root, **kwargs)


__all__ = [
    "CuratedDataset",
    "DATASETS",
    "IMAGENETTE_TO_IMAGENET",
    "IMAGENETTE_CLASS_NAMES",
    "load",
    "load_imagenette",
    "load_imagenet_val_hf",
]
