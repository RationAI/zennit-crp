"""ImageNet-1k validation split — un-gated HuggingFace mirror, self-contained.

:class:`ImagenetValHFDataset` auto-downloads the ``evanarlian/
imagenet_1k_resized_256`` parquet files (~830 MB, 50K images at 256×256,
1000 classes) on first construction and holds the JPEG bytes in memory.
Yields ``(image, class_idx)`` pairs keyed by ImageNet-1k class index
0..999 (the mirror's labels are the identity mapping).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image

from .base import ImageClassDataset

HF_REPO = "evanarlian/imagenet_1k_resized_256"
HF_PARQUET_FILES = (
    "data/val-00000-of-00002-b5248be478d25e41.parquet",
    "data/val-00001-of-00002-85f3d9c8fa1edb63.parquet",
)


class ImagenetValHFDataset(ImageClassDataset):
    """ImageNet-1k val via the un-gated HuggingFace mirror.

    Auto-downloads into ``<root>/imagenet_val_hf/cache/`` (HF Hub layout) on
    first call; subsequent constructions are cache hits. Images are decoded
    from the in-memory parquet bytes at ``__getitem__`` time.

    Always serves the FULL 50K validation split. Experiments that need a
    per-class pool subsample after construction
    (``ds.subsample(n_per_class)`` / ``ImageClassDataset.filter_classes``).

    Parameters
    ----------
    root : Path
        Data directory; the HF cache lives under ``<root>/imagenet_val_hf/``.
    transform : Callable | None
        Applied to each PIL image at ``__getitem__``.
    """

    name = "imagenet_val_hf"

    def __init__(
        self,
        root: Path,
        *,
        transform: Optional[Callable] = None,
        log: Callable = print,
    ):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        cache_dir = (Path(root) / "imagenet_val_hf" / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        log(f"  loading ImageNet-1k val from HF mirror (cache: {cache_dir})")

        items: List[Tuple[bytes, int]] = []
        for f in HF_PARQUET_FILES:
            local = hf_hub_download(
                repo_id=HF_REPO, filename=f, repo_type="dataset",
                cache_dir=str(cache_dir),
            )
            log(f"  reading {Path(local).name}")
            table = pq.read_table(local, columns=["image", "label"])
            # The 'image' column is a struct {bytes, path}; we only want bytes.
            img_bytes = table["image"].combine_chunks().field("bytes").to_pylist()
            labels = table["label"].to_pylist()
            items.extend(zip(img_bytes, labels))
        log(f"  total: {len(items)} samples, {len({c for _, c in items})} classes")

        self.items = items
        self.transform = transform

    def _decode(self, source: bytes) -> Image.Image:
        return Image.open(io.BytesIO(source)).convert("RGB")


__all__ = ["ImagenetValHFDataset", "HF_REPO", "HF_PARQUET_FILES"]
