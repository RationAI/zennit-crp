"""FunnyBirds — synthetic vision dataset for part-based XAI evaluation.

Reference: Hesse, Schaub-Meyer, Roth. ICCV 2023, oral presentation.
"FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of
Explainable XAI Methods." arXiv:2308.06248.

50 procedurally rendered bird "species" (combinations of parametric
parts: beak, eyes, wings, tail, feet) on solid backgrounds, with
ground-truth per-part segmentation maps. Designed specifically for
evaluating attribution methods — knowing which pixels belong to which
semantic part lets you measure whether an explanation lights up the
correct parts.

Usage::

    from datasets.funny_birds import FunnyBirdsDataset
    ds = FunnyBirdsDataset(root='data', split='train', transform=tfm)
    img, cls = ds[0]                 # (PIL.Image RGB, int class_idx)

    ds_with_parts = FunnyBirdsDataset(root='data', split='test',
                                      with_part_map=True, transform=tfm)
    img, cls, part_map = ds_with_parts[0]

The class auto-downloads + extracts the dataset on first construction
(~1.5 GB zip from the official mirror at TU Darmstadt). Subsequent
constructions are cache hits. Pass ``auto_download=False`` to opt out
and require the data be present.
"""
from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# Official download from the TU Darmstadt visinf mirror. ~1.5 GB zip.
FUNNY_BIRDS_URL = (
    "https://download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip"
)

# After extraction the zip produces a top-level ``FunnyBirds/`` directory
# containing ``dataset_train.json``, ``dataset_test.json``,
# ``classes.json``, ``parts.json``, ``train/<class>/<idx>.png``,
# ``test/<class>/<idx>.png`` and matching ``*_part_map/`` directories.
DATASET_SUBDIR = "FunnyBirds"

# Reference: official ColorMap of part-segmentation pixel codes (RGB → part name).
# Lifted verbatim from https://github.com/visinf/funnybirds-framework/datasets/funny_birds.py
# so callers can decode part_map tensors back to semantic part names.
PART_COLORS_TO_NAME: dict[Tuple[int, int, int], str] = {
    (255, 255, 253): "eye01",
    (255, 255, 254): "eye02",
    (255, 255, 0):   "beak",
    (255, 0, 1):     "foot01",
    (255, 0, 2):     "foot02",
    (0, 255, 1):     "wing01",
    (0, 255, 2):     "wing02",
    (0, 0, 255):     "tail",
}
BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)


def _stream_download(url: str, dest: Path, *, log=print, chunk: int = 1 << 20) -> None:
    """Download ``url`` to ``dest`` with a progress indicator and HTTP-Range
    resume support. Idempotent (re-downloads only if the final file is
    missing); resumes from a ``.part`` file if the previous download was
    interrupted.

    For large files (e.g. FunnyBirds.zip ≈ 1.5 GB) the resume is critical
    — interrupted downloads otherwise force restarting from byte zero."""
    if dest.is_file():
        log(f"  exists: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.is_file() else 0

    headers: dict[str, str] = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        log(f"  resuming download at byte {resume_from / 1e9:.2f} GB")
    log(f"  downloading {url}")
    log(f"            → {dest}")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        # Server returns 206 Partial Content if it honoured the Range header,
        # 200 OK otherwise (means we have to start over).
        status = resp.status if hasattr(resp, "status") else resp.getcode()
        content_length = int(resp.headers.get("Content-Length", 0))
        if resume_from and status == 206:
            mode = "ab"
            total = resume_from + content_length
            downloaded = resume_from
            log(f"    server confirmed resume (206 Partial Content)")
        else:
            if resume_from:
                log(f"    server ignored Range (status {status}); restarting from 0")
            mode = "wb"
            total = content_length
            downloaded = 0
        last_pct = -1
        with open(tmp, mode) as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total > 0:
                    pct = int(100 * downloaded / total)
                    if pct != last_pct and pct % 5 == 0:
                        log(f"    {pct:3d}% ({downloaded / 1e9:.2f} / {total / 1e9:.2f} GB)")
                        last_pct = pct
    tmp.rename(dest)
    log(f"  download complete: {dest.stat().st_size / 1e9:.2f} GB")


def _extract_zip(zip_path: Path, dest_dir: Path, *, log=print) -> Path:
    """Extract ``zip_path`` into ``dest_dir``; return the extracted top
    directory. Idempotent: skips if ``dest_dir/<DATASET_SUBDIR>`` already
    exists (the zip's top-level directory)."""
    extracted_root = dest_dir / DATASET_SUBDIR
    if extracted_root.is_dir() and (extracted_root / "classes.json").is_file():
        log(f"  extracted: {extracted_root}")
        return extracted_root
    log(f"  extracting {zip_path} → {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    log(f"  extraction complete: {extracted_root}")
    return extracted_root


@dataclass
class FunnyBirdsDataset(Dataset):
    """Class-API loader for the FunnyBirds dataset.

    Parameters
    ----------
    root : str | Path
        Project data root (e.g. ``"data"``). The dataset will live under
        ``<root>/funny_birds/FunnyBirds/`` after auto-extraction.
    split : {"train", "test"}
        Which split to load. The official dataset ships both.
    transform : callable, optional
        Applied to each PIL image at ``__getitem__``. Pass the timm /
        torchvision preprocessing pipeline for a model-ready tensor.
    with_part_map : bool
        If True, ``__getitem__`` returns a 3-tuple ``(image, class_idx,
        part_map)`` where ``part_map`` is a ``(3, H, W)`` uint8 tensor
        with pixel codes per :data:`PART_COLORS_TO_NAME`. Default False
        — yields the standard ``(image, class_idx)`` pair so the
        dataset is a drop-in for ``CuratedDataset`` consumers.
    auto_download : bool
        If True (default), download + extract if missing. If False,
        raise if the dataset isn't already on disk under the expected
        path.
    download_url : str
        Override the source URL (default: official TU Darmstadt mirror).
    n_per_class : int, optional
        If not None, sample at most ``n_per_class`` images per class.
        Deterministic given ``seed``.
    seed : int
        PRNG seed for ``n_per_class`` sampling.
    log : callable
        Where to send progress lines. Default ``print``.
    """

    root: Path
    split: str = "train"
    transform: Optional[Callable] = None
    with_part_map: bool = False
    auto_download: bool = True
    download_url: str = FUNNY_BIRDS_URL
    n_per_class: Optional[int] = None
    seed: int = 0
    log: Callable[[str], None] = field(default=print, repr=False)

    # Set in __post_init__:
    data_dir: Path = field(init=False)
    items: List[Tuple[Path, int]] = field(init=False, default_factory=list)
    classes: list = field(init=False, default_factory=list)
    parts: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test'; got {self.split!r}")

        # Layout under the project data root: data/funny_birds/FunnyBirds/...
        cache_root = self.root / "funny_birds"
        cache_root.mkdir(parents=True, exist_ok=True)
        zip_path = cache_root / "FunnyBirds.zip"
        self.data_dir = cache_root / DATASET_SUBDIR

        # Auto-download + extract if needed.
        manifest = self.data_dir / f"dataset_{self.split}.json"
        if not manifest.is_file():
            if not self.auto_download:
                raise FileNotFoundError(
                    f"FunnyBirds not found at {self.data_dir} and "
                    f"auto_download=False. Pass auto_download=True or "
                    f"manually extract {self.download_url} into {cache_root}."
                )
            self.log(f"FunnyBirds: setting up dataset under {cache_root}")
            _stream_download(self.download_url, zip_path, log=self.log)
            _extract_zip(zip_path, cache_root, log=self.log)
            # Optionally clean up the zip to save disk (uncomment to enable).
            # zip_path.unlink(missing_ok=True)

        # Load metadata.
        with open(self.data_dir / f"dataset_{self.split}.json") as f:
            params = json.load(f)
        with open(self.data_dir / "classes.json") as f:
            self.classes = json.load(f)
        with open(self.data_dir / "parts.json") as f:
            self.parts = json.load(f)

        # Build (image_path, class_idx) item list.
        items: List[Tuple[Path, int]] = []
        for idx, p in enumerate(params):
            cls = int(p["class_idx"])
            img_rel = f"{self.split}/{cls}/{idx:06d}.png"
            items.append((self.data_dir / img_rel, cls))

        # Optional per-class subsampling.
        if self.n_per_class is not None:
            import random
            rng = random.Random(self.seed)
            per_class: dict[int, list[Path]] = {}
            for path, c in items:
                per_class.setdefault(c, []).append(path)
            sampled: List[Tuple[Path, int]] = []
            for c, paths in per_class.items():
                rng.shuffle(paths)
                sampled.extend((p, c) for p in paths[:self.n_per_class])
            items = sampled

        self.items = items
        self.log(
            f"  FunnyBirds {self.split}: {len(self.items)} images, "
            f"{len(self.classes)} classes"
        )

    # ── torch.utils.data.Dataset interface ──────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, cls = self.items[i]
        # The PNGs are RGBA; drop the alpha channel for standard 3-channel use.
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if not self.with_part_map:
            return image, cls
        # Return part map alongside.
        idx = int(path.stem)  # 6-digit filename = sample index
        pm_path = self.data_dir / f"{self.split}_part_map" / str(cls) / f"{idx:06d}.png"
        part_map = Image.open(pm_path).convert("RGB")
        # Convert to a uint8 tensor of shape (3, H, W) with pixel codes preserved.
        import numpy as np
        pm = torch.from_numpy(np.array(part_map, dtype=np.uint8)).permute(2, 0, 1)
        return image, cls, pm

    # ── CuratedDataset-compatible properties ────────────────────────────────

    @property
    def class_indices(self) -> list[int]:
        return sorted({c for _, c in self.items})

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def name(self) -> str:
        return f"funny_birds_{self.split}"
