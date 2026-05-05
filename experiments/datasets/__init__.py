"""Per-dataset loaders for tutorials and experiments.

Each dataset has its own module under this package and exposes a class
that handles **download / extract / setup automatically** on construction.
The user never has to do manual setup steps unless technically impossible
(e.g. login-gated datasets).

Modules
-------

* ``imagenette`` — fast.ai's 10-class ImageNet subset (auto-downloaded).
  Use :class:`ImagenetteDataset` or the legacy ``load_imagenette()`` /
  ``load("imagenette")`` interface.
* ``imagenet`` — full ImageNet-1k validation split. Two backends:
  - :class:`ImagenetValDataset` (disk tree, gated, manual setup pointer).
  - :class:`ImagenetValHFDataset` (un-gated HF mirror, auto-downloaded).
* ``funny_birds`` — :class:`FunnyBirdsDataset`. Auto-downloads the
  ~1.5 GB synthetic-birds zip (Hesse et al. ICCV 2023, arXiv:2308.06248).
  Provides ground-truth part maps for explainability evaluation.
* ``dsprites`` — :class:`DSpritesDataset`. Auto-downloads the dsprites
  parquet from HuggingFace (Higgins et al. 2017). Configurable
  classification target (shape / scale / orientation / x / y).

Backward-compat
---------------

The flat ``load(name, ...)`` dispatcher and ``load_imagenette`` /
``load_imagenet_val`` / ``load_imagenet_val_hf`` functions are
re-exported here, plus the legacy module-level constants
(``IMAGENETTE_TO_IMAGENET``, ``IMAGENET_SYNSETS_PATH``, etc.). Existing
code that does ``from datasets import load`` keeps working unchanged.

Adding a new dataset
--------------------

Create ``experiments/datasets/<name>.py``:

1. Define ``<NAME>_DOWNLOAD_URL`` and ``<NAME>_DATA_DIR`` constants
   (the latter as a function of a ``data_root: Path`` arg).
2. Subclass ``torch.utils.data.Dataset`` (or use a similar shape).
3. In ``__init__``, check the local cache; download + extract if missing.
4. Re-export the class from this ``__init__.py``.
5. Add a ``DATASETS["<name>"] = <name>_loader`` entry to make it
   discoverable via :func:`load`.
"""

# ── Re-export the legacy single-file API ────────────────────────────────────
from pathlib import Path as _Path
from typing import Optional as _Optional

from ._legacy import (  # noqa: F401
    CuratedDataset,
    DATASETS as _LEGACY_DATASETS,
    IMAGENETTE_TO_IMAGENET,
    IMAGENETTE_CLASS_NAMES,
    IMAGENET_SYNSETS_PATH,
    _default_data_dir,
    load_imagenette,
    load_imagenet_val,
    load_imagenet_val_hf,
)

# ── Per-dataset class API ───────────────────────────────────────────────────
from .imagenette import ImagenetteDataset  # noqa: F401
from .imagenet import ImagenetValDataset, ImagenetValHFDataset  # noqa: F401
from .funny_birds import FunnyBirdsDataset  # noqa: F401
from .dsprites import DSpritesDataset  # noqa: F401


# Bridge: register the new class-based loaders so ``load(name)`` works for
# them too. Each entry is ``(name, factory)``; the factory takes the same
# kwargs as the existing loaders.
def _funny_birds_factory(root, *, split="train", transform=None, **kw):
    return FunnyBirdsDataset(
        root=root, split=split, transform=transform, **kw,
    )


def _dsprites_factory(root, *, target="shape", transform=None, **kw):
    return DSpritesDataset(
        root=root, target=target, transform=transform, **kw,
    )


DATASETS = dict(_LEGACY_DATASETS)
DATASETS["funny_birds"] = _funny_birds_factory
DATASETS["dsprites"] = _dsprites_factory


def load(name: str, *, root: _Optional[_Path] = None, **kwargs):
    """Dispatch to one of the per-dataset loaders by name. Default ``root``
    is ``<repo>/data``.

    Available names: ``imagenette``, ``imagenet_val``, ``imagenet_val_hf``,
    ``funny_birds``, ``dsprites``.
    """
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
    "IMAGENET_SYNSETS_PATH",
    "load",
    "load_imagenette",
    "load_imagenet_val",
    "load_imagenet_val_hf",
    # Class API (one class per dataset; each module owns its own
    # auto-download / extract / setup logic).
    "ImagenetteDataset",
    "ImagenetValDataset",
    "ImagenetValHFDataset",
    "FunnyBirdsDataset",
    "DSpritesDataset",
]
