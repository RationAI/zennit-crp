"""Per-dataset loaders for tutorials and experiments.

Each dataset has its own module under this package and exposes a class
that handles **download / extract / setup automatically** on construction.
The user never has to do manual setup steps.

Modules
-------

* ``imagenette`` — fast.ai's 10-class ImageNet subset (~98 MB).
* ``imagenet`` — :class:`ImagenetValHFDataset`. Un-gated HF mirror
  ``evanarlian/imagenet_1k_resized_256``, auto-downloaded (~830 MB).
* ``funny_birds`` — :class:`FunnyBirdsDataset`. Auto-downloads the
  ~1.5 GB synthetic-birds zip (Hesse et al. ICCV 2023).
* ``dsprites`` — :class:`DSpritesDataset`. Auto-downloads ~26 MB
  parquet (Higgins et al. 2017).

The flat ``load(name, ...)`` dispatcher and ``load_imagenette`` /
``load_imagenet_val_hf`` functions are re-exported here.

Adding a new dataset
--------------------

Create ``experiments/datasets/<name>.py``:

1. Define ``<NAME>_DOWNLOAD_URL`` and a ``data_root: Path`` arg.
2. Subclass ``torch.utils.data.Dataset`` (or use a similar shape).
3. In ``__init__``, check the local cache; download + extract if missing.
4. Re-export the class from this ``__init__.py``.
5. Add a ``DATASETS["<name>"] = <name>_loader`` entry to make it
   discoverable via :func:`load`.
"""

from pathlib import Path as _Path
from typing import Optional as _Optional

from ._legacy import (  # noqa: F401
    CuratedDataset,
    DATASETS as _LEGACY_DATASETS,
    IMAGENETTE_TO_IMAGENET,
    IMAGENETTE_CLASS_NAMES,
    _default_data_dir,
    load_imagenette,
    load_imagenet_val_hf,
)

from .imagenette import ImagenetteDataset  # noqa: F401
from .imagenet import ImagenetValHFDataset  # noqa: F401
from .funny_birds import FunnyBirdsDataset  # noqa: F401
from .dsprites import DSpritesDataset  # noqa: F401


# Bridge: register the new class-based loaders so ``load(name)`` works for
# them too. Each entry is ``(name, factory)``.
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

    Available names: ``imagenette``, ``imagenet_val_hf``, ``funny_birds``,
    ``dsprites``.
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
    "load",
    "load_imagenette",
    "load_imagenet_val_hf",
    "ImagenetteDataset",
    "ImagenetValHFDataset",
    "FunnyBirdsDataset",
    "DSpritesDataset",
]
