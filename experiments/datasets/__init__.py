"""Per-dataset loaders for tutorials and experiments.

Each dataset is **self-contained in its own module** under this package and
exposes a class that handles download / extract / setup automatically on
construction.


Adding a new dataset
--------------------

Create ``experiments/datasets/<name>.py``, fully self-contained:

1. Define the download URL(s) and a ``root: Path`` arg.
2. Subclass :class:`base.ImageClassDataset` — provide ``items`` +
   ``_decode(source)``; the base supplies len/getitem/props + the shared
   class-filter and per-class-subsampling helpers.
3. In ``__init__``, check the local cache; download + extract if missing.
4. Re-export the class here and add a ``DATASETS`` entry.
"""

from pathlib import Path as _Path
from typing import Optional as _Optional

from .base import ImageClassDataset  # noqa: F401
from .imagenette import (  # noqa: F401
    ImagenetteDataset,
    IMAGENETTE_CLASS_NAMES,
    IMAGENETTE_TO_IMAGENET,
)
from .imagenet import ImagenetValHFDataset  # noqa: F401
from .funny_birds import FunnyBirdsDataset  # noqa: F401
from .dsprites import DSpritesDataset  # noqa: F401
from .colored_mnist import ColoredMNISTDataset  # noqa: F401

# Registry name → dataset class. Names are stable identifiers used in job
# specs, cache paths and result keys — do not rename them.
DATASETS = {
    "imagenette": ImagenetteDataset,
    "imagenet_val_hf": ImagenetValHFDataset,
    "funny_birds": FunnyBirdsDataset,
    "dsprites": DSpritesDataset,
    "colored_mnist": ColoredMNISTDataset,
}


def _default_data_dir() -> _Path:
    """``<repo>/data`` — walk up from this file until ``pyproject.toml``."""
    p = _Path(__file__).resolve()
    while p != p.parent:
        if (p / "pyproject.toml").is_file():
            return p / "data"
        p = p.parent
    raise RuntimeError("repo root with pyproject.toml not found above this module")


def load(name: str, *, root: _Optional[_Path] = None, **kwargs):
    """Construct one of the registered datasets by name. Default ``root``
    is ``<repo>/data``."""
    if name not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    if root is None:
        root = _default_data_dir()
    return DATASETS[name](root=root, **kwargs)


__all__ = [
    "ImageClassDataset",
    "DATASETS",
    "IMAGENETTE_TO_IMAGENET",
    "IMAGENETTE_CLASS_NAMES",
    "load",
    "ImagenetteDataset",
    "ImagenetValHFDataset",
    "FunnyBirdsDataset",
    "DSpritesDataset",
    "ColoredMNISTDataset",
]
