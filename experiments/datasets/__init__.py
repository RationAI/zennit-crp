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


# Experiment eval-dataset keys → (loader name, loader kwargs). These keys name
# datasets in job specs, result parquets and figure trees — do not rename.
EVAL_DATASETS = {
    "funny_birds":   ("funny_birds",   {"split": "train", "clean_only": True}),
    "dsprites":      ("dsprites",      {"target": "shape"}),
    "colored_mnist": ("colored_mnist", {"split": "train"}),
    "imagenet":      ("imagenet_val_hf", {}),
}


def load_eval_dataset(key: str, transform, extra_kwargs: _Optional[dict] = None):
    """Un-normalized eval dataset for an :data:`EVAL_DATASETS` key.

    ``extra_kwargs`` is merged into the loader kwargs. ``ImagenetValHFDataset``
    always loads the FULL 50k val — an ``n_per_class`` / ``classes`` entry for
    imagenet is applied AFTER construction (``ds.subsample`` /
    ``filter_classes``, identical pools to the old in-constructor sampling),
    so experiment scripts keep their recorded pool protocol.
    """
    ds_name, ds_kw = EVAL_DATASETS[key]
    merged = {**ds_kw, **(extra_kwargs or {})}
    pool = classes = None
    if ds_name == "imagenet_val_hf":
        pool = merged.pop("n_per_class", None)
        classes = merged.pop("classes", None)
    ds = load(ds_name, transform=transform, **merged)
    if classes is not None:
        ds.items = ds.filter_classes(ds.items, classes)
    if pool is not None:
        ds.subsample(pool)
    return ds


__all__ = [
    "ImageClassDataset",
    "DATASETS",
    "EVAL_DATASETS",
    "load_eval_dataset",
    "IMAGENETTE_TO_IMAGENET",
    "IMAGENETTE_CLASS_NAMES",
    "load",
    "ImagenetteDataset",
    "ImagenetValHFDataset",
    "FunnyBirdsDataset",
    "DSpritesDataset",
    "ColoredMNISTDataset",
]
