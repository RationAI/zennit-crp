"""``ModelDataset`` — one object bundling a predefined model + its eval dataset.

Adapted from the LLEXICORP ``model_dataset_classes`` paradigm
(``crp-experimenting`` repo). A concrete subclass pairs one model with one
dataset; instantiating it gives a ready-to-use bundle:

* ``model``       — the frozen eval model (a zoo class instance),
* ``backbone``    — the ViT the composites/concepts resolve ``blocks.{i}`` against,
* ``dataset``     — the un-normalized eval dataset,
* ``transform`` / ``normalize`` — the timm split (display tensor vs forward-normalize),
* ``num_classes`` / ``label`` — metadata,
* ``tag``         — the ``<model>_<dataset>`` artefact key (FV cache / figure dir).

Scope is deliberately lean: model + dataset + transforms + metadata. CRP
composites, layer-name conventions and FeatureVisualization stay in the
experiments that use them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from PIL.Image import Image

from experiments.datasets import load_eval_dataset
from experiments.models import backbone_transforms


class ModelDataset(ABC):
    """Abstract base pairing one model with one dataset.

    Concrete subclasses set the class attributes ``MODEL`` / ``DATASET``
    (identifier strings from :mod:`.names_paths`) and implement
    :meth:`_setup_model`. Dataset behaviour (loading kwargs come from
    :data:`experiments.datasets.EVAL_DATASETS`; class-label lookup) is supplied
    by a dataset mixin.
    """

    #: Model identifier (``M_*`` constant). Set on the concrete class.
    MODEL: str = ""
    #: Dataset identifier / :data:`EVAL_DATASETS` key (``DS_*`` constant).
    DATASET: str = ""

    # populated during __init__
    model: torch.nn.Module
    backbone: torch.nn.Module
    transform: Callable[[Image], torch.Tensor]
    normalize: Callable[[torch.Tensor], torch.Tensor]
    num_classes: int
    label: str

    def __init__(self, device: Optional[str] = None, *,
                 ds_extra: Optional[dict] = None,
                 checkpoint: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._checkpoint = checkpoint
        self._ds_extra = dict(ds_extra or {})
        self._dataset = None

        self._setup_model()                       # sets model/backbone/num_classes/label
        self.transform, self.normalize = backbone_transforms(self.backbone)

    # ── to implement / mix in ─────────────────────────────────────────────────
    @abstractmethod
    def _setup_model(self) -> None:
        """Instantiate the frozen eval model; set ``model``, ``backbone``,
        ``num_classes`` and ``label``. Composes the matching zoo class."""
        ...

    @property
    def dataset(self):
        """Un-normalized eval dataset for :attr:`DATASET`, built lazily on first
        access (some eval datasets are heavy — a model-only consumer pays
        nothing). Loader defaults come from :data:`EVAL_DATASETS`; ``ds_extra``
        (per-experiment overrides such as ``split`` / ``n_per_class``) is merged
        on top."""
        if self._dataset is None:
            self._dataset = load_eval_dataset(self.DATASET, self.transform, self._ds_extra)
        return self._dataset

    def get_class_label(self, idx: int) -> str:
        """Human-readable label for a class index. Default is generic; dataset
        mixins override where real names exist."""
        return f"class {idx}"

    # ── convenience ───────────────────────────────────────────────────────────
    @property
    def tag(self) -> str:
        """Flat ``<model>_<dataset>`` artefact key — names the FV cache and the
        gallery figure tree. Preserves the historical directory layout."""
        return f"{self.MODEL}_{self.DATASET}"

    def _ckpt_kw(self) -> dict:
        """``{"checkpoint": ...}`` iff an explicit checkpoint override was given
        (finetuned-probe models only; off-the-shelf models ignore it)."""
        return {"checkpoint": self._checkpoint} if self._checkpoint else {}

    def transform_image(self, image: Image) -> torch.Tensor:
        """PIL → display-ready ``[0, 1]`` tensor (no normalize)."""
        return self.transform(image)

    def preprocess(self, image: Image) -> torch.Tensor:
        """PIL → normalized, batched, on-device model input ``(1, C, H, W)``."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        return self.normalize(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} tag={self.tag!r} num_classes={self.num_classes}>"


# ── Dataset mixins ────────────────────────────────────────────────────────────
# Each sets DATASET and, where real class names exist, overrides get_class_label.

class FunnyBirdsMixin:
    DATASET = "funny_birds"


class DspritesMixin:
    DATASET = "dsprites"


class ColoredMnistMixin:
    DATASET = "colored_mnist"

    def get_class_label(self, idx: int) -> str:
        return str(idx)          # digit label


class ImagenetMixin:
    DATASET = "imagenet"

    _IN1K_CATEGORIES: Optional[list] = None

    def get_class_label(self, idx: int) -> str:
        """ImageNet-1k class name (torchvision category table, metadata only —
        no weight download)."""
        cats = ImagenetMixin._IN1K_CATEGORIES
        if cats is None:
            from torchvision.models import ViT_B_16_Weights
            cats = list(ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"])
            ImagenetMixin._IN1K_CATEGORIES = cats
        return cats[idx]
