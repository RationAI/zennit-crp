"""Model-dataset pairs for the experiments — the LLEXICORP paradigm.

A ``(model, dataset)`` tuple selects one :class:`ModelDataset` subclass that
bundles the frozen model, its eval dataset, the timm transform/normalize split
and metadata. Instantiate the predefined pair with :func:`find`::

    from experiments.model_datasets import find
    md = find("vit_small", "funny_birds")      # or find(M_VIT_SMALL, DS_FUNNY_BIRDS)
    logits = md.model(md.preprocess(pil_image))
    md.get_class_label(logits.argmax().item())

The registry is keyed by the two independent identifier axes; the on-disk
artefact tag (``md.tag``) is their flat join, preserving existing cache/figure
directories.
"""
from .find import find
from .model_dataset import ModelDataset
from .names_paths import (
    DEFAULT_MODEL_FOR_DATASET,
    DS_COLORED_MNIST,
    DS_DSPRITES,
    DS_FUNNY_BIRDS,
    DS_IMAGENET,
    M_VIT_BASE,
    M_VIT_BASE_TORCHVISION,
    M_VIT_DINOV3_BASE,
    M_VIT_DINOV3_SMALL,
    M_VIT_SMALL,
)
from .vit import (
    VitBaseImagenet,
    VitBaseTorchvisionImagenet,
    VitDinoV3BaseImagenet,
    VitDinoV3SmallFunnyBirds,
    VitSmallColoredMnist,
    VitSmallDsprites,
    VitSmallFunnyBirds,
)

# Registry: (model, dataset) → ModelDataset subclass.
MODEL_DATASET_REGISTRY = {
    (M_VIT_SMALL, DS_FUNNY_BIRDS):            VitSmallFunnyBirds,
    (M_VIT_DINOV3_SMALL, DS_FUNNY_BIRDS):     VitDinoV3SmallFunnyBirds,
    (M_VIT_SMALL, DS_DSPRITES):               VitSmallDsprites,
    (M_VIT_SMALL, DS_COLORED_MNIST):          VitSmallColoredMnist,
    (M_VIT_BASE, DS_IMAGENET):                VitBaseImagenet,
    (M_VIT_BASE_TORCHVISION, DS_IMAGENET):    VitBaseTorchvisionImagenet,
    (M_VIT_DINOV3_BASE, DS_IMAGENET):         VitDinoV3BaseImagenet,
}

# Derived from the registry.
AVAILABLE_MODELS = sorted({m for m, _ in MODEL_DATASET_REGISTRY})
AVAILABLE_DATASETS = sorted({d for _, d in MODEL_DATASET_REGISTRY})


def get_available_combinations() -> list[tuple[str, str]]:
    """All valid ``(model, dataset)`` pairs."""
    return sorted(MODEL_DATASET_REGISTRY.keys())


__all__ = [
    "find",
    "ModelDataset",
    "MODEL_DATASET_REGISTRY",
    "DEFAULT_MODEL_FOR_DATASET",
    "AVAILABLE_MODELS",
    "AVAILABLE_DATASETS",
    "get_available_combinations",
    "VitSmallFunnyBirds",
    "VitDinoV3SmallFunnyBirds",
    "VitSmallDsprites",
    "VitSmallColoredMnist",
    "VitBaseImagenet",
    "VitBaseTorchvisionImagenet",
    "VitDinoV3BaseImagenet",
]
