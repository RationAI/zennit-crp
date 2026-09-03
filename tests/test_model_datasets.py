"""Contract tests for the experiments/model_datasets/ package.

Structural / API tests — no weight downloads or GPU. The registry, the flat
``tag`` join (which must reproduce existing on-disk FV-cache / figure dirs),
``find`` validation, the default-model map and the dataset-mixin class labels
are all checked without instantiating a heavy model. One optional end-to-end
``find`` smoke runs only when the local FunnyBirds probe checkpoint is present
(skipped otherwise, so CI stays light).

Run::

    uv run pytest tests/test_model_datasets.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The canonical model tags that name persistent artefacts on disk (FV caches,
# gallery figure trees). The registry's (model, dataset) tuples must join to
# exactly this set — a regression here would orphan existing cached data.
EXPECTED_TAGS = {
    "vit_small_funny_birds",
    "vit_dinov3_small_funny_birds",
    "vit_small_dsprites",
    "vit_small_colored_mnist",
    "vit_base_imagenet",
    "vit_base_torchvision_imagenet",
    "vit_dinov3_base_imagenet",
}


def test_imports():
    """Public surface importable from the package."""
    from experiments.model_datasets import (  # noqa: F401
        find, ModelDataset, MODEL_DATASET_REGISTRY, DEFAULT_MODEL_FOR_DATASET,
        AVAILABLE_MODELS, AVAILABLE_DATASETS, get_available_combinations,
    )
    assert len(MODEL_DATASET_REGISTRY) == len(EXPECTED_TAGS)


def test_registry_keys_are_axis_tuples():
    """Every key is a (model, dataset) tuple drawn from the declared axes."""
    from experiments.model_datasets import MODEL_DATASET_REGISTRY
    from experiments.model_datasets import names_paths as np_

    models = {np_.M_VIT_SMALL, np_.M_VIT_BASE, np_.M_VIT_BASE_TORCHVISION,
              np_.M_VIT_DINOV3_SMALL, np_.M_VIT_DINOV3_BASE}
    datasets = {np_.DS_FUNNY_BIRDS, np_.DS_IMAGENET, np_.DS_DSPRITES,
                np_.DS_COLORED_MNIST}
    for key in MODEL_DATASET_REGISTRY:
        assert isinstance(key, tuple) and len(key) == 2
        model, dataset = key
        assert model in models, model
        assert dataset in datasets, dataset


def test_registry_values_are_modeldataset_subclasses():
    from experiments.model_datasets import MODEL_DATASET_REGISTRY, ModelDataset
    for cls in MODEL_DATASET_REGISTRY.values():
        assert issubclass(cls, ModelDataset)


def test_tag_join_reproduces_ondisk_dirs():
    """flat f"{model}_{dataset}" over the registry == the canonical tag set.

    This is the invariant that lets the tuple registry keep the historical flat
    directory names (zero data migration).
    """
    from experiments.model_datasets import MODEL_DATASET_REGISTRY
    derived = {f"{m}_{d}" for m, d in MODEL_DATASET_REGISTRY}
    assert derived == EXPECTED_TAGS


def test_concrete_class_attrs_match_registry_key():
    """Each class's (MODEL, DATASET) attrs equal its registry key."""
    from experiments.model_datasets import MODEL_DATASET_REGISTRY
    for (model, dataset), cls in MODEL_DATASET_REGISTRY.items():
        assert cls.MODEL == model, (cls.__name__, cls.MODEL, model)
        assert cls.DATASET == dataset, (cls.__name__, cls.DATASET, dataset)


def test_available_derived_sets():
    from experiments.model_datasets import (
        MODEL_DATASET_REGISTRY, AVAILABLE_MODELS, AVAILABLE_DATASETS,
        get_available_combinations,
    )
    assert AVAILABLE_MODELS == sorted({m for m, _ in MODEL_DATASET_REGISTRY})
    assert AVAILABLE_DATASETS == sorted({d for _, d in MODEL_DATASET_REGISTRY})
    assert get_available_combinations() == sorted(MODEL_DATASET_REGISTRY.keys())


def test_default_model_for_dataset_are_registered_pairs():
    """Every (default_model, dataset) is a real registry pair, and the map
    covers every dataset axis exactly once."""
    from experiments.model_datasets import (
        DEFAULT_MODEL_FOR_DATASET, MODEL_DATASET_REGISTRY, AVAILABLE_DATASETS,
    )
    for dataset, model in DEFAULT_MODEL_FOR_DATASET.items():
        assert (model, dataset) in MODEL_DATASET_REGISTRY, (model, dataset)
    assert sorted(DEFAULT_MODEL_FOR_DATASET) == AVAILABLE_DATASETS


def test_find_unknown_pair_raises():
    """Unknown (model, dataset) raises ValueError before any model is built."""
    from experiments.model_datasets import find
    with pytest.raises(ValueError, match="no ModelDataset"):
        find("not_a_model", "not_a_dataset")
    with pytest.raises(ValueError, match="no ModelDataset"):
        find("vit_small", "imagenet")   # valid axes, unregistered combo


def test_imagenet_mixin_class_labels():
    """ImagenetMixin maps indices to the ImageNet-1k category names (metadata
    table only — no model instantiation)."""
    from experiments.model_datasets.model_dataset import ImagenetMixin
    m = ImagenetMixin()
    assert m.get_class_label(0) == "tench"
    assert m.get_class_label(207) == "golden retriever"
    assert m.get_class_label(817) == "sports car"


def test_colored_mnist_and_default_class_labels():
    from experiments.model_datasets.model_dataset import (
        ColoredMnistMixin, ModelDataset,
    )
    assert ColoredMnistMixin().get_class_label(7) == "7"
    # Datasets without real names (FunnyBirds/dsprites) fall back to the base's
    # generic label — it ignores self, so call it off the class directly.
    assert ModelDataset.get_class_label(None, 3) == "class 3"


# ── optional end-to-end smoke (only if the local probe checkpoint exists) ──────
_FB_CKPT_DIR = REPO_ROOT / "data" / "runs" / "finetune_vit_small_funny-birds-train-clean"


@pytest.mark.skipif(
    not (_FB_CKPT_DIR.is_dir() and list(_FB_CKPT_DIR.glob("*/best.pt"))),
    reason="FunnyBirds ViT-S probe checkpoint not present",
)
def test_find_instantiates_funnybirds_cpu():
    """find() builds a usable bundle on CPU; tag/num_classes correct; the
    dataset is lazy (not built by construction)."""
    from experiments.model_datasets import find
    md = find("vit_small", "funny_birds", device="cpu")
    assert md.tag == "vit_small_funny_birds"
    assert md.num_classes == 50
    assert md.label
    assert callable(md.transform) and callable(md.normalize)
    assert md._dataset is None          # lazy: dataset untouched
    assert md.get_class_label(0) == "class 0"
