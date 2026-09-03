"""``find`` — instantiate a predefined ``(model, dataset)`` pair."""
from __future__ import annotations

from typing import Optional

from .model_dataset import ModelDataset


def find(model: str, dataset: str, *,
         device: Optional[str] = None,
         ds_extra: Optional[dict] = None,
         checkpoint: Optional[str] = None) -> ModelDataset:
    """Instantiate the :class:`ModelDataset` for ``(model, dataset)``.

    Args:
        model:      model identifier (``M_*`` constant, e.g. ``"vit_small"``).
        dataset:    dataset identifier (``DS_*`` constant, e.g. ``"funny_birds"``).
        device:     torch device; defaults to cuda-if-available.
        ds_extra:   per-experiment dataset-loader overrides merged over the
                    :data:`EVAL_DATASETS` defaults (e.g. ``{"split": "test"}``,
                    ``{"n_per_class": 10}``).
        checkpoint: explicit checkpoint path for finetuned-probe models (off-the-
                    shelf models ignore it; finetuned models otherwise pick the
                    newest matching ``data/runs/...`` checkpoint).

    Raises:
        ValueError: if ``(model, dataset)`` is not a registered pair.
    """
    # Imported here (not at module top) to avoid a package import cycle:
    # experiments.model_datasets.__init__ imports this module.
    from . import MODEL_DATASET_REGISTRY

    key = (model.lower(), dataset.lower())
    if key not in MODEL_DATASET_REGISTRY:
        raise ValueError(
            f"no ModelDataset for model={model!r} dataset={dataset!r}; "
            f"choose from {sorted(MODEL_DATASET_REGISTRY)}")
    cls = MODEL_DATASET_REGISTRY[key]
    return cls(device=device, ds_extra=ds_extra, checkpoint=checkpoint)


def find_by_tag(tag: str, *,
                device: Optional[str] = None,
                ds_extra: Optional[dict] = None,
                checkpoint: Optional[str] = None) -> ModelDataset:
    """Like :func:`find`, but selects by the flat ``<model>_<dataset>`` tag
    (e.g. ``"vit_base_imagenet"``) rather than the two axes.

    For call sites whose stable identifier / CLI is the historical single tag
    (``--model-key vit_base_imagenet``): this maps that tag to the registry pair
    without the caller having to split it (which is ambiguous — model names
    themselves contain underscores). ``md.tag`` round-trips the input.
    """
    from . import MODEL_DATASET_REGISTRY

    t = tag.lower()
    for model, dataset in MODEL_DATASET_REGISTRY:
        if f"{model}_{dataset}" == t:
            return find(model, dataset, device=device,
                        ds_extra=ds_extra, checkpoint=checkpoint)
    known = sorted(f"{m}_{d}" for m, d in MODEL_DATASET_REGISTRY)
    raise ValueError(f"no ModelDataset for tag {tag!r}; choose from {known}")
