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
