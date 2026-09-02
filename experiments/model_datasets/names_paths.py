"""Canonical model / dataset identifier constants.

The two axes are kept **independent** (LLEXICORP paradigm): a model string and a
dataset string. The on-disk artefact tag is their flat underscore-join
``f"{model}_{dataset}"`` (see :pyattr:`ModelDataset.tag`) — chosen so every
existing FV-cache / gallery-figure directory (``vit_base_imagenet``,
``vit_dinov3_base_imagenet``, …) round-trips unchanged. **Keep these strings
stable** — they name persistent artefacts on disk.
"""

# ── Model axis (must match the `<base>` half of the historical fused tag) ──────
M_VIT_SMALL = "vit_small"
M_VIT_BASE = "vit_base"
M_VIT_BASE_TORCHVISION = "vit_base_torchvision"
M_VIT_DINOV3_SMALL = "vit_dinov3_small"
M_VIT_DINOV3_BASE = "vit_dinov3_base"

# ── Dataset axis (must match the eval-dataset keys in experiments.datasets) ────
DS_FUNNY_BIRDS = "funny_birds"
DS_IMAGENET = "imagenet"
DS_DSPRITES = "dsprites"
DS_COLORED_MNIST = "colored_mnist"

# Journal-default model per dataset — for dataset-keyed experiments (e.g.
# concept flipping) that imply the model. Supersedes the old
# ``experiments.models.zoo.DEFAULT_MODELS``.
DEFAULT_MODEL_FOR_DATASET = {
    DS_FUNNY_BIRDS:   M_VIT_SMALL,
    DS_DSPRITES:      M_VIT_SMALL,
    DS_COLORED_MNIST: M_VIT_SMALL,
    DS_IMAGENET:      M_VIT_BASE,
}
