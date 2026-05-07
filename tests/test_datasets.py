"""Smoke tests for the experiments/datasets/ package.

Pure import + API contract tests — no actual download / training.
Datasets that require ~GB downloads are tested with ``auto_download=False``
and the test verifies the loader raises ``FileNotFoundError`` cleanly when
the data is missing (i.e. the auto-setup machinery exists and is wired
correctly, even if we can't run the download in CI).

Datasets we do exercise end-to-end (when small enough to download):
* dsprites — ~26 MB, downloads if missing during the test.

Run::

    uv run pytest tests/test_datasets.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_imports():
    """All expected classes + functions importable from the package."""
    from experiments.datasets import (
        CuratedDataset, DATASETS, IMAGENETTE_TO_IMAGENET,
        load, load_imagenette, load_imagenet_val_hf,
        ImagenetteDataset, ImagenetValHFDataset,
        FunnyBirdsDataset, DSpritesDataset,
    )
    expected_names = {"imagenette", "imagenet_val_hf",
                      "funny_birds", "dsprites"}
    assert set(DATASETS) == expected_names


def test_load_dispatcher_unknown_name():
    """Dispatcher rejects unknown dataset name with a clear error."""
    from experiments.datasets import load
    with pytest.raises(ValueError, match="unknown dataset"):
        load("not_a_real_dataset")


def test_funny_birds_no_auto_download_missing():
    """FunnyBirdsDataset(auto_download=False) raises cleanly when data missing."""
    from experiments.datasets import FunnyBirdsDataset
    with pytest.raises(FileNotFoundError, match="auto_download=False"):
        FunnyBirdsDataset(
            root=REPO_ROOT / "tests" / "_nonexistent_data",
            split="train", auto_download=False,
        )


def test_funny_birds_invalid_split():
    """FunnyBirdsDataset rejects invalid split."""
    from experiments.datasets import FunnyBirdsDataset
    with pytest.raises(ValueError, match="split must be"):
        FunnyBirdsDataset(
            root=REPO_ROOT / "tests" / "_nonexistent_data",
            split="invalid", auto_download=False,
        )


def test_dsprites_no_auto_download_missing(tmp_path):
    """DSpritesDataset(auto_download=False) raises cleanly when data missing."""
    from experiments.datasets import DSpritesDataset
    with pytest.raises(FileNotFoundError, match="auto_download=False"):
        DSpritesDataset(
            root=tmp_path / "no_dsprites",
            target="shape", auto_download=False,
        )


def test_dsprites_invalid_target(tmp_path):
    """DSpritesDataset rejects invalid target factor."""
    from experiments.datasets import DSpritesDataset
    with pytest.raises(ValueError, match="target must be one of"):
        DSpritesDataset(
            root=tmp_path / "no_dsprites",
            target="invalid", auto_download=False,
        )
    # 'color' has only 1 latent value and is also rejected as a target.
    with pytest.raises(ValueError, match="target must be one of"):
        DSpritesDataset(
            root=tmp_path / "no_dsprites",
            target="color", auto_download=False,
        )


def test_dsprites_invalid_image_mode(tmp_path):
    """DSpritesDataset rejects invalid image mode."""
    from experiments.datasets import DSpritesDataset
    with pytest.raises(ValueError, match="image_mode must be"):
        DSpritesDataset(
            root=tmp_path / "no_dsprites",
            image_mode="invalid", auto_download=False,
        )


# Use the project's existing data dir if available so we don't re-download
# in environments where the test runner runs multiple times.
@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "dsprites" / "dsprites.npz").is_file(),
    reason="dsprites cached file not present — skip end-to-end test "
           "(unset to actually trigger the ~26 MB download)",
)
def test_dsprites_end_to_end_cached():
    """If dsprites is already on disk, verify it loads + serves a sample."""
    from experiments.datasets import DSpritesDataset
    ds = DSpritesDataset(root=REPO_ROOT / "data", target="shape")
    assert len(ds) == 737280
    assert ds.num_classes == 3
    img, label = ds[0]
    # PIL image in RGB mode (default).
    from PIL import Image as _PIL
    assert isinstance(img, _PIL.Image)
    assert img.mode == "RGB"
    assert img.size == (64, 64)
    assert label in (0, 1, 2)
