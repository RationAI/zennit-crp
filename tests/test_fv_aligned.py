"""Tests for the condition-class-aligned FV index (experiments.fv_aligned).

Covers the prediction-target selection rule, the multi-target fan-out through
stock ``run_distributed``, the per-conditioning-class Statistics store, and the
exact-match serving rule. CPU, vit_tiny, tiny synthetic dataset — plumbing and
store semantics, not visual quality.

Run::

    uv run pytest tests/test_fv_aligned.py -v
"""
from pathlib import Path

import numpy as np
import pytest
import torch

timm = pytest.importorskip("timm")
pytest.importorskip("zennit")

from crp.attribution import CondAttribution
from crp.concepts import EmbeddingDimConcept
from crp.helper import load_stat_targets, load_statistics

from experiments import fv_aligned
from zennit_extensions.lrp_composites import CPLRPComposite


class _TinyDataset:
    """Deterministic (image, label) dataset of n random 224x224 images."""

    def __init__(self, n: int = 8, num_classes: int = 10):
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 224, 224, generator=g)
        self.labels = [i % num_classes for i in range(n)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


@pytest.fixture(scope="module")
def model():
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    return m.eval()


@pytest.fixture(scope="module")
def ds():
    return _TinyDataset()


def test_predict_targets_rule(model, ds):
    """Top-3 with p > 0.10, top-1 always kept, PAD elsewhere; rows must agree
    with a direct softmax computation."""
    rows = fv_aligned.predict_targets(model, ds, lambda x: x, "cpu", batch_size=3)
    assert rows.shape == (len(ds), fv_aligned.TOP_K)
    with torch.no_grad():
        probs = model(ds.images).softmax(-1)
    top_p, top_c = probs.topk(fv_aligned.TOP_K, dim=-1)
    for i in range(len(ds)):
        expect = [int(top_c[i, k]) for k in range(fv_aligned.TOP_K)
                  if k == 0 or float(top_p[i, k]) > fv_aligned.PROB_THRESHOLD]
        got = [int(t) for t in rows[i] if t != fv_aligned.PAD]
        assert got == expect
    # top-1 present in every row
    assert (rows[:, 0] != fv_aligned.PAD).all()


def test_aligned_index_buckets_and_serving(model, ds, tmp_path):
    """End-to-end aligned build on one layer: every RelStats bucket contains
    only samples that had that class among their predicted targets, and the
    exact-match serving returns indices from the requested bucket only."""
    layer = "blocks.10.attn.proj_drop"
    rows = fv_aligned.predict_targets(model, ds, lambda x: x, "cpu")
    fv_aligned.save_predicted_targets(tmp_path, rows, provenance={"test": True})
    assert fv_aligned.load_predicted_targets(tmp_path).tolist() == rows.tolist()

    attribution = CondAttribution(model)
    concept = EmbeddingDimConcept(num_heads=3)
    fv = fv_aligned.AlignedFeatureVisualization(
        attribution, fv_aligned.PredTargetsDataset(ds, rows), {layer: concept},
        preprocess_fn=None, path=str(tmp_path), device="cpu")
    fv.run(CPLRPComposite(), 0, len(ds), batch_size=3)

    stats_path = Path(fv.RelStats.PATH)
    targets = load_stat_targets(stats_path)
    # the stored conditioning classes are exactly the predicted ones
    predicted = sorted({int(t) for r in rows for t in r if t != fv_aligned.PAD})
    assert sorted(int(t) for t in targets) == predicted

    for t in targets:
        d_sorted, _, _ = load_statistics(stats_path, layer, int(t))
        conditioned_on_t = {i for i in range(len(ds)) if int(t) in rows[i]}
        assert set(np.asarray(d_sorted).ravel().astype(int)) <= conditioned_on_t

    # exact-match serving: bucket t only
    t0 = int(targets[0])
    idxs, ys = fv_aligned.aligned_references_for_class(fv, layer, 0, t0, n_ref=4)
    assert ys == [t0] * len(idxs)
    assert all(t0 in rows[i] for i in idxs)

    # unknown class fails loud, no fallback
    with pytest.raises(KeyError):
        fv_aligned.aligned_references_for_class(fv, layer, 0, 9999, n_ref=4)

    # merged serving: globally sorted by indexed relevance, each row remembers
    # its conditioning class
    idxs_m, ys_m = fv_aligned.aligned_references_merged(fv, layer, 0, n_ref=6)
    assert len(idxs_m) == len(ys_m) <= 6
    assert all(y in rows[i] for i, y in zip(idxs_m, ys_m))


def test_single_sample_batch_fanout(model, ds, tmp_path):
    """A trailing batch of length 1 must still fan out its predicted classes as
    conditions of ONE sample (the stock length-1 unwrap would misread the
    target row as separate samples)."""
    layer = "blocks.10.attn.proj_drop"
    rows = fv_aligned.predict_targets(model, ds, lambda x: x, "cpu")
    attribution = CondAttribution(model)
    concept = EmbeddingDimConcept(num_heads=3)
    fv = fv_aligned.AlignedFeatureVisualization(
        attribution, fv_aligned.PredTargetsDataset(ds, rows), {layer: concept},
        preprocess_fn=None, path=str(tmp_path), device="cpu")
    # batch_size 7 over 8 samples → final batch has exactly 1 sample
    fv.run(CPLRPComposite(), 0, len(ds), batch_size=7)
    targets = load_stat_targets(Path(fv.RelStats.PATH))
    last_targets = {int(t) for t in rows[len(ds) - 1] if t != fv_aligned.PAD}
    assert last_targets <= {int(t) for t in targets}
