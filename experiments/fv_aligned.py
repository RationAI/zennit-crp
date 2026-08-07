"""Condition-class-aligned FeatureVisualization index.

The stock FV index conditions every image's attribution on its ground-truth
label — mislabeled images then contaminate the representatives. The *aligned*
index instead conditions each image on its **top-3 predicted classes with
probability > 0.10** (at least the top-1), one backward pass per class, and
stores the references **per conditioning class** (stock CRP's per-target
``Statistics`` store). Serving then only returns representatives whose indexing
condition matches the class the current explanation conditions on.

Layout inside the aligned FV directory (sibling of the original index, suffix
``--aligned`` — never mixed):

* ``predicted_targets.npz`` — ``targets`` (N, 3) int array padded with -1 +
  provenance (threshold, top-k, model tag, commit).
* stock CRP stores: ``RelMax_sum_normed/`` (target-agnostic, multi-conditioned)
  and ``RelStats_sum_normed/<layer>/<class>_{data,rel,rf}.npy`` + ``targets.npy``
  — the per-(concept, conditioning-class) index this module serves from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from crp.helper import load_stat_targets, load_statistics
from crp.visualization import FeatureVisualization

TOP_K = 3
PROB_THRESHOLD = 0.10
PAD = -1                       # padding value in the (N, 3) target rows; never a class


# ─────────────────────────────────────────────────────────────────────────────
# Prediction pass — which classes each image conditions the index on
# ─────────────────────────────────────────────────────────────────────────────

def predict_targets(model, ds, normalize, device: str, *, batch_size: int = 64) -> np.ndarray:
    """(N, TOP_K) int array: per image the top-``TOP_K`` predicted classes with
    softmax probability > ``PROB_THRESHOLD``, padded with ``PAD``. The top-1
    prediction is always kept, threshold or not, so every image conditions on at
    least one class."""
    model.eval()
    rows = np.full((len(ds), TOP_K), PAD, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            idxs = range(start, min(start + batch_size, len(ds)))
            x = torch.stack([ds[i][0] for i in idxs])
            probs = model(normalize(x.to(device))).softmax(-1)
            top_p, top_c = probs.topk(TOP_K, dim=-1)
            keep = top_p > PROB_THRESHOLD
            keep[:, 0] = True                              # top-1 always conditions
            top_c[~keep] = PAD
            rows[start:start + len(x)] = top_c.cpu().numpy()
    return rows


def targets_path(fv_dir: Path) -> Path:
    return Path(fv_dir) / "predicted_targets.npz"


def save_predicted_targets(fv_dir: Path, targets: np.ndarray, *, provenance: dict) -> None:
    fv_dir = Path(fv_dir)
    fv_dir.mkdir(parents=True, exist_ok=True)
    np.savez(targets_path(fv_dir), targets=targets,
             provenance=json.dumps({"top_k": TOP_K, "prob_threshold": PROB_THRESHOLD,
                                    **provenance}))


def load_predicted_targets(fv_dir: Path) -> np.ndarray:
    with np.load(targets_path(fv_dir)) as z:
        return z["targets"]


# ─────────────────────────────────────────────────────────────────────────────
# Aligned FV build — one image fanned out into one condition per predicted class
# ─────────────────────────────────────────────────────────────────────────────

class PredTargetsDataset:
    """Wrap an (image, label) dataset so item ``i`` yields
    ``(image, targets_row)`` with the fixed-width predicted-class row. Only used
    to feed :class:`AlignedFeatureVisualization`; everything else (ranking,
    sample picking) keeps reading the plain dataset."""

    def __init__(self, ds, targets: np.ndarray):
        if len(ds) != len(targets):
            raise ValueError(f"targets rows ({len(targets)}) != dataset size ({len(ds)})")
        self.ds = ds
        self.targets = targets

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0], self.targets[i]


class AlignedFeatureVisualization(FeatureVisualization):
    """FeatureVisualization over a :class:`PredTargetsDataset`: each image is
    broadcast into one ``{y: [c]}`` condition per predicted class (stock
    ``run_distributed`` fan-out via :meth:`multitarget_to_single`), so the
    per-target ``Statistics`` store becomes the condition-class-aligned index."""

    def multitarget_to_single(self, multi_target) -> list:
        return [int(t) for t in np.asarray(multi_target) if int(t) != PAD]

    def get_data_concurrently(self, indices, preprocessing=False):
        # Stock impl unwraps a length-1 batch to a bare (data, target) pair,
        # which run_distributed would then iterate target-elementwise —
        # treating each predicted class of ONE image as a separate sample.
        # Always return a batch (data, [row, ...]) so the fan-out loop sees
        # one fixed-width row per sample regardless of batch length.
        if len(indices) == 1:
            data, target = self.get_data_sample(indices[0], preprocessing)
            return data, [target]
        return super().get_data_concurrently(indices, preprocessing)


# ─────────────────────────────────────────────────────────────────────────────
# Aligned serving — representatives whose indexing condition matches
# ─────────────────────────────────────────────────────────────────────────────

def aligned_references_for_class(fv, layer: str, cid: int, target: int,
                                 n_ref: int) -> tuple:
    """Top-``n_ref`` (dataset_index, conditioning_class) references of concept
    ``cid`` **conditioned on class ``target``** — the exact-match rule for
    explaining an image conditionally on ``target``. Fewer than ``n_ref`` rows
    is a data property (few images had ``target`` among their predicted
    classes), not an error."""
    stats_path = fv.RelStats.PATH
    known = set(int(t) for t in load_stat_targets(stats_path))
    if int(target) not in known:
        raise KeyError(
            f"class {target} was never a conditioning target in the aligned index "
            f"at {stats_path} (known targets: {len(known)})")
    d_sorted, _, _ = load_statistics(stats_path, layer, int(target))
    idxs = [int(i) for i in np.asarray(d_sorted)[:n_ref, int(cid)]]
    return idxs, [int(target)] * len(idxs)


def aligned_references_merged(fv, layer: str, cid: int, n_ref: int) -> tuple:
    """Aggregate view: top-``n_ref`` references of concept ``cid`` across ALL
    conditioning classes, each remembering the class it was conditioned on (its
    heatmap is later recomputed under that same class). Global order by indexed
    relevance; one row per (sample, class) pair — the same image may appear
    under two classes, which is the point of the aligned index."""
    stats_path = fv.RelStats.PATH
    rows = []                                  # (rel, ds_index, class)
    for t in load_stat_targets(stats_path):
        d_sorted, rel_sorted, _ = load_statistics(stats_path, layer, int(t))
        d_col = np.asarray(d_sorted)[:, int(cid)]
        r_col = np.asarray(rel_sorted)[:, int(cid)]
        for d, r in zip(d_col, r_col):
            rows.append((float(r), int(d), int(t)))
    rows.sort(key=lambda x: x[0], reverse=True)
    top = rows[:n_ref]
    return [d for _, d, _ in top], [t for _, _, t in top]
