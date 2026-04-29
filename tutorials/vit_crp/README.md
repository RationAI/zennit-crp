# Vision-Transformer Concept Comparison

End-to-end demos for the four attention-concept granularities defined in
[`crp.attention_concepts`](../../crp/attention_concepts.py):

| Concept class       | Granularity                                        | `attribute()` shape           |
|---------------------|----------------------------------------------------|-------------------------------|
| `HeadConcept`       | one concept per attention head                     | `(B, num_heads)`              |
| `KQVConcept`        | three concepts per block (whole Q / K / V)         | `(B, 3)`                      |
| `KQVHeadConcept`    | per `(part, head)` — `3 × num_heads`               | `(B, 3, num_heads)`           |
| `HeadDimConcept`    | per `(part, head, dim)` — `3 × num_heads × head_dim` | `(B, 3, num_heads, head_dim)` |

All four hook the same named tap (`attn.qkv_tap`, an `nn.Identity` injected
by `inject_qkv_taps`) so they can be compared on equal footing.

## Setup

Dependencies are managed with `uv`. From the repo root:

```bash
uv sync --extra vit --extra dev --extra notebook
```

The optional extras: `vit` pulls in `timm` and `transformers`; `dev` adds
`pytest`/`ruff`; `notebook` adds Jupyter and `torchvision` (used by the
walkthrough notebook for its dataset wrapper).

To add a new dependency, prefer `uv add`:

```bash
uv add some-package                # runtime dep
uv add --optional dev some-package # dev-only
```

`uv sync` then re-resolves and rewrites `uv.lock`, which is committed for
reproducibility.

## Walkthrough notebook (start here)

[`walkthrough.ipynb`](walkthrough.ipynb) is the recommended entry point. It:

1. Downloads an Imagenette subset (~98 MB) — real ImageNet images, ten
   classes, mapped back to ImageNet-1k indices.
2. Builds a `FeatureVisualization` index for each of the four concept
   granularities, all hooking the same `qkv_tap`.
3. For one target image, ranks each granularity's concepts by relevance
   under the true class, then displays top-k reference samples and
   conditional heatmaps.

Defaults to `vit_base_patch16_224` and 64 indexed images. The first cell of
section 2 is the only place you should need to override (model / sample
count / device).

The notebook source lives in [`_build_notebook.py`](_build_notebook.py) for
reviewable diffs; re-emit `walkthrough.ipynb` with
`uv run python tutorials/vit_crp/_build_notebook.py`.

## Standalone demo (CLI): comparative heatmaps

`demo.py` runs the four concept granularities on a single image, picks the
top-`k` concepts per granularity (ranked by absolute relevance under the
target class), and renders a comparison grid.

```bash
uv run python tutorials/vit_crp/demo.py \
    --image path/to/image.jpg \
    --target-class 281 \
    --block 6 \
    --top-k 4 \
    --out figures/comparison.png
```

ImageNet target-class indices: 281 is *tabby cat*, 207 is *golden retriever*,
817 is *sports car*, etc. (full list at
`https://github.com/anishathalye/imagenet-simple-labels`).

The `--block` flag chooses which ViT attention block's `qkv_tap` is hooked.
For `vit_base_patch16_224` (12 blocks), mid-network blocks (5–9) tend to
carry the most class-relevant structure.

## Quantitative comparison

`metrics.py` computes two faithfulness metrics (deletion AUC and insertion
AUC, Petsiuk et al., BMVC 2018) for each concept granularity, against a
random-concept baseline:

```bash
mkdir -p data/imagenet_subset
# populate data/imagenet_subset with 8–16 images of the chosen class
uv run python tutorials/vit_crp/metrics.py \
    --image-dir data/imagenet_subset \
    --target-class 281 \
    --block 6 \
    --top-k 8 \
    --out results.csv
```

The CSV has one row per `(image, concept_def, mode)` triple, where `mode ∈
{true, random}`. The expected reading is:

* **deletion AUC**: lower is better. The faster the model's
  target-class probability collapses as the heatmap-ranked top patches are
  masked, the more faithful the heatmap.
* **insertion AUC**: higher is better. The faster the probability rises as
  the heatmap-ranked top patches are revealed (from a blurred baseline), the
  more faithful.
* **true vs random**: the gap quantifies how much of the faithfulness comes
  from the *concept structure* vs. just having any heatmap of comparable
  energy.

## Hyperparameters that matter

* **Composite**: two choices, both pre-bundled with `TimmViTCanonizer`:
  * `AttnLRPEpsilonComposite` — ε-LRP on every linear (`Linear`, `Conv2d`).
    Fast, but per AttnLRP §3.2.1 prone to gradient-shattering noise on deep
    ViTs.
  * `AttnLRPGammaComposite(gamma=0.25)` — γ-LRP on linears (single-branch
    positive-weight clamp). The recommended default for ViT linears.
* **Block index**: each attention block is independent; the choice is
  empirical. Mid- to late-network blocks usually carry class-relevant
  structure; very-early blocks carry low-level features.
* **`top_k`**: for `head_dim` (3 × 12 × 64 = 2304 concepts on ViT-B), `k=8`
  picks the eight most-relevant feature dimensions across all parts and
  heads. The right `k` is granularity-dependent: `head` has only 12
  concepts, so `k=4` gives a third of them.

## What's next (this fork's roadmap)

The active backlog lives in [`FUTURE_STATE.md`](../../FUTURE_STATE.md). Headline
items:

1. Stability metric (heatmap cosine sim under input noise).
2. Localisation metric (pointing game on ImageNet-S or annotation-augmented
   val) — needs box/segmentation labels.
3. Conservation quantitative test; optional PA-LRP positional-encoding rule
   (Bakish et al., NeurIPS 2025) if it fails materially.
