# Vision-Transformer Concept Comparison

End-to-end demos for the four attention-concept granularities defined in
[`crp.attention_concepts`](../../crp/attention_concepts.py):

| Concept class         | Hook tap         | Granularity                                  | `attribute()` shape            |
|-----------------------|------------------|----------------------------------------------|--------------------------------|
| `HeadConcept`         | `attn_out_tap`   | output tokens, per head                      | `(B, num_heads)`               |
| `HeadDimConcept`      | `attn_out_tap`   | output tokens, per `(head, dim)`             | `(B, num_heads, head_dim)`     |
| `KQVHeadConcept`      | `qkv_tap`        | K/Q/V projections, per `(part, head)`        | `(B, 3, num_heads)`            |
| `KQVHeadDimConcept`   | `qkv_tap`        | K/Q/V projections, per `(part, head, dim)`   | `(B, 3, num_heads, head_dim)`  |

The four classes cross two orthogonal granularity dimensions: **whether
to split by K/Q/V** (qkv_tap vs the per-head output tokens at attn_out_tap)
and **whether to split per head_dim** (sum the head_dim axis or keep it).
Both `qkv_tap` and `attn_out_tap` are `nn.Identity` submodules installed
by `AttentionTapsCanonizer`.

## Setup

Dependencies are managed with `uv`. From the repo root:

```bash
uv sync --extra vit --extra dev --extra notebook
```

`timm` and `Pillow` are pinned in main `dependencies`. The optional extras:
`vit` pulls in HuggingFace `transformers` (reserved for the upcoming HF-ViT
canonizer); `dev` adds `pytest`/`ruff`; `notebook` adds Jupyter and
`torchvision` (used by the walkthrough notebook for its dataset wrapper).

To add a new dependency, prefer `uv add`:

```bash
uv add some-package                # runtime dep
uv add --optional dev some-package # dev-only
```

`uv sync` then re-resolves and rewrites `uv.lock`, which is committed for
reproducibility.

### HuggingFace token (optional but recommended)

`timm.create_model(..., pretrained=True)` downloads weights via the
HuggingFace Hub. Without an authenticated request you'll see a *"sending
unauthenticated requests to the HF Hub"* warning and may hit rate limits
on a shared host. The `huggingface_hub` library auto-reads a token from
`~/.cache/huggingface/token` (its default `HF_HOME/token`), so the simplest
setup is:

```bash
mkdir -p ~/.cache/huggingface
echo "hf_xxx_your_token" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

No env var or code change needed; `timm` will pick it up automatically.
Alternatively `uv run huggingface-cli login` writes the same file
interactively.

## Walkthrough notebook (start here)

[`walkthrough.ipynb`](walkthrough.ipynb) is the recommended entry point. It:

1. Downloads an Imagenette subset (~98 MB) — real ImageNet images, ten
   classes, mapped back to ImageNet-1k indices.
2. Builds a `FeatureVisualization` index for each of the four concept
   granularities. Each granularity reads from its own tap (`attn_out_tap`
   or `qkv_tap`) — the notebook resolves the layer name from the
   concept's `tap_name`.
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

The `--block` flag chooses the ViT attention block to attribute through;
each concept hooks the right tap on that block automatically. For
`vit_base_patch16_224` (12 blocks), mid-network blocks (5–9) tend to carry
the most class-relevant structure.

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
* **`top_k`**: granularity-dependent. On ViT-B (12 heads × 64 head_dim):
  `head` has 12 concepts (`k=4` ≈ ⅓), `head_dim` 768, `kqv_head` 36,
  `kqv_head_dim` 2 304 (sparse `k=8` ≈ 0.3 %). The
  `PER_GRANULARITY_TOP_K` map in `metrics.py` ships sensible defaults; the
  Petsiuk methodology requires `k ≪ num_concepts` for the random baseline
  to actually differ from the relevance-ranked top-k.

## What's next (this fork's roadmap)

The active backlog lives in [`FUTURE_STATE.md`](../../FUTURE_STATE.md). Headline
items:

1. Stability metric (heatmap cosine sim under input noise).
2. Localisation metric (pointing game on ImageNet-S or annotation-augmented
   val) — needs box/segmentation labels.
3. Conservation quantitative test; optional PA-LRP positional-encoding rule
   (Bakish et al., NeurIPS 2025) if it fails materially.
