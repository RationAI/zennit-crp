# Vision-Transformer CRP — tutorials

End-to-end demos for the four attention-concept granularities defined in
[`crp.attention_concepts`](../../crp/attention_concepts.py):

| Concept class         | Hook tap         | Granularity                                  | `attribute()` shape            |
|-----------------------|------------------|----------------------------------------------|--------------------------------|
| `HeadConcept`         | `attn_out_tap`   | output tokens, per head                      | `(B, num_heads)`               |
| `HeadDimConcept`      | `attn_out_tap`   | output tokens, per `(head, dim)`             | `(B, num_heads, head_dim)`     |
| `KQVHeadConcept`      | `qkv_tap`        | K/Q/V projections, per `(part, head)`        | `(B, 3, num_heads)`            |
| `KQVHeadDimConcept`   | `qkv_tap`        | K/Q/V projections, per `(part, head, dim)`   | `(B, 3, num_heads, head_dim)`  |

The four classes cross two orthogonal granularity axes — *split by K/Q/V?*
(qkv_tap vs the per-head output tokens at attn_out_tap) and *split by
head_dim?* (sum the head_dim axis or keep it). Both `qkv_tap` and
`attn_out_tap` are `nn.Identity` submodules installed by
`AttentionTapsCanonizer`.

This directory only contains tutorial-shaped material — the walkthrough
notebook and a single-image CLI demo. Sweeps, milestone drivers and
diagnostic scripts that drove the design live in
[`experiments/`](../../experiments/) and write their artefacts under
[`<repo>/data/`](../../data) (gitignored).

## Setup

Dependencies are managed with `uv`. From the repo root:

```bash
uv sync --extra vit --extra dev --extra notebook
```

`timm` and `Pillow` are pinned in main `dependencies`. The optional extras:
`vit` pulls in HuggingFace `transformers` (reserved for the upcoming
HF-ViT canonizer); `dev` adds `pytest` / `ruff`; `notebook` adds Jupyter
and `torchvision` (used by the walkthrough notebook for its dataset
wrapper).

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

[`walkthrough.ipynb`](walkthrough.ipynb) is the recommended entry point.
It:

1. Downloads an Imagenette subset (~98 MB) — real ImageNet images, ten
   classes, mapped back to ImageNet-1k indices. Cached at
   `data/imagenette2-160/`.
2. Builds a `FeatureVisualization` index for each of the four concept
   granularities. Each granularity reads from its own tap (`attn_out_tap`
   or `qkv_tap`) — the notebook resolves the layer name from the
   concept's `tap_name`. Cached at `data/feature_visualization/<name>/`.
3. For one target image, ranks each granularity's concepts by relevance
   under the true class, then displays top-k reference samples and
   conditional heatmaps.

Defaults to `vit_base_patch16_224` and 64 indexed images. The first cell
of section 2 is the only place you should need to override (model /
sample count / device).

The notebook is committed directly; edit it in Jupyter. To rebuild from
scratch, delete `data/feature_visualization/` (and optionally
`data/imagenette2-160*` to re-download).

## Single-image comparative heatmaps

The notebook also runs the four concept granularities on one selected
image, picks the top-k concepts per granularity (ranked by absolute
relevance under the target class), and renders a comparison grid — same
content as the previous standalone `demo.py`, now folded into the
walkthrough so all results are presented in one notebook.

ImageNet target-class indices: 281 is *tabby cat*, 207 is *golden
retriever*, 817 is *sports car*, etc. (full list at
`https://github.com/anishathalye/imagenet-simple-labels`).

The notebook's `MID_LAYER` cell chooses which ViT attention layer to
attribute through; each concept class auto-resolves the right tap
(`attn_out_tap` or `qkv_tap`) on that layer. For
`vit_base_patch16_224` (12 layers), mid-network layers (5–9) tend to carry
the most class-relevant structure.

## Hyperparameters that matter

* **Composite**: two choices, both pre-bundled with `TimmViTCanonizer`:
  * `AttnLRPEpsilonComposite` — ε-LRP on every linear (`Linear`,
    `Conv2d`). Fast, but per AttnLRP §3.2.1 prone to gradient-shattering
    noise on deep ViTs.
  * `AttnLRPGammaComposite(gamma=0.25)` — γ-LRP on linears (single-branch
    positive-weight clamp). The recommended default for ViT linears.
* **Layer index**: each attention layer is independent; the choice is
  empirical. Mid- to late-network layers usually carry class-relevant
  structure; very-early layers carry low-level features.

## What's next (this fork's roadmap)

The active backlog lives in [`FUTURE_STATE.md`](../../FUTURE_STATE.md).
Sweeps and audits supporting it sit in
[`experiments/`](../../experiments/).
