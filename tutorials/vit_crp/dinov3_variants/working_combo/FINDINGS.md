# `working_combo` — the DINOv3 AttnLRP recipe

```python
from crp.transformer_patches import AttnLRPCombinedComposite
composite = AttnLRPCombinedComposite(
    matmul_factor_2=True,        # AttnLRP Prop 3.3 bilinear rule
    layerscale_uniform=True,     # AttnLRP Eq. 7 uniform on LayerScale γ
    residual_lrp="ratio",        # Otsuki ratio split on residual additions
)
```

**Status:** ✅ produces well-conditioned heatmaps on DINOv3 ViT-L/16.
Median across 5 distinct Imagenette classes:

| metric | value | interpretation |
|---|---:|---|
| `max\|R\|`    | 1.5e+02 | well within fp32 range |
| `focus@10%`  | 0.81    | strongly localised (random = 0.10, perfect = 1.0) |
| conservation | within ~2 OOM of input logit | tight enough for inspection |
| NaN         | 0/5     | no overflow on any sample |

## Why all three rules are required

The diagnostic showed each ingredient addresses a distinct failure
mode in the bare ε-LRP pipeline on DINOv3:

* **`matmul_factor_2`** — bare PyTorch matmul has no LRP rule. Without
  this rule, relevance through `Q@Kᵀ` and `attn@V` is just unstabilised
  raw gradient and propagates incorrectly. Achtibat et al. ICML 2024,
  Prop. 3.3 (arXiv:2402.05602).
* **`residual_lrp='ratio'`** — bare residual `y = x + branch` doubles
  relevance per step (autograd backward is `grad_x = grad_y,
  grad_branch = grad_y`). Over 24×2 = 48 residuals, that's 2⁴⁸ ≈ 10¹⁴
  amplification. Otsuki-style ratio split distributes ∝ `|x|`/`|branch|`
  and conserves.
* **`layerscale_uniform`** — DINOv3 (and all CaiT/EvaBlock-style
  models) multiply each branch by a learnable scalar γ ≈ 1e-4. Without
  the uniform rule the LayerScale node passes through with simple
  `* γ` deflation; with the rule it splits relevance under AttnLRP
  Eq. 7. Required for any model with LayerScale.

Each ingredient was historically called a "remedy" in the early
triage. After the rule audit they are more accurately described as
**the AttnLRP rules for the corresponding op types**, not optional
remedies — bare ε-LRP has no rule for matmul / residual / LayerScale,
and the missing rules are what AttnLRP exists to add.

## What's in this folder

* [`FINDINGS.md`](FINDINGS.md) — this file.
* [`walkthrough.ipynb`](walkthrough.ipynb) — full demo:
  * Loads DINOv3 ViT-L/16 + linear probe trained on Imagenette
    (`data/vit_large_patch16_dinov3_probe_imagenette.pt`).
  * Picks 3 correctly-classified, class-distinct images.
  * Runs the diagnostic to show finite-share=1.0 and
    `max|R|` magnitudes per layer.
  * Plots side-by-side heatmaps (input | raw | normalised). Spatial
    structure visibly localises onto the class-relevant object.
  * Compares against the broken `AttnLRPEpsilonComposite()`
    baseline (all-NaN) for explicit contrast.
* [`_build_notebook.py`](./_build_notebook.py) — idempotent
  regenerator for the notebook.

## Caveats

* The 5 register tokens still receive zero relevance from the head
  (DINOv3's avg-pool head only looks at patch tokens). Per-block
  residual leakage *into* register tokens is observed downstream and
  is **expected** behaviour per Darcet et al. ICLR 2024
  (arXiv:2309.16588) — register tokens absorb high-norm artifacts; the
  diagnostic separates them out so the spatial heatmap stays clean.
* Conservation is "within ~2 OOM" not "within ε" because (a) we still
  use the ratio (not symmetric) residual rule for backward
  reproducibility with our ResNet pipeline, and (b) the bilinear rule
  conservation is approximate when many `Y` entries are near 0
  (softmax outputs). Tightening either is a follow-up.
* Earlier reports in this folder framed the recipe as a "winning
  remedy combination." After the rule audit the correct framing is:
  **this is the AttnLRP-baseline for transformers with LayerScale**;
  the prior single-remedy variants are *each* incomplete subsets.
