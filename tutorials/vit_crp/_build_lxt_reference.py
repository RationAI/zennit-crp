"""Build ``lxt_reference.ipynb`` — side-by-side LXT vs zennit-canonizer
comparison on torchvision's ``vit_b_16``.

Idempotent — re-run to regenerate the notebook. The notebook runs the
SAME torchvision ViT-B/16 model under two LRP recipes that should
produce the same input-relevance map:

* **Path A — zennit canonizers (ours)**: build a
  :class:`zennit.composites.LayerMapComposite` with our
  :class:`crp.transformer_patches.TorchvisionMHACPLRPCanonizer` (CP-LRP
  on every ``nn.MultiheadAttention``) + ``LayerNormForwardCanonizer`` +
  ``GELUIdentityRuleCanonizer`` + ``DropoutPassthroughCanonizer``, with
  ``Epsilon`` rule on Linear/Conv2d. zennit-native relevance space.

* **Path B — LXT baseline**: call
  ``lxt.efficient.monkey_patch_zennit()`` (patches
  ``zennit.core.BasicHook``) and
  ``lxt.efficient.monkey_patch(vit_torch.vision_transformer, cp_LRP)``
  (patches ``nn.GELU`` / ``nn.LayerNorm`` / ``nn.MultiheadAttention``
  classes), then build a composite with the same ``Epsilon`` rule.
  gradient×input space throughout.

The two paths use the same model, same weights, same input. Heatmaps
are normalised to the same convention before comparison:

* Path A: ``R_input = data.grad`` (zennit-stock returns R directly).
* Path B: ``R_input = data.grad * data`` (LXT-patched zennit returns
  R/input; multiply by input to recover R).

**Path-order matters.** ``monkey_patch_zennit`` and
``monkey_patch(vision_transformer, ...)`` mutate global state (zennit's
``BasicHook`` class + torchvision's ``nn.GELU`` etc. classes) that is
NOT reverted automatically. Run Path A first (stock classes); then
Path B (globally patched). Restart the kernel to start over.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(*lines):
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines):
    return nbf.v4.new_code_cell("\n".join(lines))


nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        "# LXT Reference vs Our Zennit Canonizers — torchvision ViT-B/16",
        "",
        "**Recipe ported from**",
        "[RationAI/crp-experimenting @ 99045d2](https://github.com/RationAI/crp-experimenting/commit/99045d227414ed04cff1dd782082e55721329d29)",
        "(`tutorials/attributions_transformer.py` + `tutorials/run_lrp.py`)",
        "— the prior working LXT-on-ViT setup that produces nice heatmaps.",
        "",
        "Side-by-side LRP heatmap on the same model under two recipes:",
        "",
        "* **Path A** — OUR zennit canonizers (",
        "  `TorchvisionMHACPLRPCanonizer` + `LayerNormForwardCanonizer` +",
        "  `GELUIdentityRuleCanonizer` + `DropoutPassthroughCanonizer`),",
        "  with LXT's `monkey_patch_zennit()` applied so zennit's",
        "  `BasicHook` runs in gradient×input convention.",
        "* **Path B** — LXT's PUBLISHED recipe end-to-end (",
        "  `monkey_patch_zennit()` +",
        "  `monkey_patch(vision_transformer)` — the call form used in the",
        "  reference notebook, which resolves the default `cp_LRP` map for",
        "  torchvision ViT).",
        "  Uses LXT's class-level forward-method patches on `nn.GELU` /",
        "  `nn.LayerNorm` / `nn.MultiheadAttention`.",
        "",
        "Both use ``torchvision.models.vit_b_16(weights=IMAGENET1K_V1)``,",
        "the same input image (``tutorials/images/lizard.jpg``), and the",
        "reference's layer_map: `Conv2d → Gamma(0.25)`, `Linear → Gamma(0.10)`.",
        "No `Pass()` entries on GELU/LayerNorm/Dropout — those modules'",
        "forwards are rewritten by either our canonizers (Path A) or LXT's",
        "class patches (Path B), and their backwards flow through autograd",
        "naturally. Both paths use the gradient×input convention via",
        "`monkey_patch_zennit()`.",
        "",
        "**Why both paths call `monkey_patch_zennit()`.** zennit's stock",
        "`Epsilon` rule computes `R_in = input · R_y / (W·x + ε)`",
        "(relevance-space). On ViT-B/16 this compounds catastrophically",
        "across ~24 residual additions (LRP rule absent → autograd",
        "duplicates relevance at every `+`), yielding magnitudes of",
        "10¹⁹+ and unusable heatmaps. LXT's patched `BasicHook`",
        "reformulates the same rule in gradient×input convention",
        "(multiply by output going in, divide by input going out) which",
        "stays bounded. Sharing the convention isolates the comparison",
        "to **how each path patches attention / GELU / LayerNorm**.",
        "",
        "**Goal**: confirm OUR canonizers (instance-level, reversible",
        "via `composite.context()`) reproduce LXT's class-level",
        "monkey-patches numerically. If A == B, our canonizers are a",
        "drop-in zennit-side equivalent of LXT's vit_torch recipe.",
        "",
        "**Important — path ordering.** `monkey_patch_zennit()` and",
        "`monkey_patch(vision_transformer, cp_LRP)` mutate global state",
        "(zennit's `BasicHook` class + torchvision's `nn.GELU` etc.)",
        "that is NOT reverted automatically. Both paths apply",
        "`monkey_patch_zennit` once (no-op on the second call), and",
        "Path A runs FIRST — its canonizers are scoped to a `with",
        "composite.context(model)` block. Path B then applies LXT's",
        "class-level `monkey_patch(vit_torch, cp_LRP)` which globally",
        "rewrites the GELU/LayerNorm/MHA forwards. **Restart the kernel**",
        "to re-run from the top.",
        "",
        "## Installing LXT (reproduction-only dep)",
        "",
        "`lxt` is NOT in this project's `pyproject.toml`. It's used here",
        "purely as the reference for the comparison; once our canonizers",
        "match (this notebook's whole point), `lxt` can be removed.",
        "",
        "`lxt==2.1` was built against the old HuggingFace stack —",
        "`transformers<5` and `huggingface-hub<1`. The project ships",
        "`huggingface-hub>=1.12`, so a plain `uv pip install lxt`",
        "leaves you in a state where importing `lxt.efficient` fails",
        "(`lxt.efficient.__init__` transitively loads `bert.py` →",
        "`transformers` → version-check raises against hub 1.x).",
        "",
        "Working install recipe — run from the repo root:",
        "",
        "```powershell",
        "uv pip install --native-tls lxt 'transformers<5' 'huggingface-hub<1'",
        "```",
        "",
        "Then run the notebook with `uv run --no-sync ...` (or",
        "`uv run --frozen ...`) so `uv` doesn't re-resolve back to",
        "hub 1.x. `--native-tls` is needed if your network blocks uv's",
        "bundled TLS (SSL cert errors). The downgraded hub / transformers",
        "are scoped to this notebook's dep needs only — the rest of the",
        "project still runs (we don't actually invoke any v5-only API).",
        "",
        "To remove afterwards:",
        "",
        "```powershell",
        "uv pip uninstall lxt && uv sync",
        "```",
    ),

    # ─── 1. Setup ──────────────────────────────────────────────────────────
    md("## 1. Setup"),
    code(
        "from __future__ import annotations",
        "from pathlib import Path",
        "",
        "REPO_ROOT = Path.cwd()",
        "while not (REPO_ROOT / 'pyproject.toml').is_file():",
        "    REPO_ROOT = REPO_ROOT.parent",
        "",
        "import torch",
        "import torch.nn as nn",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from PIL import Image",
        "",
        "import torchvision",
        "from torchvision.models import vit_b_16, ViT_B_16_Weights",
        "",
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "torch.manual_seed(0)",
        "print(f'torch        : {torch.__version__}')",
        "print(f'torchvision  : {torchvision.__version__}')",
        "print(f'device       : {DEVICE}')",
    ),

    # ─── 2. Model + input ──────────────────────────────────────────────────
    md(
        "## 2. Load torchvision ViT-B/16 + sample image",
        "",
        "`vit_b_16(weights=IMAGENET1K_V1)` loads the original Google ViT-B/16",
        "ImageNet-1k checkpoint. Preprocessing comes from `weights.transforms()`",
        "— the exact resize/crop/normalize the checkpoint was trained with.",
    ),
    code(
        "weights = ViT_B_16_Weights.IMAGENET1K_V1",
        "preprocess = weights.transforms()",
        "categories = weights.meta['categories']",
        "",
        "model = vit_b_16(weights=weights).to(DEVICE).eval()",
        "for p in model.parameters():",
        "    p.requires_grad_(False)",
        "",
        "img_path = REPO_ROOT / 'tutorials' / 'images' / 'lizard.jpg'",
        "pil = Image.open(img_path).convert('RGB')",
        "x_norm = preprocess(pil).unsqueeze(0).to(DEVICE)",
        "",
        "with torch.no_grad():",
        "    logits = model(x_norm)",
        "top1 = int(logits.argmax(-1).item())",
        "print(f'top-1 class  : {top1} ({categories[top1]})')",
        "print(f'top-1 logit  : {logits[0, top1].item():.4f}')",
        "print(f'input shape  : {tuple(x_norm.shape)}')",
        "",
        "# Display image (after un-normalising for visualization).",
        "mean = torch.tensor(weights.meta.get('_metrics', {}).get('mean')",
        "                    if isinstance(weights.meta.get('_metrics', {}).get('mean'), (list, tuple))",
        "                    else (0.5, 0.5, 0.5)).view(3, 1, 1)",
        "std  = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)",
        "# IMAGENET1K_V1 torchvision ViT uses (0.5, 0.5, 0.5) for both mean and std.",
        "img_disp = (x_norm[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()",
        "TARGET_CLASS = top1   # attribute against the top-1 class",
    ),

    # ─── 3. Convention-sharing monkey-patch (used by BOTH paths) ───────────
    md(
        "## 3. Apply `monkey_patch_zennit()` (shared by both paths)",
        "",
        "Patches zennit's `BasicHook.forward` to additionally store",
        "module outputs, and `BasicHook.backward` to operate in the",
        "gradient×input convention (multiply by output going in, divide",
        "by input going out). After this call, EVERY zennit hook in the",
        "process uses that convention.",
        "",
        "Why we share this: zennit's stock relevance-space `Epsilon` rule",
        "compounds badly through ViT residuals without an explicit add-rule",
        "(magnitudes ~1e19). The grad×input convention reformulates the",
        "rule so the per-layer ratio is bounded near unity → numerically",
        "stable on ViT depth.",
    ),
    code(
        "from lxt.efficient import monkey_patch_zennit",
        "monkey_patch_zennit(verbose=True)",
    ),

    # ─── 4. Path A — our zennit canonizers ─────────────────────────────────
    md(
        "## 4. Path A — our zennit canonizers",
        "",
        "Build a `LayerMapComposite` whose layer_map applies",
        "`Epsilon(1e-6)` to Linear / Conv2d and `Pass()` to GELU /",
        "LayerNorm / Dropout. Canonizers install the in-graph forward",
        "rewrites that AttnLRP / LXT requires:",
        "",
        "* `LayerNormForwardCanonizer` — stop-grad on std.",
        "* `GELUIdentityRuleCanonizer` — identity rule on GELU.",
        "* `DropoutPassthroughCanonizer` — disable dropout in backward.",
        "* `TorchvisionMHACPLRPCanonizer` — CP-LRP wrap on every",
        "  `nn.MultiheadAttention` (`query`, `key` detached before",
        "  delegating to the original forward, exactly mirroring",
        "  `lxt.efficient.patches.cp_multi_head_attention_forward`).",
        "",
        "All canonizers are instance-level and revert on `composite.context()`",
        "exit. zennit's `BasicHook` is in grad×input convention (patched",
        "above), so `R_input = data.grad * data`.",
    ),
    code(
        "from zennit.rules import Gamma",
        "from zennit.composites import LayerMapComposite",
        "",
        "from crp.transformer_patches import (",
        "    LayerNormForwardCanonizer,",
        "    GELUIdentityRuleCanonizer,",
        "    DropoutPassthroughCanonizer,",
        "    TorchvisionMHACPLRPCanonizer,",
        ")",
        "",
        "# Reference rules (RationAI/crp-experimenting @ 99045d2):",
        "# Conv2d → Gamma(0.25), Linear → Gamma(0.1). No Pass() entries —",
        "# canonizers (Path A) / LXT class patches (Path B) handle the",
        "# non-Linear/Conv backwards via their patched FORWARDS.",
        "layer_map_a = [",
        "    (nn.Conv2d, Gamma(gamma=0.25)),",
        "    (nn.Linear, Gamma(gamma=0.10)),",
        "]",
        "",
        "canonizers_a = [",
        "    LayerNormForwardCanonizer(),",
        "    # `formula='grad_times_input'` matches LXT's identity-rule",
        "    # kernel (`y/(x+ε)`). Required under monkey_patch_zennit —",
        "    # the relevance-space `y/(y+ε)` over-passes through inactive",
        "    # GELU elements and shatters parity with Path B.",
        "    GELUIdentityRuleCanonizer(epsilon=1e-10, formula='grad_times_input'),",
        "    DropoutPassthroughCanonizer(),",
        "    TorchvisionMHACPLRPCanonizer(),",
        "]",
        "",
        "composite_a = LayerMapComposite(layer_map=layer_map_a, canonizers=canonizers_a)",
        "print(f'composite_a: {len(layer_map_a)} layer_map entries, {len(canonizers_a)} canonizers')",
    ),
    code(
        "# Backward under composite_a. zennit BasicHook is now in",
        "# grad×input convention (monkey_patch_zennit applied in sec. 3);",
        "# recover R_input = data.grad * data.",
        "x_a = x_norm.detach().clone().requires_grad_(True)",
        "with composite_a.context(model):",
        "    out_a = model(x_a)",
        "    out_a[0, TARGET_CLASS].backward()",
        "",
        "R_a = (x_a.grad * x_a).detach().cpu()   # shape (1, 3, 224, 224)",
        "heatmap_a = R_a.sum(dim=1)[0].numpy()   # (224, 224)",
        "logit_a = out_a[0, TARGET_CLASS].item()",
        "print(f'Path A | sum |R| = {R_a.abs().sum().item():.4e}'",
        "      f'   logit = {logit_a:.4f}'",
        "      f'   R-sum / logit = {R_a.sum().item() / logit_a:.4f}')",
    ),

    # ─── 5. Path B — LXT baseline ──────────────────────────────────────────
    md(
        "## 5. Path B — LXT baseline (class-level monkey-patching)",
        "",
        "Apply `lxt.efficient.monkey_patch(vit_torch.vision_transformer,",
        "vit_torch.cp_LRP)` to globally rewrite `nn.GELU`,",
        "`nn.LayerNorm`, and `nn.MultiheadAttention` forwards with LXT's",
        "published vit_torch recipe. The BasicHook patch is already in",
        "place from section 3 (shared with Path A).",
        "",
        "Then run the SAME `Epsilon/Pass` layer_map (NO canonizers —",
        "class-level patches do the work for GELU/LayerNorm/MHA). The",
        "model is unchanged structurally; only per-class `forward` methods",
        "are rewritten.",
        "",
        "Recovery: same as Path A — `R_input = data.grad * data`.",
    ),
    code(
        "from lxt.efficient import monkey_patch",
        "from torchvision.models import vision_transformer",
        "",
        "# `monkey_patch` with no `patch_map` resolves the default for the",
        "# target module — for torchvision.vision_transformer that is",
        "# `lxt.efficient.models.vit_torch.cp_LRP` (GELU + LayerNorm + MHA).",
        "monkey_patch(vision_transformer, verbose=True)",
        "print('LXT vit_torch.cp_LRP class-patches applied (default map)')",
    ),
    code(
        "# Same Gamma layer_map as Path A. NO canonizers — the class-level",
        "# patches from `monkey_patch(vision_transformer)` cover GELU /",
        "# LayerNorm / MultiheadAttention.",
        "layer_map_b = [",
        "    (nn.Conv2d, Gamma(gamma=0.25)),",
        "    (nn.Linear, Gamma(gamma=0.10)),",
        "]",
        "composite_b = LayerMapComposite(layer_map=layer_map_b, canonizers=[])",
        "",
        "x_b = x_norm.detach().clone().requires_grad_(True)",
        "with composite_b.context(model):",
        "    out_b = model(x_b)",
        "    out_b[0, TARGET_CLASS].backward()",
        "",
        "R_b = (x_b.grad * x_b).detach().cpu()",
        "heatmap_b = R_b.sum(dim=1)[0].numpy()",
        "logit_b = out_b[0, TARGET_CLASS].item()",
        "print(f'Path B | sum |R| = {R_b.abs().sum().item():.4e}'",
        "      f'   logit = {logit_b:.4f}'",
        "      f'   R-sum / logit = {R_b.sum().item() / logit_b:.4f}')",
    ),

    # ─── 6. Side-by-side comparison ────────────────────────────────────────
    md(
        "## 6. Compare heatmaps",
        "",
        "Plot input, Path A, Path B, and pixelwise difference. If our",
        "canonizers correctly mimic LXT's recipe, the two heatmaps",
        "should be visually identical and the absolute pixelwise diff",
        "should be a few orders of magnitude below the heatmap range.",
    ),
    code(
        "fig, axes = plt.subplots(1, 4, figsize=(16, 4))",
        "",
        "axes[0].imshow(img_disp); axes[0].axis('off')",
        "axes[0].set_title(f'input — class {TARGET_CLASS}\\n{categories[TARGET_CLASS]}', fontsize=9)",
        "",
        "vmax_a = float(np.abs(heatmap_a).max() or 1.0)",
        "axes[1].imshow(heatmap_a, cmap='seismic', vmin=-vmax_a, vmax=vmax_a)",
        "axes[1].axis('off'); axes[1].set_title('Path A — our canonizers', fontsize=9)",
        "",
        "vmax_b = float(np.abs(heatmap_b).max() or 1.0)",
        "axes[2].imshow(heatmap_b, cmap='seismic', vmin=-vmax_b, vmax=vmax_b)",
        "axes[2].axis('off'); axes[2].set_title('Path B — LXT baseline', fontsize=9)",
        "",
        "diff = heatmap_a - heatmap_b",
        "vmax_d = float(np.abs(diff).max() or 1.0)",
        "axes[3].imshow(diff, cmap='seismic', vmin=-vmax_d, vmax=vmax_d)",
        "axes[3].axis('off'); axes[3].set_title('A − B (pixelwise diff)', fontsize=9)",
        "",
        "plt.tight_layout(); plt.show()",
        "",
        "print(f'  |R_A| max        : {np.abs(heatmap_a).max():.4e}')",
        "print(f'  |R_B| max        : {np.abs(heatmap_b).max():.4e}')",
        "print(f'  |R_A - R_B| max  : {np.abs(diff).max():.4e}')",
        "print(f'  |R_A - R_B| mean : {np.abs(diff).mean():.4e}')",
        "if np.abs(heatmap_a).max() > 0:",
        "    rel = np.abs(diff).max() / np.abs(heatmap_a).max()",
        "    print(f'  relative ∞-error : {rel:.4e}')",
    ),

    # ─── 7. Notes ──────────────────────────────────────────────────────────
    md(
        "## 7. Notes & expected discrepancies",
        "",
        "* **Identity-rule kernel formula.** Our `_IdentityRuleFn` saves",
        "  `output / (output + ε)` on forward and multiplies on backward",
        "  (relevance space). LXT's `identity_rule_implicit_fn` saves",
        "  `output / (input + ε)` (gradient×input space). For active",
        "  GELU inputs these agree to within ε; for near-zero inputs",
        "  they differ. If you see systematic deviation in the diff plot",
        "  concentrated at locations where GELU input crosses zero,",
        "  that's the source.",
        "* **Stabiliser values.** Our canonizer uses `epsilon=1e-6` by",
        "  default. LXT uses `epsilon=1e-10` in its identity-rule kernel",
        "  and `epsilon=1e-6` in its `linear_epsilon`. Tweak to match",
        "  exactly if needed.",
        "* **Bilinear `softmax(QKᵀ) @ V` matmul.** Under CP-LRP the Q,K",
        "  paths are detached, so this matmul gets standard autograd",
        "  backward in BOTH paths. No rule discrepancy.",
        "* **Residual additions / `x + pos_embed`.** Neither path",
        "  applies an LRP rule here — autograd duplicates relevance at",
        "  each `+`. That's LXT's published behaviour. If you add a",
        "  uniform / ratio rule on one side, the diff blows up.",
    ),
]

out = HERE / "lxt_reference.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
