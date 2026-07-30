"""Competing input-attribution saliency methods for ViTs (the non-CRP baselines).

Single home for the saliency methods compared against LRP/CRP in the register
study and the CRP gallery — factored out of ``registers_e2_overlap.py`` /
``registers_saliency_compare.py`` so the scripts and the gallery share ONE
implementation instead of re-deriving the same primitives:

* :func:`attention_rollout` — Abnar & Zuidema rollout (class-agnostic; the CLS
  row over patches).
* :func:`chefer_relevance` — Chefer/Gur/Wolf (CVPR'21) grad-weighted rollout,
  class-conditional on a target logit.
* :func:`occlusion_deltap` — occlusion Δp⁺: per-patch drop in the true/target
  class probability under a mean-colour patch mask.
* :func:`lrp_patch` — the LRP/CRP baseline map, patch-aggregated the same way,
  so all four sit in one comparable row.

All methods return a **patch-grid** map (``grid × grid``, non-negative for the
three competing methods; signed |·|-aggregated for LRP) so they are directly
comparable. The geometry (prefix-token count, patch grid) is read from the model
at call time, so the same code serves the standard ViTs (M1/M2: 1 CLS prefix,
224px → 14×14) and the register-bearing DINOv3 models (M3/M4: 1 CLS + 4 register
= 5 prefix, 256px → 16×16). Patch statistics and the CLS-row selector always
skip **all** prefix tokens (CLS *and* registers).

Raw maps are returned unnormalised; display normalisation (percentile clip +
colormap) lives in :func:`render_patch_map`, used by the gallery.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SD_K = 4.0  # μ + k·σ hot-patch threshold (per-sample, over that sample's patches)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — prefix tokens (CLS + registers) and patch grid, read from the model
# ─────────────────────────────────────────────────────────────────────────────

def model_geometry(model, x: torch.Tensor) -> Tuple[int, int, int]:
    """``(n_prefix, grid, patch)`` for a timm-backboned ViT and an input ``x``.

    * ``n_prefix`` — number of non-patch tokens (1 for a plain ViT CLS; 5 for
      DINOv3 = 1 CLS + 4 registers), from ``backbone.num_prefix_tokens``.
    * ``grid`` — patches per side (14 at 224px/16, 16 at 256px/16).
    * ``patch`` — patch side in pixels.
    """
    backbone = model.backbone
    n_prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    ps = backbone.patch_embed.patch_size
    patch = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
    grid = int(x.shape[-1]) // patch
    return n_prefix, grid, patch


def to_patch_grid(pixel_map: torch.Tensor, grid: int, patch: int) -> torch.Tensor:
    """``(H, W)`` pixel saliency → ``(grid, grid)`` sum of |values| per patch."""
    pm = pixel_map.abs().reshape(grid, patch, grid, patch).sum(dim=(1, 3))
    return pm


def saliency_flags(patch_map: np.ndarray, k: float = SD_K) -> np.ndarray:
    """Boolean hot-patch mask: ``value > mean + k·sd`` over the map's own patches
    (the per-sample μ+kσ rule reused across every method / OOD detection)."""
    flat = patch_map.reshape(-1)
    return (patch_map > flat.mean() + k * flat.std())


# ─────────────────────────────────────────────────────────────────────────────
# Attention capture (fused_attn OFF, hook attn_drop) — shared by rollout/chefer
# ─────────────────────────────────────────────────────────────────────────────

def capture_attention(model, xn: torch.Tensor, *, keep_graph: bool = False,
                      ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Forward ``xn`` with ``fused_attn`` disabled and a forward hook on each
    block's ``attn.attn_drop``; return ``(logits, attns)`` where ``attns[b]`` is
    the post-softmax attention ``(B, heads, T, T)`` of block ``b``.

    ``keep_graph=True`` keeps the autograd graph on the captured tensors (needed
    by :func:`chefer_relevance`, which differentiates the target logit w.r.t.
    them); otherwise capture runs under ``no_grad`` and detaches (rollout)."""
    blocks = model.backbone.blocks
    store: Dict[int, torch.Tensor] = {}
    hooks, prev = [], []
    for b, blk in enumerate(blocks):
        prev.append(blk.attn.fused_attn)
        blk.attn.fused_attn = False

        def f(m, i, o, b=b):
            store[b] = o if keep_graph else o.detach()
        hooks.append(blk.attn.attn_drop.register_forward_hook(f))
    try:
        if keep_graph:
            logits = model(xn)
        else:
            with torch.no_grad():
                logits = model(xn)
    finally:
        for h in hooks:
            h.remove()
        for blk, p in zip(blocks, prev):
            blk.attn.fused_attn = p
    return logits, [store[b] for b in range(len(blocks))]


def _cls_row_to_grid(row: torch.Tensor, n_prefix: int, grid: int) -> torch.Tensor:
    """CLS→patch attention row ``(B, T)`` → ``(B, grid, grid)`` (drop all prefix
    columns, then reshape the patch columns row-major)."""
    patch = row[:, n_prefix:n_prefix + grid * grid]
    return patch.reshape(-1, grid, grid)


def attention_rollout(attns: List[torch.Tensor], n_prefix: int, grid: int,
                      ) -> torch.Tensor:
    """Abnar & Zuidema attention rollout → ``(B, grid, grid)``.

    Per block: head-average, add identity for the skip connection
    (``0.5·A + 0.5·I``), row-normalise, chain by matmul; read the CLS row over
    the patch tokens. Class-agnostic (raw attention, no target)."""
    r = None
    for a in attns:
        a = a.mean(dim=1)                                   # (B, T, T)
        eye = torch.eye(a.shape[-1], device=a.device).unsqueeze(0)
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True)
        r = a if r is None else a @ r
    return _cls_row_to_grid(r[:, 0], n_prefix, grid)


def chefer_relevance(model, xn: torch.Tensor, targets: List[int], *,
                     n_prefix: int, grid: int) -> torch.Tensor:
    """Chefer/Gur/Wolf (CVPR'21) grad-weighted attention rollout → ``(B, grid,
    grid)``, class-conditional on ``targets``.

    Per block: ``cam = mean_heads((∂logit_t/∂A ⊙ A)⁺)``, ``Ā = I + cam``
    row-normalised, chained by matmul; read the CLS row over patches. The target
    logit is summed over the batch so one backward yields every block's grad."""
    # The backbone params are frozen, so the autograd graph exists only if the
    # INPUT requires grad — set it here (mirrors the register-study scripts).
    xn = xn.detach().clone().requires_grad_(True)
    logits, attns = capture_attention(model, xn, keep_graph=True)
    tgt = torch.as_tensor(targets, device=logits.device)
    logit = logits[torch.arange(len(targets), device=logits.device), tgt].sum()
    grads = torch.autograd.grad(logit, attns)
    n_tok = attns[0].shape[-1]
    eye = torch.eye(n_tok, device=xn.device).unsqueeze(0)
    r = None
    for a, g in zip(attns, grads):
        a = a.detach()
        cam = (g * a).clamp(min=0).mean(dim=1)              # (B, T, T)
        ab = eye + cam
        ab = ab / ab.sum(dim=-1, keepdim=True)
        r = ab if r is None else ab @ r
    return _cls_row_to_grid(r[:, 0], n_prefix, grid)


def occlusion_deltap(model, normalize, x: torch.Tensor, target: int, *,
                     grid: int, patch: int, batch: int = 64) -> torch.Tensor:
    """Occlusion Δp⁺ → ``(grid, grid)``: for each patch, replace its pixels with
    the per-image mean colour and measure the drop in the target-class softmax
    probability, clamped at 0. One image at a time (``grid²`` forwards).

    ``x`` is a single un-normalised ``(3, H, W)`` image in [0,1]; ``normalize``
    is applied before the forward (the model's boundary normalize)."""
    device = next(model.parameters()).device
    x = x.to(device)
    n_patch = grid * grid
    with torch.no_grad():
        p_clean = model(normalize(x[None])).softmax(-1)[0, int(target)]
        mean_col = x.mean(dim=(1, 2))
        occ = x[None].repeat(n_patch, 1, 1, 1)
        for p in range(n_patch):
            r, c = divmod(p, grid)
            occ[p, :, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = \
                mean_col[:, None, None]
        probs = torch.cat([model(normalize(occ[s:s + batch])).softmax(-1)[:, int(target)]
                           for s in range(0, n_patch, batch)])
    return (p_clean - probs).clamp(min=0).reshape(grid, grid).cpu()


def lrp_patch(attribution, xn: torch.Tensor, target: int, composite, *,
              grid: int, patch: int) -> torch.Tensor:
    """LRP/CRP baseline map, patch-aggregated (sum |R| per patch) → ``(grid,
    grid)``. ``xn`` is the normalised input (``requires_grad_`` set here);
    relevance is initialised at the target logit (class-conditional)."""
    xin = xn.clone().detach().requires_grad_(True)
    res = attribution(xin, [{"y": [int(target)]}], composite)
    heat = res.heatmap.detach().cpu()[0]                    # (H, W) signed
    return to_patch_grid(heat, grid, patch)


# ─────────────────────────────────────────────────────────────────────────────
# Display: patch map → PNG (upsample to input res, percentile-clip, colormap)
# ─────────────────────────────────────────────────────────────────────────────

def render_patch_map(patch_map: torch.Tensor, out_path, *, res: int = 224,
                     cmap: str = "inferno", clip: float = 0.99) -> None:
    """Save a ``(grid, grid)`` non-negative saliency map as a PNG: nearest-upsample
    to ``res``², clip to the ``clip`` quantile of positive values (so a single hot
    patch does not wash the map out), min-max to [0,1], apply ``cmap``.

    Same normalisation *spirit* as the LRP sample heat (percentile clip so the
    structure is visible), adapted to non-negative patch maps."""
    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path
    from PIL import Image

    pm = patch_map.detach().cpu().float().numpy()
    grid = pm.shape[0]
    up = np.kron(pm, np.ones((res // grid, res // grid), dtype=pm.dtype))
    pos = up[up > 0]
    vmax = float(np.quantile(pos, clip)) if pos.size else float(up.max())
    if vmax <= 0:
        vmax = float(up.max()) or 1.0
    norm = np.clip(up, 0, vmax) / vmax
    rgb = (matplotlib.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)


# Provenance captions shown under each method in the web (kept next to the code
# that produces them so they stay truthful).
METHOD_CAPTIONS: Dict[str, str] = {
    "lrp": "LRP/CRP baseline (cp_lrp_baseline), value-path AttnLRP; input relevance to "
           "the predicted class, |R| summed per 16×16 patch.",
    "chefer": "Chefer et al. CVPR'21 — per block I + mean_heads((∂logit/∂A · A)⁺), "
              "row-normalised & chained; CLS→patch row; gradient of the predicted-class logit.",
    "rollout": "Abnar & Zuidema attention rollout — per block row-normalised 0.5A+0.5I, "
               "chained; CLS→patch row. Class-agnostic (raw attention, no gradient).",
    "occlusion": "Occlusion Δp⁺ — per patch masked with the image mean colour; positive drop "
                 "in the predicted-class softmax probability (grid² forwards).",
}
METHODS: Tuple[str, ...] = ("lrp", "chefer", "rollout", "occlusion")


__all__ = [
    "SD_K", "METHODS", "METHOD_CAPTIONS",
    "model_geometry", "to_patch_grid", "saliency_flags", "capture_attention",
    "attention_rollout", "chefer_relevance", "occlusion_deltap", "lrp_patch",
    "render_patch_map",
]
