"""Competing saliency methods for ViTs (non-CRP baselines for the register study
+ gallery). Shared by the scripts and the gallery.

All return a ``grid × grid`` patch map (non-negative; LRP signed-|·|):

* :func:`attention_rollout` — Abnar & Zuidema; class-agnostic CLS→patch row.
* :func:`chefer_relevance` — Chefer/Gur/Wolf ICCV'21 generic-attention rollout
  (raw attention · gradient), class-conditional.
* :func:`chefer_transformer_attribution` — Chefer CVPR'21 code-exact
  (LRP relevance ``R_A`` · gradient), class-conditional.
* :func:`occlusion_deltap` — per-patch mean-fill Δp⁺.
* :func:`lrp_patch` — LRP/CRP baseline, patch-aggregated.

Geometry (prefix tokens, patch grid) read from the model at call time: standard
ViTs (1 CLS, 14×14) and DINOv3 with registers (1 CLS + 4 reg, 16×16); patch
stats + CLS-row selector skip all prefix tokens. Raw maps unnormalised; display
in :func:`render_patch_map`.
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
    """``(n_prefix, grid, patch)`` for a timm-backboned ViT and input ``x``.
    ``n_prefix`` = non-patch tokens (``backbone.num_prefix_tokens``); ``grid`` =
    patches/side; ``patch`` = patch side in px."""
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


def to_patch_max(pixel_map: torch.Tensor, grid: int, patch: int) -> torch.Tensor:
    """``(H, W)`` pixel saliency → ``(grid, grid)`` **signed max** per patch.
    MAX aggregation for Insertion-Deletion (patch = its most-salient pixel).
    Signed (no ``abs``) so an all-negative patch sorts last."""
    return pixel_map.reshape(grid, patch, grid, patch).amax(dim=(1, 3))


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
    """Forward ``xn`` with ``fused_attn`` off + a forward hook on each block's
    ``attn.attn_drop``; return ``(logits, attns)``, ``attns[b]`` the post-softmax
    attention ``(B, heads, T, T)``. ``keep_graph=True`` keeps the graph (for
    grad-w.r.t.-attention); else ``no_grad`` + detach."""
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
    """Abnar & Zuidema attention rollout → ``(B, grid, grid)``. Per block:
    head-average, ``0.5·A + 0.5·I`` (skip), row-normalise, chain by matmul; read
    CLS→patch row. Class-agnostic."""
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
    """Chefer/Gur/Wolf **ICCV'21** generic-attention rollout → ``(B, grid, grid)``,
    class-conditional. Per block ``cam = mean_heads((∂logit_t/∂A ⊙ A)⁺)``,
    ``Ā = I + cam`` row-normalised, chained; CLS→patch row. Weights the gradient
    by **raw attention** ``A`` — NOT the CVPR'21 rule (which uses LRP relevance
    ``R_A``; see :func:`chefer_transformer_attribution`)."""
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


def softmax_layer_names(model) -> List[str]:
    """Per-block softmax layer names ``backbone.blocks.{b}.attn.softmax`` (exist
    only inside the canonized attribution context). Pass as ``record_layer`` to
    tap the post-softmax relevance ``R_A`` (Chefer's ``get_attn_cam``)."""
    return [f"backbone.blocks.{b}.attn.softmax"
            for b in range(len(model.backbone.blocks))]


def chefer_transformer_attribution(model, attribution, composite, xn: torch.Tensor,
                                   target: int, *, n_prefix: int, grid: int,
                                   softmax_layers: List[str],
                                   return_blocks: bool = False):
    """Chefer et al. **CVPR'21** Transformer Attribution → ``(1, grid, grid)``,
    class-conditional. Code-exact reproduction of ``generate_LRP(method=
    "transformer_attribution")`` (Transformer-Explainability, ViT_LRP.py, commit
    c3e578f). One image at a time. Validated pearson≈1.0 vs the reference in
    ``tutorials/vit_crp/chefer_reference.ipynb``. Algorithm::

        for blk: cam_b = mean_heads((G_b ⊙ R_A_b)⁺)      # G = ∂logit/∂A, R_A = LRP rel
        rollout = ∏_b (I + cam_b)                          # NO row-norm (code-exact)
        return rollout[0, 1:]                              # CLS→patch row

    Two ingredients, no repo vendoring:

    * ``G`` — clean autograd gradient of the target logit w.r.t. each block's
      post-softmax attention (:func:`capture_attention` + one ``autograd.grad``).
    * ``R_A`` — attention-map LRP relevance, **recorded** at each block's
      ``attn.softmax`` under ``composite`` (an attention-conducting recipe, e.g.
      ``chefer_lrp``) — the analogue of the authors' ``get_attn_cam``. The
      relevance is *read* here (an intermediate tap), so nothing below the
      softmax (Q/K path) enters the map.

    Both ``(1, heads, T, T)``, head-aligned. No row-norm matches the released
    ``compute_rollout_attention`` (norm lines commented out, ViT_LRP.py:44-45).
    ``return_blocks=True`` also yields per-block ``(G, R_A, cam_b)`` for audit.
    """
    if xn.shape[0] != 1:
        raise ValueError("chefer_transformer_attribution runs one image at a time")
    # (1) clean autograd gradient of the target logit w.r.t. each block's attn.
    xg = xn.detach().clone().requires_grad_(True)
    logits, attns = capture_attention(model, xg, keep_graph=True)   # attns[b] (1,h,T,T)
    logit = logits[0, int(target)]
    grads = torch.autograd.grad(logit, attns)                       # tuple of (1,h,T,T)
    # (2) attention-map LRP relevance R_A at each softmax (composite).
    xa = xn.detach().clone().requires_grad_(True)
    res = attribution(xa, [{"y": [int(target)]}], composite,
                      record_layer=list(softmax_layers), init_rel=1)
    n_tok = attns[0].shape[-1]
    eye = torch.eye(n_tok, device=xn.device).unsqueeze(0)
    r = None
    blocks_out = [] if return_blocks else None
    for b, ln in enumerate(softmax_layers):
        g = grads[b]                                                # (1,h,T,T)
        cam = res.relevances[ln]                                    # (1,h,T,T)  R_A
        c = (g * cam).clamp(min=0).mean(dim=1)                      # (1,T,T)
        ab = eye + c                                                # NO row-norm (code-exact)
        r = ab if r is None else ab @ r
        if return_blocks:
            blocks_out.append((g.detach(), cam.detach(), c.detach()))
    result = _cls_row_to_grid(r[:, 0], n_prefix, grid)
    if return_blocks:
        return result, blocks_out
    return result


def rise_saliency(model, normalize, x: torch.Tensor, target: int, *, input_size: int,
                  n_masks: int = 2000, s: int = 8, p: float = 0.5, batch: int = 128,
                  seed: int = 0) -> torch.Tensor:
    """RISE (Petsiuk et al. BMVC'18) → ``(H, W)`` pixel saliency for ``target``.
    ``n_masks`` binary ``s×s`` Bernoulli(p) grids, bilinear-upsampled to
    ``(s+1)·cell`` (``cell=ceil(H/s)``), random-cropped to ``H×H``. Saliency
    ``= (1/(N·p)) Σ_i f_target(x ⊙ mask_i) · mask_i``, ``f`` = softmax prob.
    Zero-fill masking (not the benchmark's mean-fill occlusion). ``x`` = single
    un-normalised ``(3,H,W)`` in [0,1]."""
    import torch.nn.functional as F
    device = next(model.parameters()).device
    x = x.to(device)
    g = torch.Generator().manual_seed(seed)                         # CPU generator
    cell = int(math.ceil(input_size / s))
    up = (s + 1) * cell
    grid = (torch.rand(n_masks, 1, s, s, generator=g) < p).float()
    big = F.interpolate(grid, size=(up, up), mode="bilinear", align_corners=False)
    masks = torch.empty(n_masks, 1, input_size, input_size)
    shifts = torch.randint(0, cell, (n_masks, 2), generator=g)
    for i in range(n_masks):
        ox, oy = int(shifts[i, 0]), int(shifts[i, 1])
        masks[i] = big[i, :, ox:ox + input_size, oy:oy + input_size]
    masks = masks.to(device)
    sal = torch.zeros(input_size, input_size, device=device)
    with torch.no_grad():
        for s0 in range(0, n_masks, batch):
            m = masks[s0:s0 + batch]                                # (b,1,H,W)
            probs = model(normalize(m * x[None])).softmax(-1)[:, int(target)]
            sal += (probs[:, None, None] * m[:, 0]).sum(0)
    return (sal / (n_masks * p)).cpu()


def occlusion_deltap(model, normalize, x: torch.Tensor, target: int, *,
                     grid: int, patch: int, batch: int = 64) -> torch.Tensor:
    """Occlusion Δp⁺ → ``(grid, grid)``: per patch, mean-colour fill, measure the
    drop in target-class softmax prob (clamped ≥0). ``grid²`` forwards, one image.
    ``x`` = un-normalised ``(3,H,W)`` in [0,1]; ``normalize`` applied pre-forward."""
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
    """LRP/CRP baseline, patch-aggregated (sum |R| per patch) → ``(grid, grid)``.
    ``xn`` = normalised input; relevance seeded at the target logit."""
    xin = xn.clone().detach().requires_grad_(True)
    res = attribution(xin, [{"y": [int(target)]}], composite)
    heat = res.heatmap.detach().cpu()[0]                    # (H, W) signed
    return to_patch_grid(heat, grid, patch)


# ─────────────────────────────────────────────────────────────────────────────
# Display: patch map → PNG (upsample to input res, percentile-clip, colormap)
# ─────────────────────────────────────────────────────────────────────────────

def render_patch_map(patch_map: torch.Tensor, out_path, *, res: int = 224,
                     cmap: str = "inferno", clip: float = 0.99) -> None:
    """Save ``(grid, grid)`` non-negative map as PNG: nearest-upsample to ``res``²,
    clip to the ``clip`` quantile of positive values (one hot patch won't wash it
    out), min-max to [0,1], apply ``cmap``."""
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
    "model_geometry", "to_patch_grid", "to_patch_max", "saliency_flags",
    "capture_attention", "attention_rollout", "chefer_relevance",
    "chefer_transformer_attribution", "softmax_layer_names", "rise_saliency",
    "occlusion_deltap", "lrp_patch", "render_patch_map",
]
