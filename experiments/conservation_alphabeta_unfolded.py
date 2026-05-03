"""Per-layer LRP conservation probe on the unfolded DINOv3 path.

Substitutes all 24 EvaAttention modules with EvaAttentionUnfolded
(rule = AlphaBeta α=0.5/β=0.5, the recommended variant from
RESEARCH_NOTES Entry 6). Runs one attribution and records, at every
hookable nn.Module in the model, the relevance-flow statistics:

* sum(R)
* sum(R+) and sum(R-)  — pos/neg split
* max|R|
* mean|R|
* whether the module is governed by zennit's Pass rule (in which
  case the per-module R_in shown is the natural autograd value, not
  the post-Pass-override value — see RESEARCH_NOTES Entry 4 caveat)

Output:
* Coarse table (per-block summary) — block-level relevance trajectory.
* Fine table (one mid-stack block in detail) — every named submodule
  inside the block, including the unfolded attention's atomic kernels.
* Both tables in markdown, printed in backward order (head → input)
  so reading top-to-bottom traces the LRP backward pass naturally.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config, create_transform

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from crp.attribution import CondAttribution  # noqa: E402
from crp.attention_unfolded import (  # noqa: E402
    EvaAttentionSubstitutionCanonizer,
    EvaAttentionUnfolded,
    BilinearMatmul,
    SoftmaxAlongLastDim,
    ChunkAlongLastDim,
    RotaryEmbedding,
    ScaleByConstant,
    LayerScaleMul,
    ResidualAdd,
    AddBias,
    ReshapeMergeHeads,
)
from crp.transformer_patches import AttnLRPCombinedComposite  # noqa: E402
from datasets import load as load_dataset  # noqa: E402


# Modules whose Pass rule overrides grad_input — per-module probe shows
# natural autograd value, NOT the propagated R. Mark with a flag.
PASS_RULED_TYPES = (nn.LayerNorm, nn.GELU, nn.Dropout, nn.Identity)


class Probe:
    def __init__(self):
        self.records: dict = OrderedDict()
        self._handles = []

    def attach(self, model, names_to_hook):
        for name, module in model.named_modules():
            if name not in names_to_hook:
                continue
            self.records[name] = {
                "module_type": type(module).__name__,
                "is_pass_ruled": isinstance(module, PASS_RULED_TYPES),
            }
            self._handles.append(
                module.register_full_backward_hook(self._make_hook(name))
            )

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name):
        rec = self.records[name]

        def hook(module, grad_input, grad_output):
            # We care about R coming IN (grad_output) and R going OUT
            # (grad_input[0]). For Pass-ruled modules, grad_input is
            # the natural autograd value — see Entry 4 caveat.
            for key, t in [("R_out", grad_output), ("R_in", grad_input)]:
                if not isinstance(t, tuple) or len(t) == 0 or t[0] is None:
                    continue
                tensor = t[0]
                if not isinstance(tensor, torch.Tensor):
                    continue
                if not torch.isfinite(tensor).all():
                    rec[f"{key}_total"] = float("nan")
                    rec[f"{key}_pos"] = float("nan")
                    rec[f"{key}_neg"] = float("nan")
                    rec[f"{key}_max_abs"] = float("nan")
                    rec[f"{key}_mean_abs"] = float("nan")
                    continue
                rec[f"{key}_total"] = tensor.sum().item()
                rec[f"{key}_pos"] = tensor.clamp(min=0).sum().item()
                rec[f"{key}_neg"] = tensor.clamp(max=0).sum().item()
                rec[f"{key}_max_abs"] = tensor.abs().max().item()
                rec[f"{key}_mean_abs"] = tensor.abs().mean().item()

        return hook


def build_model(probe_path: Path, device: str):
    ckpt = torch.load(probe_path, map_location=device, weights_only=False)
    model = timm.create_model(
        ckpt["model_name"], pretrained=True, num_classes=ckpt["num_classes"],
    )
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)
    return model, transform


def pick_correct_sample(model, dataset, device):
    for i in range(0, len(dataset), max(1, len(dataset) // 30)):
        x, y = dataset[i]
        x_dev = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_dev).argmax(-1).item()
        if pred == int(y):
            return x_dev.detach().requires_grad_(True), pred
    raise RuntimeError("no correct sample")


def run_with_probe(model, composite, x, target, names):
    probe = Probe()
    with composite.context(model) as modified:
        probe.attach(modified, names)
        try:
            x.grad = None
            out = modified(x)
            R0 = torch.zeros_like(out)
            R0[0, target] = out[0, target].detach()
            modified.zero_grad()
            out.backward(R0)
        finally:
            probe.detach()
    return x.grad, probe.records, R0[0, target].item()


def fmt(x, width=12):
    if x is None:
        return f"{'—':>{width}}"
    if isinstance(x, float) and (x != x):
        return f"{'NaN':>{width}}"
    if isinstance(x, float):
        if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
            return f"{x:>{width}.2e}"
        return f"{x:>{width}.3f}"
    return f"{x:>{width}}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_path = REPO_ROOT / "data/vit_large_patch16_dinov3_probe_imagenette.pt"
    out_path = REPO_ROOT / "tutorials/vit_crp/dinov3_variants/alphabeta/CONSERVATION.md"

    model, transform = build_model(probe_path, device)
    dataset = load_dataset("imagenette", split="val", transform=transform)
    x, target = pick_correct_sample(model, dataset, device)
    n_blocks = len(model.blocks)

    # --- Coarse hookset: block boundaries + framing modules.
    coarse_names = {"head", "norm", "patch_embed.proj"}
    for i in range(n_blocks):
        coarse_names |= {f"blocks.{i}"}

    # --- Fine hookset: every named module in one mid-stack block plus the
    # block-12 framing on either side. Add ALL modules from the substituted
    # block 12; block name will become e.g. blocks.12.attn.qkv_scores.
    target_block = 12

    # Use AlphaBeta α=0.5/β=0.5 — the recommended magnitude-controlled variant.
    composite = AttnLRPCombinedComposite(
        layerscale_uniform=True, residual_lrp="ratio",
    )
    sub = EvaAttentionSubstitutionCanonizer(
        block_indices=tuple(range(n_blocks)),
        matmul_rule="alpha_beta", alpha=0.5, beta=0.5,
    )
    composite.canonizers = list(composite.canonizers) + [sub]

    # First pass: discover the unfolded module's submodule names by entering
    # the context once.
    with composite.context(model) as modified:
        unfolded_attn = modified.blocks[target_block].attn
        # Hook every direct + named-descendant submodule of the unfolded attention.
        # (named_modules with prefix.)
        attn_submodule_names = [
            f"blocks.{target_block}.attn.{n}"
            for n, _ in unfolded_attn.named_modules() if n != ""
        ]
    fine_names = set(coarse_names)
    fine_names |= set(attn_submodule_names)
    # Also add the MLP innards.
    fine_names |= {
        f"blocks.{target_block}.norm1",
        f"blocks.{target_block}.norm2",
        f"blocks.{target_block}.mlp",
        f"blocks.{target_block}.mlp.fc1",
        f"blocks.{target_block}.mlp.fc2",
        f"blocks.{target_block}.mlp.act",
        f"blocks.{target_block}.mlp.drop1",
        f"blocks.{target_block}.mlp.drop2",
    }

    # --- Run the attribution with the probe.
    grad, records, R0_total = run_with_probe(model, composite, x, target, fine_names)

    # ======================================================================
    # Report
    # ======================================================================

    lines = []
    lines.append("# Per-layer LRP conservation probe — DINOv3 ViT-L/16 + AlphaBeta(0.5,0.5)")
    lines.append("")
    lines.append("Generated by `experiments/conservation_alphabeta_unfolded.py`.")
    lines.append(f"Sample: class {target}.")
    lines.append(f"Initial relevance (= target logit): **{R0_total:.4f}**.")
    lines.append(f"Final input sum(R): **{grad.sum().item():.4f}**.")
    lines.append(f"Input pos/neg split: **+{grad.clamp(min=0).sum().item():.3e} / "
                 f"{grad.clamp(max=0).sum().item():.3e}**.")
    lines.append(f"Input max|R|: **{grad.abs().max().item():.3e}**.")
    lines.append("")
    lines.append("Reading: each table is in **backward order** (top = closest to head, "
                 "bottom = closest to input pixels), so reading top-to-bottom traces "
                 "the LRP backward pass.")
    lines.append("")
    lines.append("`R_out` = relevance entering the module from above (towards head). "
                 "`R_in` = relevance leaving the module towards the input. "
                 "`absorbed` = `R_out − R_in` (positive = bias absorption, negative = "
                 "magnitude inflation in this module).")
    lines.append("")
    lines.append("⚠️ Pass-ruled modules (LayerNorm, GELU, Dropout, Identity) show "
                 "the **natural autograd** `R_in`, not the post-Pass-override "
                 "value — Entry 4 caveat. Pass-ruled rows are tagged `[Pass]`.")
    lines.append("")

    # --- Table 1: per-block trajectory (coarse).
    lines.append("## Coarse: per-block trajectory")
    lines.append("")
    lines.append("Block-by-block summary in backward order. Tracks how relevance flows "
                 "through each EvaBlock as a unit.")
    lines.append("")
    lines.append("| step | layer | type | R_out | R_in | absorbed | sum(R+) | sum(R−) | max\\|R_in\\| |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    walk_order = ["head", "norm"] + [f"blocks.{i}" for i in range(n_blocks - 1, -1, -1)] + ["patch_embed.proj"]
    step = 0
    for name in walk_order:
        if name not in records:
            continue
        rec = records[name]
        ro = rec.get("R_out_total")
        ri = rec.get("R_in_total")
        if ro is None and ri is None:
            continue
        absorbed = (ro - ri) if (ro is not None and ri is not None and not (isinstance(ro, float) and ro != ro)) else None
        pass_tag = " [Pass]" if rec["is_pass_ruled"] else ""
        step += 1
        lines.append(
            f"| {step} | `{name}`{pass_tag} | {rec['module_type']} | "
            f"{fmt(ro)} | {fmt(ri)} | {fmt(absorbed)} | "
            f"{fmt(rec.get('R_in_pos'))} | {fmt(rec.get('R_in_neg'))} | "
            f"{fmt(rec.get('R_in_max_abs'))} |"
        )
    lines.append("")

    # --- Table 2: fine — block 12 internals.
    lines.append(f"## Fine: block {target_block} internals")
    lines.append("")
    lines.append(f"Every named submodule inside `blocks.{target_block}`, in backward order. "
                 f"Shows the relevance trajectory through one EvaBlock at the granularity "
                 f"the unfolded refactor exposes (Q/K/V split, RoPE, scale, qk_scores, "
                 f"softmax, context, projection).")
    lines.append("")
    lines.append("| step | layer | type | R_out | R_in | absorbed | sum(R+) | sum(R−) | max\\|R_in\\| |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    # Walk ONLY block target_block and its submodules in REVERSE module-tree order.
    # Use the actual hooked names that start with f"blocks.{target_block}".
    block_names = [
        n for n in records.keys()
        if n.startswith(f"blocks.{target_block}") or n == f"blocks.{target_block}"
    ]
    # Sort by descent depth (shallowest first); then within same prefix, leaf order
    # matches forward order. We want backward order = reverse of forward processing
    # within the block. The block's own hook fires LAST in forward + FIRST in backward.
    # The deepest sub-leaves fire LAST in backward (closest to block input).
    # So sort: block itself first, then submodules in reverse-forward order.
    # Heuristic: reverse alphabetical of (module_path), with block-self at top.
    # To get forward order of submodules, use the order they appear in records (named_modules
    # iteration is forward order). Reverse for backward.
    # Easy hack: iterate records in insertion order; reverse only the submodules,
    # keep block-itself at top.
    block_self = f"blocks.{target_block}"
    sub_names_in_fwd_order = [n for n in block_names if n != block_self]
    sub_names_bwd = list(reversed(sub_names_in_fwd_order))
    walk = [block_self] + sub_names_bwd if block_self in records else sub_names_bwd
    step = 0
    for name in walk:
        if name not in records:
            continue
        rec = records[name]
        ro = rec.get("R_out_total")
        ri = rec.get("R_in_total")
        if ro is None and ri is None:
            continue
        absorbed = (ro - ri) if (ro is not None and ri is not None and not (isinstance(ro, float) and ro != ro)) else None
        pass_tag = " [Pass]" if rec["is_pass_ruled"] else ""
        step += 1
        # Strip the block prefix for readability inside this table.
        short = name.replace(f"blocks.{target_block}.", "").replace(f"blocks.{target_block}", "(block)")
        lines.append(
            f"| {step} | `{short}`{pass_tag} | {rec['module_type']} | "
            f"{fmt(ro)} | {fmt(ri)} | {fmt(absorbed)} | "
            f"{fmt(rec.get('R_in_pos'))} | {fmt(rec.get('R_in_neg'))} | "
            f"{fmt(rec.get('R_in_max_abs'))} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Per-block factor analysis** (coarse table). For each block, "
                 "`R_in / R_out` is the per-block magnitude factor: factor > 1 means "
                 "the block inflated R, factor < 1 means it deflated. Compare to the "
                 "baseline 2Y+ε per-block factors of 0.21× to 9.15× recorded in "
                 "`RESEARCH_NOTES.md` Entry 4.")
    lines.append("")
    lines.append("| block | R_in | R_out | factor (R_in/R_out) |")
    lines.append("|---|---:|---:|---:|")
    for i in range(n_blocks - 1, -1, -1):
        rec = records[f"blocks.{i}"]
        ri = rec.get("R_in_total")
        ro = rec.get("R_out_total")
        factor = (ri / ro) if (ri is not None and ro is not None and abs(ro) > 1e-9) else float("nan")
        lines.append(f"| `blocks.{i}` | {fmt(ri)} | {fmt(ro)} | {fmt(factor)} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n--- preview (first 60 lines) ---")
    for line in lines[:60]:
        print(line)


if __name__ == "__main__":
    main()
