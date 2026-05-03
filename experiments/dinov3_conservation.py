"""Per-module conservation probe for DINOv3 + the working AttnLRP composite.

Runs one attribution and records, at every chosen module:

* ``sum(R_in)``, ``sum(R_in_pos)``, ``sum(R_in_neg)`` — total / positive /
  negative relevance leaving the module (going down the LRP graph).
* ``sum(R_out)``, ``sum(R_out_pos)``, ``sum(R_out_neg)`` — same but
  entering the module from upstream (coming down from the head).
* ``absorbed = sum(R_out) - sum(R_in)`` — what didn't pass through. For a
  Linear ``y = Wx + b`` with ε-LRP, the implied bias-absorbed relevance
  is the difference, since R_in receives only ``x · ∇y / stab(y) · R_y``
  (the bias contribution ``b · R_y / stab(y)`` is "discharged" to the
  bias node — Bach et al. 2015; Montavon et al. 2019).

The point: if the ``max|R| ≈ 200`` we observed at the input is a real
positive blob (not a positive blob cancelled by a comparable negative
blob) AND the per-layer trajectory shows monotonic deflation consistent
with bias absorption, we have evidence the working composite is
behaving like proper LRP (not a numerical accident).

Usage::

    uv run python experiments/dinov3_conservation.py
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import timm
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from crp.transformer_patches import (  # noqa: E402
    AttnLRPCombinedComposite,
    AttnLRPEpsilonComposite,
)
from datasets import load as load_dataset  # noqa: E402

from timm.data import resolve_data_config, create_transform  # noqa: E402


class ConservationProbe:
    """Hooks every chosen module so backward records ``sum(R)`` going in
    and out, plus positive/negative split. Lifecycle: ``attach()`` after
    the composite context is open, ``detach()`` before exiting."""

    def __init__(self):
        self.records: dict[str, dict] = OrderedDict()
        self._handles: list = []

    def attach(self, model: nn.Module, names: set[str]):
        for name, module in model.named_modules():
            if name not in names:
                continue
            self.records[name] = {"module_type": type(module).__name__}
            self._handles.append(module.register_forward_hook(self._fhook(name)))
            self._handles.append(module.register_full_backward_hook(self._bhook(name)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _fhook(self, name):
        rec = self.records[name]

        def hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            y = output[0] if isinstance(output, tuple) else output
            if isinstance(y, torch.Tensor):
                rec["x_abs_max"] = x.abs().max().item() if isinstance(x, torch.Tensor) else None
                rec["y_abs_max"] = y.abs().max().item()

        return hook

    def _bhook(self, name):
        rec = self.records[name]

        def hook(module, grad_input, grad_output):
            # grad_output: relevance flowing INTO the module (from above)
            r_out = grad_output[0] if isinstance(grad_output, tuple) and grad_output[0] is not None else None
            # grad_input: relevance flowing OUT of the module (to below)
            r_in = grad_input[0] if isinstance(grad_input, tuple) and len(grad_input) > 0 and grad_input[0] is not None else None
            for key, t in [("R_out", r_out), ("R_in", r_in)]:
                if t is None or not isinstance(t, torch.Tensor):
                    continue
                finite = torch.isfinite(t)
                if not finite.all():
                    rec[f"{key}_total"] = float("nan")
                    rec[f"{key}_pos"] = float("nan")
                    rec[f"{key}_neg"] = float("nan")
                    rec[f"{key}_abs_max"] = float("nan")
                    rec[f"{key}_finite_share"] = finite.float().mean().item()
                    continue
                rec[f"{key}_total"] = t.sum().item()
                rec[f"{key}_pos"] = t.clamp(min=0).sum().item()
                rec[f"{key}_neg"] = t.clamp(max=0).sum().item()
                rec[f"{key}_abs_max"] = t.abs().max().item()
                rec[f"{key}_finite_share"] = 1.0
            if "R_in_total" in rec and "R_out_total" in rec:
                rec["absorbed"] = rec["R_out_total"] - rec["R_in_total"]

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


def pick_one_correct_sample(model, dataset, device):
    for i in range(0, len(dataset), max(1, len(dataset) // 30)):
        x, y = dataset[i]
        x_dev = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x_dev).argmax(-1).item()
        if pred == int(y):
            return x_dev.detach().requires_grad_(True), pred
    raise RuntimeError("no correct sample found")


def attribute_with_probe(model, composite, x, target, names):
    probe = ConservationProbe()
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_path = REPO_ROOT / "data/vit_large_patch16_dinov3_probe_imagenette.pt"
    print(f"device: {device}")

    model, transform = build_model(probe_path, device)
    print(f"model : vit_large_patch16_dinov3 ({len(model.blocks)} blocks)")

    dataset = load_dataset("imagenette", split="val", transform=transform)
    x, target = pick_one_correct_sample(model, dataset, device)
    print(f"sample: class {target}, image shape {tuple(x.shape)}\n")

    # Pick the boundary modules to instrument:
    #   model.head            — final Linear (logit ← cls/avg-pool)
    #   model.norm            — final LayerNorm
    #   blocks.{i}            — each EvaBlock (whole-block conservation)
    #   blocks.{i}.attn.proj  — attention output Linear
    #   blocks.{i}.mlp.fc1, fc2 — MLP linears
    #   blocks.{i}.norm1, norm2 — pre-attn / pre-mlp LayerNorm
    #   patch_embed.proj      — input Conv2d
    names = {"head", "norm", "patch_embed.proj"}
    for i in range(len(model.blocks)):
        names |= {
            f"blocks.{i}",
            f"blocks.{i}.norm1",
            f"blocks.{i}.norm2",
            f"blocks.{i}.attn.proj",
            f"blocks.{i}.mlp.fc1",
            f"blocks.{i}.mlp.fc2",
        }

    print(f"running attribution under working composite (matmul + layerscale + ratio)")
    composite = AttnLRPCombinedComposite(
        matmul_factor_2=True,
        layerscale_uniform=True,
        residual_lrp="ratio",
    )
    x_run = x.detach().clone()
    x_run.requires_grad_(True)
    grad, records, R0_total = attribute_with_probe(
        model, composite, x_run, target, names,
    )
    print(f"target logit (= sum of initial R) = {R0_total:.4f}")
    print(f"sum(R_input) = {grad.sum().item():.4f}")
    print(f"input |R|_max = {grad.abs().max().item():.4f}")
    print(f"input pos / neg split = {grad.clamp(min=0).sum().item():.4f} / "
          f"{grad.clamp(max=0).sum().item():.4f}")
    print()

    # Per-block trajectory: walk in autograd order (last block first, since
    # backward goes head → block.23 → ... → block.0 → patch_embed).
    print(f"{'module':<35} | {'R_out_total':>14} | {'R_in_total':>14} | "
          f"{'absorbed':>14} | {'R_pos':>14} | {'R_neg':>14} | {'|R|_max':>10}")
    print("-" * 130)
    walk_order = ["head", "norm"] + [f"blocks.{i}" for i in range(len(model.blocks) - 1, -1, -1)] + ["patch_embed.proj"]
    for name in walk_order:
        if name not in records:
            continue
        rec = records[name]
        if "R_out_total" not in rec:
            continue
        absorbed = rec.get("absorbed", float("nan"))
        absorbed_pct = (absorbed / rec["R_out_total"] * 100) if rec["R_out_total"] != 0 else float("nan")
        absorbed_str = f"{absorbed:>14.4f}"
        if absorbed_pct == absorbed_pct:  # not NaN
            absorbed_str += f" ({absorbed_pct:>+5.1f}%)"
        print(
            f"{name:<35} | {rec['R_out_total']:>14.4f} | "
            f"{rec.get('R_in_total', float('nan')):>14.4f} | "
            f"{absorbed_str:<22} | "
            f"{rec.get('R_in_pos', float('nan')):>14.4f} | "
            f"{rec.get('R_in_neg', float('nan')):>14.4f} | "
            f"{rec.get('R_in_abs_max', float('nan')):>10.2e}"
        )

    print()
    print("Interpretation guide:")
    print("  R_out_total = sum of relevance ENTERING the module from above (head side)")
    print("  R_in_total  = sum of relevance LEAVING the module to below (input side)")
    print("  absorbed    = R_out_total − R_in_total")
    print("                positive value = bias absorption (proper LRP behaviour)")
    print("                negative value = leakage (residual addition or other rule gap)")
    print("                near zero      = perfect per-module conservation")
    print("  R_pos / R_neg = positive vs negative relevance entering R_in")
    print("                 large opposite-sign values = sign-cancellation (max|R| is misleading)")


if __name__ == "__main__":
    main()
