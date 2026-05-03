"""Per-layer relevance diagnostic for DINOv3 LRP debugging.

Drives one attribution call with a configurable composite, records relevance
at every attention output tap, and reports per-layer health metrics:

* ``nan_count`` — how many entries are NaN at this layer
* ``inf_count`` — how many are ±inf
* ``max_abs_finite`` — magnitude of the largest finite entry (overflow warning if > 1e30)
* ``finite_share`` — fraction of finite entries
* ``mean_abs_finite`` — typical magnitude (helps see slow drift vs sudden explosion)

Plus heatmap-level health:

* ``finite`` — does the input.grad heatmap have any finite entries?
* ``heatmap_max_abs`` — peak magnitude
* ``focus_top10pct`` — fraction of |relevance| concentrated in the top-10%
  of pixels (a uniform/random heatmap = 0.10; a well-localised one = 0.5–0.9)

The harness also supports **register-token isolation**: the DINOv3 backbone
prepends ``num_prefix_tokens=5`` (cls + 4 register), and per Darcet et al.
2023 (arXiv:2309.16588) the register tokens absorb high-norm artifacts and
attach to no input pixel. Relevance landing on register positions is
reported separately as a leak metric and dropped from the spatial heatmap.

Usage from a notebook or a runner script::

    from experiments.dinov3_diagnose import diagnose_attribution
    report = diagnose_attribution(
        model, attribution, composite, image, target_class,
        num_prefix_tokens=5,
    )
    print(format_report(report))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class LayerHealth:
    layer_name: str
    shape: tuple
    nan_count: int
    inf_count: int
    max_abs_finite: float
    mean_abs_finite: float
    finite_share: float


@dataclass
class DiagnosticReport:
    composite_label: str
    target_class: int
    finite_heatmap: bool
    heatmap_max_abs: float
    heatmap_focus_top10pct: float
    register_leak_share: float
    """Fraction of total |relevance| that lands on register-token positions
    (input-disconnected; would be discarded from the spatial heatmap)."""
    per_layer: list[LayerHealth] = field(default_factory=list)
    error: Optional[str] = None


def _layer_health(layer_name: str, rel: torch.Tensor) -> LayerHealth:
    nan = rel.isnan().sum().item()
    inf = rel.isinf().sum().item()
    finite_mask = torch.isfinite(rel)
    finite_share = finite_mask.float().mean().item()
    if finite_mask.any():
        finite = rel[finite_mask]
        max_abs = finite.abs().max().item()
        mean_abs = finite.abs().mean().item()
    else:
        max_abs = float("nan")
        mean_abs = float("nan")
    return LayerHealth(
        layer_name=layer_name,
        shape=tuple(rel.shape),
        nan_count=nan,
        inf_count=inf,
        max_abs_finite=max_abs,
        mean_abs_finite=mean_abs,
        finite_share=finite_share,
    )


def diagnose_attribution(
    model: torch.nn.Module,
    attribution,
    composite,
    image: torch.Tensor,
    target_class: int,
    *,
    composite_label: Optional[str] = None,
    num_prefix_tokens: int = 5,
    record_layers: Optional[list[str]] = None,
) -> DiagnosticReport:
    """Run one attribution and produce a structured health report.

    Records relevance at every ``blocks.{i}.attn.attn_out_tap`` (and
    ``qkv_tap`` if installed) so the per-layer trajectory of relevance
    magnitude is visible — pinpoints the layer at which a remedy first
    diverges or converges.

    ``num_prefix_tokens`` matches the ``Eva.num_prefix_tokens`` (cls + reg).
    The register-token slice is reported as a "leak" metric and zeroed in
    the spatial-heatmap focus calculation.
    """
    if composite_label is None:
        composite_label = type(composite).__name__

    # Record at every attention output tap by walking the model.
    if record_layers is None:
        record_layers = []
        for name, _ in model.named_modules():
            if name.endswith(".attn"):
                # The canonizer installs both .qkv_tap and .attn_out_tap; we
                # only need the output tap to monitor backward magnitudes
                # at every transformer-layer boundary.
                record_layers.append(f"{name}.attn_out_tap")

    image.grad = None
    try:
        result = attribution(
            image, [{"y": [target_class]}], composite,
            record_layer=record_layers,
        )
    except Exception as e:
        return DiagnosticReport(
            composite_label=composite_label,
            target_class=target_class,
            finite_heatmap=False,
            heatmap_max_abs=float("nan"),
            heatmap_focus_top10pct=float("nan"),
            register_leak_share=float("nan"),
            error=f"{type(e).__name__}: {e}",
        )

    per_layer = [
        _layer_health(ln, result.relevances[ln])
        for ln in record_layers if ln in result.relevances
    ]

    # Heatmap (channel-summed) health.
    hm = result.heatmap[0]
    if hm.dim() == 3:
        hm = hm.sum(dim=0)
    hm = hm.detach().cpu()
    hm_finite = torch.isfinite(hm)
    if hm_finite.all():
        flat = hm.abs().flatten()
        total = flat.sum().item()
        if total > 0:
            top10_threshold = torch.quantile(flat, 0.9).item()
            focus = flat[flat >= top10_threshold].sum().item() / total
        else:
            focus = float("nan")
        finite_heatmap = True
        max_abs = hm.abs().max().item()
    else:
        finite_heatmap = False
        focus = float("nan")
        max_abs = float("nan")

    # Register leakage: take any recorded layer whose tensor has the prefix
    # axis (Eva-style, B, N, D), and compute the fraction of |relevance|
    # carried by the first num_prefix_tokens. We use the deepest layer's
    # relevance since that's where the initial backward signal lands.
    register_leak = float("nan")
    for ln in reversed(record_layers):
        if ln not in result.relevances:
            continue
        rel = result.relevances[ln]
        if rel.dim() != 3 or rel.shape[1] <= num_prefix_tokens:
            continue
        finite = torch.isfinite(rel)
        if not finite.all():
            break
        total = rel.abs().sum().item()
        if total > 0:
            reg_total = rel[:, :num_prefix_tokens, :].abs().sum().item()
            register_leak = reg_total / total
        break

    return DiagnosticReport(
        composite_label=composite_label,
        target_class=target_class,
        finite_heatmap=finite_heatmap,
        heatmap_max_abs=max_abs,
        heatmap_focus_top10pct=focus,
        register_leak_share=register_leak,
        per_layer=per_layer,
    )


def format_report(report: DiagnosticReport, *, max_layers_to_show: int = 8) -> str:
    """Pretty-print a DiagnosticReport for stdout / notebook display."""
    lines = []
    lines.append(f"=== {report.composite_label}  •  cls {report.target_class} ===")
    if report.error:
        lines.append(f"  ERROR: {report.error}")
        return "\n".join(lines)
    lines.append(
        f"  heatmap: finite={report.finite_heatmap}  "
        f"max_abs={report.heatmap_max_abs:.3e}  "
        f"focus_top10%={report.heatmap_focus_top10pct:.3f}  "
        f"register_leak={report.register_leak_share:.3f}"
    )
    if report.per_layer:
        lines.append(f"  per-layer relevance ({len(report.per_layer)} taps):")
        # Show first, last, and stride through the rest.
        n = len(report.per_layer)
        if n <= max_layers_to_show:
            sel = list(range(n))
        else:
            stride = max(1, (n - 2) // (max_layers_to_show - 2))
            sel = sorted(set([0, n - 1] + list(range(0, n, stride))))[:max_layers_to_show]
        for i in sel:
            h = report.per_layer[i]
            tag = ""
            if h.nan_count > 0:
                tag = " ← NaN"
            elif h.max_abs_finite > 1e20:
                tag = " ← overflow risk"
            lines.append(
                f"    [{i:2d}] {h.layer_name:50}  "
                f"max_abs={h.max_abs_finite:.2e}  "
                f"mean_abs={h.mean_abs_finite:.2e}  "
                f"nan={h.nan_count:>4d}{tag}"
            )
    return "\n".join(lines)


def diverges_at(report: DiagnosticReport, threshold: float = 1e10) -> Optional[str]:
    """Return the name of the first layer whose ``max_abs_finite`` exceeds
    ``threshold`` (going from deepest → shallowest, i.e. backward order).
    Returns None if the report stays well-behaved."""
    for h in report.per_layer:
        if h.nan_count > 0:
            return f"{h.layer_name} (NaN)"
        if h.max_abs_finite > threshold:
            return f"{h.layer_name} ({h.max_abs_finite:.2e})"
    return None


__all__ = [
    "LayerHealth",
    "DiagnosticReport",
    "diagnose_attribution",
    "format_report",
    "diverges_at",
]
