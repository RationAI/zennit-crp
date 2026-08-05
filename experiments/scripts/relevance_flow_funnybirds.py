"""Quantify residual vs module relevance flow per transformer block.

For the visinf FunnyBirds-pretrained ``vit_base_patch16_224``, attribute
predictions on the first ``N`` images of the test split and, at every
block's two residual-add junctions (post-attn and post-mlp), record:

* ``mod_rel_*`` — relevance flowing through the module branch
  (``attn`` or ``mlp``), the gradient produced by
  ``_ResidualRatioFn.backward`` and assigned to the branch operand.
* ``res_rel_*`` — relevance flowing through the skip path, the gradient
  produced by ``_ResidualRatioFn.backward`` and assigned to the ``x``
  operand.

Both halves come from a single ``_ResidualRatioFn.backward`` call
(Otsuki ratio split: ``R_x = R_y · |x|/(|x|+|b|+ε)``,
``R_branch = R_y · |b|/(|x|+|b|+ε)``). The branch is then propagated
back through the module (attn/mlp), and the skip's relevance reaches
the previous block as-is.

Measurement mechanism
---------------------
We can't just read ``.grad`` on the residual-add operands, because
``x_after_attn`` is also consumed by the MLP branch — its ``.grad``
accumulates BOTH the skip-side split AND the relevance flowing back
through the MLP module. To isolate the per-junction skip vs module
split we wrap each residual ``_ResidualRatioFn.apply(x, b)`` call in a
custom autograd ``Function`` (:class:`_CapturingResidualRatio`) that
performs the SAME forward ``x + b`` and the SAME backward (Otsuki
ratio) but ALSO writes the two output gradients into a per-block
captures dict before returning them.

The override is installed by setting a probe-variant ``forward`` on
each :class:`timm.models.vision_transformer.Block` instance (this
overrides the ``forward`` bound by ``TimmBlockResidualCanonizer`` for
the duration of the composite context; the composite's ``remove()`` on
context exit clears our attribute via ``delattr``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torchvision.transforms import functional as TF
import polars as pl
import typer

from timm.models.vision_transformer import Block as TimmBlock
from zennit.canonizers import AttributeCanonizer

import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Pass
from zennit_extensions.attention_unfolded import (
    BilinearMatmul, ScaleByConstant, SoftmaxAlongLastDim,
)
from zennit_extensions.canonisation.canonizers import (
    EvaAttentionSubstitutionCanonizer, TimmAttentionSubstitutionCanonizer,
)
from zennit_extensions.rules.bajger_contrib import AlphaBetaMatmul
from crp.attribution import CondAttribution
from experiments.datasets import load as load_dataset
from experiments.models import build_probe


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROBE_PATH = DATA_DIR / "vit_base_linear_probe_funny_birds.pt"
OUT_PATH = DATA_DIR / "relevance_flow_funnybirds_vit_base.parquet"

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def visinf_transform(pil):
    """Visinf vit_base preprocessing: ToTensor (no normalize) + bilinear
    resize to 224×224."""
    t = TF.to_tensor(pil)[:3]
    t = F.interpolate(
        t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False,
    ).squeeze(0)
    return t


# A per-call-site bucket; the forward stashes (block_idx, junction) on
# ctx and backward looks them up to choose which captures slot to write.
# Captures dict is a module-level handle so the autograd Function can
# write to it without closure tricks across CUDA streams.
_CAPTURES: Dict[int, Dict[str, torch.Tensor]] = {}


class _CapturingResidualRatio(Function):
    """``y = x + branch``; backward: Otsuki ratio split, AND store the two
    output gradients in ``_CAPTURES[block_idx][junction + '_grad_x']`` and
    ``_CAPTURES[block_idx][junction + '_grad_branch']``.

    Forward + backward are bit-identical to ``_ResidualRatioFn`` in
    :mod:`zennit_ext`. The capture is a side-effect on
    ``backward`` only — no autograd-graph perturbation.
    """

    @staticmethod
    def forward(ctx, x, branch, block_idx, junction, epsilon=1e-6):
        ctx.save_for_backward(x, branch)
        ctx.epsilon = epsilon
        ctx.block_idx = block_idx
        ctx.junction = junction
        return x + branch

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        x, branch = ctx.saved_tensors
        abs_x = x.abs()
        abs_b = branch.abs()
        denom = abs_x + abs_b + ctx.epsilon
        grad_x = grad_output * (abs_x / denom)
        grad_branch = grad_output * (abs_b / denom)
        slot = _CAPTURES.setdefault(ctx.block_idx, {})
        # Detach + move to CPU isn't required; we extract scalar stats
        # later and never reuse the tensor.
        slot[f"{ctx.junction}_grad_x"] = grad_x.detach()
        slot[f"{ctx.junction}_grad_branch"] = grad_branch.detach()
        return grad_x, grad_branch, None, None, None


def _make_probe_block_forward(block_idx: int):
    """Build a TimmBlock.forward replacement that routes each residual
    add through :class:`_CapturingResidualRatio` for this block."""
    def fwd(self, x, attn_mask=None, is_causal=False):
        branch1 = self.drop_path1(
            self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal))
        )
        x = _CapturingResidualRatio.apply(x, branch1, block_idx, "attn")
        branch2 = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = _CapturingResidualRatio.apply(x, branch2, block_idx, "mlp")
        return x
    return fwd


class CapturingBlockCanonizer(AttributeCanonizer):
    """Replacement for ``TimmBlockResidualCanonizer`` that routes both
    residual adds through :class:`_CapturingResidualRatio`. Side-effect:
    the per-junction ``grad_x`` and ``grad_branch`` from
    ``_ResidualRatioFn``'s backward are stashed into the
    module-global ``_CAPTURES`` dict keyed by block index.

    Identifying each block: we walk ``root_module.named_modules()`` in
    ``apply``; the block index is parsed from the name (``backbone.blocks.{i}``).
    """

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, name: str, module):
        if not isinstance(module, TimmBlock):
            return None
        # Names are of the form ``...blocks.<i>`` — pull the integer.
        try:
            block_idx = int(name.rsplit(".blocks.", 1)[-1])
        except (ValueError, IndexError):
            return None
        fwd = _make_probe_block_forward(block_idx)
        return {"forward": fwd.__get__(module, type(module))}

    def copy(self):
        return type(self)()


def _stats(g: torch.Tensor) -> dict:
    flat = g.reshape(-1).double()
    return {
        "sum": float(flat.sum().item()),
        "max": float(flat.max().item()),
        "min": float(flat.min().item()),
        "var": float(flat.var(unbiased=False).item()),
    }


@app.command()
def main(
    n_images: int = typer.Option(200, "--n-images", help="Test images to process."),
    out_path: Path = typer.Option(OUT_PATH, "--out", help="Output parquet path."),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu", "--device"
    ),
) -> None:
    print(f"device: {device}")
    if not PROBE_PATH.is_file():
        raise FileNotFoundError(
            f"Probe not found at {PROBE_PATH}. Run:\n"
            f"  uv run python experiments/scripts/setup_funnybirds_vit_base.py --force"
        )

    # ── model ───────────────────────────────────────────────────────────
    ckpt = torch.load(PROBE_PATH, map_location=device, weights_only=False)
    model = build_probe(
        base=ckpt["base"], head=ckpt["head"], num_classes=ckpt["num_classes"],
        head_kwargs=ckpt.get("head_kwargs", {}),
    ).eval().to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.backbone.load_state_dict(ckpt["backbone_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    num_blocks = len(model.backbone.blocks)
    print(f"num_blocks: {num_blocks}")

    # ── dataset ─────────────────────────────────────────────────────────
    dataset = load_dataset("funny_birds", transform=visinf_transform, split="test")
    n_total = min(n_images, len(dataset))
    print(f"dataset: funny_birds test split — using {n_total}/{len(dataset)} imgs")

    # ── composite + attribution wiring ──────────────────────────────────
    # Inline composite: this diagnostic's own recipe (AlphaBeta bilinears,
    # epsilon linears) with ``CapturingBlockCanonizer`` in place of the stock
    # block-residual canonizer — it installs the residual rule AND the
    # per-junction gradient capture in one go.
    composite = LayerMapComposite(
        layer_map=[
            (BilinearMatmul, AlphaBetaMatmul(alpha=0.5, beta=0.5, epsilon=1e-6)),
            (SoftmaxAlongLastDim, Pass()),
            (ScaleByConstant, Pass()),
            (nn.Linear, Epsilon(epsilon=1e-6)),
            (nn.Conv2d, Epsilon(epsilon=1e-6)),
            (nn.GELU, Pass()), (nn.LayerNorm, Pass()), (nn.Dropout, Pass()),
            (nn.Identity, Pass()),
        ],
        canonizers=[
            CapturingBlockCanonizer(),
            EvaAttentionSubstitutionCanonizer(block_indices=None),
            TimmAttentionSubstitutionCanonizer(block_indices=None),
        ],
    )
    attribution = CondAttribution(model)

    rows = []

    # Pre-compute argmax predictions OUTSIDE the composite context. Doing
    # this inside ``composite.context`` would trip zennit's pre_forward
    # hook, which tries to inject a ``grad_fn`` via ``Identity.apply``
    # under ``torch.no_grad()`` and raises ``Backward hook could not be
    # registered!``.
    pred_classes = []
    with torch.no_grad():
        for img_idx in range(n_total):
            x, _y = dataset[img_idx]
            logits = model(x.unsqueeze(0).to(device))
            pred_classes.append(int(logits.argmax(-1).item()))
    print(f"prefilled {len(pred_classes)} argmax predictions")

    # No outer composite context — ``CondAttribution.__call__`` enters
    # its own ``composite.context``, which applies our
    # ``CapturingBlockCanonizer`` and reverts on exit.
    for img_idx in range(n_total):
        x, _y = dataset[img_idx]
        x_attr = x.unsqueeze(0).to(device).requires_grad_(True)
        pred_class = pred_classes[img_idx]
        _CAPTURES.clear()
        _ = attribution(x_attr, [{"y": [pred_class]}], composite)

        for blk in range(num_blocks):
            cap = _CAPTURES[blk]
            # ── post-attn residual junction ────────────────────────
            attn_mod = _stats(cap["attn_grad_branch"])
            attn_skp = _stats(cap["attn_grad_x"])
            rows.append({
                "image_id":   img_idx,
                "module_id":  f"block_{blk}_attn",
                "mod_rel_sum": attn_mod["sum"], "res_rel_sum": attn_skp["sum"],
                "mod_rel_max": attn_mod["max"], "res_rel_max": attn_skp["max"],
                "mod_rel_min": attn_mod["min"], "res_rel_min": attn_skp["min"],
                "mod_rel_var": attn_mod["var"], "res_rel_var": attn_skp["var"],
            })
            # ── post-mlp residual junction ─────────────────────────
            mlp_mod = _stats(cap["mlp_grad_branch"])
            mlp_skp = _stats(cap["mlp_grad_x"])
            rows.append({
                "image_id":   img_idx,
                "module_id":  f"block_{blk}_mlp",
                "mod_rel_sum": mlp_mod["sum"], "res_rel_sum": mlp_skp["sum"],
                "mod_rel_max": mlp_mod["max"], "res_rel_max": mlp_skp["max"],
                "mod_rel_min": mlp_mod["min"], "res_rel_min": mlp_skp["min"],
                "mod_rel_var": mlp_mod["var"], "res_rel_var": mlp_skp["var"],
            })

        if (img_idx + 1) % 25 == 0:
            print(f"  processed {img_idx + 1}/{n_total}", flush=True)

    schema = {
        "image_id":     pl.UInt32,
        "module_id":    pl.Utf8,
        "mod_rel_sum":  pl.Float64,
        "res_rel_sum":  pl.Float64,
        "mod_rel_max":  pl.Float64,
        "res_rel_max":  pl.Float64,
        "mod_rel_min":  pl.Float64,
        "res_rel_min":  pl.Float64,
        "mod_rel_var":  pl.Float64,
        "res_rel_var":  pl.Float64,
    }
    df = pl.DataFrame(rows, schema=schema)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"\nWrote {len(df)} rows → {out_path}")

    # ── on-the-fly summary ──────────────────────────────────────────────
    # |mod| / (|mod| + |skip|) of the per-junction sums (signed sums can
    # cancel; using abs of the SUM is the standard "magnitude share"
    # readout used in the AttnLRP residual-skip analyses).
    fdf = (
        df.with_columns(
            (
                pl.col("mod_rel_sum").abs()
                / (pl.col("mod_rel_sum").abs() + pl.col("res_rel_sum").abs() + 1e-12)
            ).alias("mod_frac")
        )
        .group_by("module_id")
        .agg(pl.col("mod_frac").mean().alias("mean_mod_frac"))
        .sort("module_id")
    )
    print("\nmean module-branch magnitude fraction per junction (200 imgs):")
    for row in fdf.iter_rows(named=True):
        print(f"  {row['module_id']:>18s}: {row['mean_mod_frac']:.4f}")


if __name__ == "__main__":
    app()
