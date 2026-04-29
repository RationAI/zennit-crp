"""Conservation diagnostic across ViT sizes and composites.

Per ``FUTURE_STATE.md`` Milestone D13: measure
``R_input.sum() / R_output.sum()`` per attribution. ``R_output`` here is
the masked logit value (no concept-mask in the condition). Ideal value 1.0
under perfect AttnLRP conservation; current pipeline drifts because (i) the
additive ``pos_embed`` step is plain tensor + (no LRP rule unless
``palrp=True``), and (ii) per-block residual additions are also plain
tensor + with no rule applied.

Usage::

    uv run python experiments/conservation_check.py \\
        --models vit_tiny_patch16_224,vit_small_patch16_224,vit_base_patch16_224 \\
        --pretrained --target 217 --image-dir data/curated_milestone_a/217

Without ``--image-dir``, uses random Gaussian input. With ``--pretrained``
and a real image, the ratios reflect what the milestone-A sweep actually
sees.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crp.attribution import CondAttribution
from crp.transformer_patches import (
    AttnLRPEpsilonComposite,
    AttnLRPGammaComposite,
)


def load_image(path: Path, model: torch.nn.Module, device: str) -> torch.Tensor:
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device).requires_grad_(True)


def random_input(model: torch.nn.Module, device: str) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224, requires_grad=True, device=device)


def measure(model, attribution, composite, data, target):
    with torch.no_grad():
        logit = model(data)[0, target].item()
    data.grad = None
    attribution(data, [{"y": [target]}], composite)
    s = data.grad.sum().item()
    sa = data.grad.abs().sum().item()
    ratio = s / logit if logit != 0 else float("nan")
    return s, sa, logit, ratio


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--models",
        default="vit_tiny_patch16_224,vit_small_patch16_224,vit_base_patch16_224",
        help="comma-separated timm model names",
    )
    p.add_argument("--pretrained", action="store_true")
    p.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="optional: take the first image of this dir as input",
    )
    p.add_argument("--target", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    if args.image_dir is not None:
        first = sorted(args.image_dir.iterdir())[0]
        print(f"using image: {first}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(
        f"{'model':28} {'rule':18} {'sum(R)':>14} {'sum(|R|)':>14} "
        f"{'logit':>8} {'ratio':>14}"
    )
    print("-" * 100)
    for name in models:
        model = timm.create_model(name, pretrained=args.pretrained).eval().to(device)
        data = (
            load_image(sorted(args.image_dir.iterdir())[0], model, device)
            if args.image_dir is not None
            else random_input(model, device)
        )
        attribution = CondAttribution(model, device=torch.device(device))
        for label, comp in [
            ("eps           ", AttnLRPEpsilonComposite(palrp=False)),
            ("eps + palrp   ", AttnLRPEpsilonComposite(palrp=True)),
            (f"γ={args.gamma}        ", AttnLRPGammaComposite(gamma=args.gamma, palrp=False)),
            (f"γ={args.gamma} + palrp", AttnLRPGammaComposite(gamma=args.gamma, palrp=True)),
        ]:
            s, sa, logit, ratio = measure(model, attribution, comp, data, args.target)
            # Exponent only for very large numbers to keep table readable.
            r_str = f"{ratio:>14.4f}" if abs(ratio) < 1e7 else f"{ratio:>14.2e}"
            s_str = f"{s:>14.4f}" if abs(s) < 1e7 else f"{s:>14.2e}"
            sa_str = f"{sa:>14.4f}" if abs(sa) < 1e7 else f"{sa:>14.2e}"
            print(
                f"{name:28} {label:18} {s_str} {sa_str} "
                f"{logit:>8.3f} {r_str}"
            )
        # Free GPU memory before next model
        del model, attribution
        torch.cuda.empty_cache() if device == "cuda" else None

    print()
    print("Ideal conservation: ratio == 1.0  (under perfect AttnLRP).")
    print("ε without palrp drifts from over-counting (vit_base ≈ 1.8×).")
    print("ε with palrp halves at the additive pos_embed step.")
    print("γ-LRP ratios blow up — the positive-weight clamp amplifies noise.")


if __name__ == "__main__":
    main()
