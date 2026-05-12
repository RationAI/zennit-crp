"""Download the visinf/funnybirds-framework FunnyBirds-pretrained ViT-base
checkpoint and repackage it into our probe ``.pt`` payload format.

The visinf checkpoint is end-to-end-trained ``vit_base_patch16_224`` on
the 50-class FunnyBirds task. It's a standard timm-compatible state_dict
(no prefix), with a ``Linear(768, 50)`` classifier under the ``head.*``
keys. We split it into:

* ``backbone_state_dict`` (everything not under ``head.``) — slots into
  ``timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)``.
* ``head_state_dict``  — ``head.weight`` / ``head.bias`` remapped to
  ``linear.weight`` / ``linear.bias`` so it slots into our
  :class:`experiments.models.heads.LinearHead`.

After running, ``data/vit_base_linear_probe_funny_birds.pt`` contains a
``finetune_cmd``-style payload (backbone + head). The walkthrough
notebook's probe-load cell consumes it via the same code path as any
fine-tuned probe.

Usage::

    uv run python experiments/scripts/setup_funnybirds_vit_base.py
"""
from __future__ import annotations

from pathlib import Path

import torch
import typer

from experiments.datasets.funny_birds import _stream_download
from experiments.models import build_probe

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

VISINF_URL = (
    "https://download.visinf.tu-darmstadt.de/data/funnybirds/models/"
    "vit_base_patch16_224_final_1_checkpoint_best.pth.tar"
)

# Where the raw .pth.tar lands after download (we keep it around so a
# re-run reuses the cached file instead of re-downloading).
RAW_PATH = DATA_DIR / "vit_base_funnybirds_pretrained.pth.tar"

# The repackaged probe payload. The filename matches the walkthrough
# notebook's PROBE_PATH convention ``{BASE}_{HEAD}_probe_{DATASET}.pt``
# for ``BASE='vit_base', HEAD='linear', DATASET='funny_birds'``.
PROBE_PATH = DATA_DIR / "vit_base_linear_probe_funny_birds.pt"


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    url: str = typer.Option(VISINF_URL, "--url", help="Source URL."),
    raw_path: Path = typer.Option(RAW_PATH, "--raw-path", help="Where to cache the .pth.tar."),
    out_path: Path = typer.Option(PROBE_PATH, "--out", help="Output probe .pt path."),
    force: bool = typer.Option(False, "--force", help="Re-process even if out exists."),
) -> None:
    """Download visinf ViT-base FunnyBirds checkpoint + repackage."""
    if out_path.is_file() and not force:
        print(f"probe already exists at {out_path}")
        print("→ pass --force to re-process")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"step 1: download → {raw_path}", flush=True)
    _stream_download(url, raw_path)

    print(f"\nstep 2: load + split state_dict from {raw_path}", flush=True)
    ckpt = torch.load(raw_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    print(f"  raw keys: {len(sd)}; sample: {list(sd)[:3]} ...")

    # Split backbone vs head.
    backbone_sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    raw_head_w = sd["head.weight"]
    raw_head_b = sd["head.bias"]
    print(f"  backbone keys: {len(backbone_sd)}")
    print(f"  head.weight shape: {tuple(raw_head_w.shape)}")
    print(f"  head.bias shape  : {tuple(raw_head_b.shape)}")

    # Map head.* → LinearHead.linear.* (our head wraps the classifier as
    # ``self.linear = nn.Linear(embed_dim, num_classes)``).
    head_sd = {
        "linear.weight": raw_head_w,
        "linear.bias": raw_head_b,
    }

    num_classes = int(raw_head_w.shape[0])
    embed_dim = int(raw_head_w.shape[1])
    if (num_classes, embed_dim) != (50, 768):
        raise RuntimeError(
            f"Unexpected head shape {(num_classes, embed_dim)} — visinf checkpoint "
            "is supposed to be 50-class FunnyBirds on vit_base_patch16_224 (768)."
        )

    # The visinf training pipeline (train.py + models/model_wrapper.py)
    # feeds the ViT raw [0, 1] tensors via `transforms.ToTensor()` (NO
    # ImageNet/JFT normalize) and resizes 256x256 → 224x224 with
    # F.interpolate's default `mode='nearest'` inside ViTModel.forward.
    # Loaders MUST replicate this preprocessing or accuracy crashes from
    # ~99% to ~85% (with timm's (0.5,0.5,0.5)-normalize + crop_pct=0.9
    # default). The notebook reads `transform_spec` and dispatches to
    # the matching preprocessing recipe.
    advertised_acc = float(ckpt.get("best_acc1", 0.0)) / 100.0 or None
    payload = {
        "base": "vit_base",
        "head": "linear",
        "head_kwargs": {},
        "num_classes": num_classes,
        "embed_dim": embed_dim,
        "dataset": "funny_birds",
        "backbone_state_dict": backbone_sd,
        "head_state_dict": head_sd,
        "transform_spec": "visinf_funnybirds_vit_base",
        "val_acc": advertised_acc,
        "val_acc5": None,
        "val_loss": None,
        "finetuned_from": (
            "visinf/funnybirds-framework: "
            "vit_base_patch16_224_final_1_checkpoint_best"
        ),
    }
    print(f"\nstep 3: write probe payload → {out_path}", flush=True)
    torch.save(payload, out_path)
    print(f"  wrote {out_path.stat().st_size / 1e6:.0f} MB")

    print("\nstep 4: smoke-test (build_probe + load + forward)", flush=True)
    model = build_probe("vit_base", "linear", num_classes=num_classes)
    missing_b, unexpected_b = model.backbone.load_state_dict(
        backbone_sd, strict=False,
    )
    missing_h, unexpected_h = model.head.load_state_dict(head_sd, strict=True)
    if missing_b or unexpected_b:
        print(
            f"  backbone load — missing: {len(missing_b)}, unexpected: {len(unexpected_b)}"
        )
        if unexpected_b:
            print(f"    unexpected[:5]: {unexpected_b[:5]}")
    else:
        print("  backbone load: all keys matched")
    print(f"  head load: missing={len(missing_h)}, unexpected={len(unexpected_h)}")

    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
    assert logits.shape == (2, num_classes), (
        f"smoke-test forward produced {tuple(logits.shape)}, expected (2, {num_classes})"
    )
    print(f"  forward OK: logits.shape={tuple(logits.shape)}")
    print(
        f"\ndone. Walkthrough notebook will pick this up automatically when:\n"
        f"    BASE = 'vit_base'; HEAD = 'linear'; DATASET = 'funny_birds'"
    )


if __name__ == "__main__":
    app()
