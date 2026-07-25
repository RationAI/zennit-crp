"""Evaluate a public DINOv3 ImageNet-1k linear head on an ImageNet val subset.

The canvit probes (https://github.com/m2b3/dinov3-in1k-probes) are linear
classifiers on the frozen DINOv3 **cls token**, trained on IN1k at 512x512.
This script wires one onto the timm DINOv3 backbone and measures top-1/top-5
on a class-diverse subset of ``imagenet_val_hf`` (the un-gated 256px-resized
mirror — note 512px eval therefore upsamples from 256px sources).

Usage::

    python -m experiments.scripts.eval_dinov3_in1k_probe --image-size 256
    python -m experiments.scripts.eval_dinov3_in1k_probe --image-size 512

Defaults: DINOv3-B/16 backbone + canvit vitb16 head, 5 images/class (5000).
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
import typer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import transforms

from experiments.datasets import load_imagenet_val_hf

REPO_ROOT = Path(__file__).resolve().parents[2]

HEAD_REPOS = {
    "vit_base_patch16_dinov3.lvd1689m":
        "canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe",
    "vit_small_patch16_dinov3.lvd1689m":
        "canvit/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe",
    "vit_large_patch16_dinov3.lvd1689m":
        "canvit/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe",
}


def load_in1k_linear_head(timm_name: str, device: str = "cpu") -> nn.Linear:
    """Download the matching canvit IN1k linear head (cls-token -> 1000
    logits) and return it as a frozen ``nn.Linear``."""
    repo = HEAD_REPOS[timm_name]
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    head = nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    head.load_state_dict(sd)
    head.requires_grad_(False)
    return head.eval().to(device)


def main(
    timm_name: str = typer.Option("vit_base_patch16_dinov3.lvd1689m", "--timm-name"),
    image_size: int = typer.Option(256, "--image-size",
                                   help="Eval resolution (head was trained at 512)."),
    n_per_class: int = typer.Option(5, "--n-per-class"),
    batch_size: int = typer.Option(64, "--batch-size"),
    device: str = typer.Option("cuda"),
):
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0,
                                 img_size=image_size)
    backbone.requires_grad_(False)
    backbone = backbone.eval().to(device)
    head = load_in1k_linear_head(timm_name, device)

    cfg = backbone.pretrained_cfg
    tf = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    ds = load_imagenet_val_hf(root=REPO_ROOT / "data",
                              n_per_class=n_per_class, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)

    top1 = top5 = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # canvit probes are trained on the CLS token (Meta's
            # x_norm_clstoken). timm's dinov3 is an Eva model with
            # global_pool='avg', so forward_head(pre_logits=True) would
            # return mean patch tokens — the WRONG feature (costs ~30
            # points top-1). forward_features applies the final norm;
            # token 0 is the normed cls token.
            cls = backbone.forward_features(x)[:, 0]
            logits = head(cls)
            top1 += (logits.argmax(-1) == y).sum().item()
            top5 += (logits.topk(5, dim=-1).indices == y[:, None]).any(-1).sum().item()
            n += len(y)
            if n % 1024 < batch_size:
                print(f"  {n} imgs: top1 {top1/n:.4f} top5 {top5/n:.4f}", flush=True)
    print(f"\n{timm_name} + {HEAD_REPOS[timm_name]}")
    print(f"image_size={image_size} n={n}  top1={top1/n:.4f}  top5={top5/n:.4f}")


if __name__ == "__main__":
    typer.run(main)
