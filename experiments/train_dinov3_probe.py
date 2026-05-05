"""Train an ImageNet-1k linear probe on top of frozen DINOv3 features.

Dev recipe (Imagenette, ~9k train imgs):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset imagenette --split train --epochs 10 \\
        --out data/dinov3_vitl16_probe_imagenette.pt

Prod recipe (full ImageNet train, ~1.28M imgs; gated, see datasets.py):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset imagenet_train --split train --epochs 5 \\
        --out data/dinov3_vitl16_probe_in1k.pt

Pipeline:

1. Load ``vit_large_patch16_dinov3`` (304 M params), freeze backbone.
2. One pass over the dataset: extract cls-token features per image, cache
   to ``<out_stem>_feats.pt``. Subsequent epochs read the cache (instant).
3. Train an ``nn.Linear(1024, 1000)`` head — AdamW, cosine schedule,
   cross-entropy. ImageNet-1k targets (Imagenette → maps to 10 of 1000).
4. Save head state-dict to ``<out>``.

The head is what the CRP pipeline attribute-conditions on (``y =
class_idx``). Backbone is never updated — DINOv3 weights stay frozen.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform
import timm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load as load_dataset  # noqa: E402


def build_backbone(model_name: str, device: str):
    """Frozen DINOv3 backbone. ``num_classes=0`` strips timm's default head;
    we use the cls-token feature directly."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    embed_dim = model.embed_dim if hasattr(model, "embed_dim") else model.num_features
    return model, embed_dim


@torch.no_grad()
def extract_features(
    model, dataset, *, device: str, batch_size: int, num_workers: int = 4,
    log=print,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward every image once, collect cls-token features + labels."""
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True,
    )
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        # forward_features returns (B, num_tokens, D); we want the cls token.
        # For Eva models, forward_head(..., pre_logits=True) handles cls
        # extraction + pooling consistently with how the timm-default head
        # would have been wired up. Use that as the canonical path.
        out = model.forward_features(x)
        f = model.forward_head(out, pre_logits=True)  # (B, D)
        feats.append(f.cpu())
        labels.append(y.clone())
        if (i + 1) % 50 == 0:
            done = (i + 1) * batch_size
            log(f"  extracted {done}/{len(dataset)}  ({(time.time() - t0):.1f}s)")
    return torch.cat(feats, 0), torch.cat(labels, 0)


def cached_features(
    model, dataset, cache_path: Path, *, device: str, batch_size: int,
    num_workers: int = 4, force: bool = False, log=print,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached features if present and valid; otherwise extract + save."""
    if cache_path.exists() and not force:
        log(f"loading cached features from {cache_path}")
        d = torch.load(cache_path)
        if d["feats"].shape[0] == len(dataset):
            return d["feats"], d["labels"]
        log(f"  cache size mismatch ({d['feats'].shape[0]} vs {len(dataset)}); re-extracting")
    log(f"extracting features for {len(dataset)} images")
    feats, labels = extract_features(
        model, dataset, device=device, batch_size=batch_size,
        num_workers=num_workers, log=log,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": feats, "labels": labels}, cache_path)
    log(f"saved {cache_path}  ({feats.shape}, {labels.shape})")
    return feats, labels


def train_test_split(
    feats: torch.Tensor, labels: torch.Tensor, *, test_frac: float = 0.1,
    seed: int = 0,
):
    """Stratified per-class split: from each class's samples, hold out
    ``test_frac`` for evaluation. Stratification matters because some
    datasets (e.g. dsprites) have a single dominant class — random
    unstratified would leave classes unrepresented in the test set.
    Backbone features are deterministic given the input, so the split
    on (cached) features yields the same train/test partition the
    backbone would have given on the raw images.
    """
    g = torch.Generator().manual_seed(seed)
    train_idx, test_idx = [], []
    for c in labels.unique().tolist():
        cls_idx = (labels == c).nonzero(as_tuple=True)[0]
        n_test = max(1, int(round(len(cls_idx) * test_frac)))
        perm = cls_idx[torch.randperm(len(cls_idx), generator=g)]
        test_idx.append(perm[:n_test])
        train_idx.append(perm[n_test:])
    train_idx = torch.cat(train_idx)
    test_idx = torch.cat(test_idx)
    return (
        feats[train_idx], labels[train_idx],
        feats[test_idx], labels[test_idx],
    )


@torch.no_grad()
def evaluate_probe(
    head: nn.Linear, feats: torch.Tensor, labels: torch.Tensor,
    *, batch_size: int = 1024, device: str = "cuda",
) -> dict:
    """Held-out top-1 / top-5 accuracy + cross-entropy on a feature cache."""
    head.eval()
    feats = feats.to(device); labels = labels.to(device)
    n = feats.shape[0]
    correct1 = correct5 = 0
    loss_sum = 0.0
    for i in range(0, n, batch_size):
        x, y = feats[i:i + batch_size], labels[i:i + batch_size]
        logits = head(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        topk = logits.topk(min(5, logits.shape[-1]), dim=-1).indices  # (B, k)
        correct1 += (topk[:, 0] == y).sum().item()
        correct5 += (topk == y.unsqueeze(-1)).any(dim=-1).sum().item()
    return {
        "n": n,
        "loss": loss_sum / n,
        "top1": correct1 / n,
        "top5": correct5 / n,
    }


def train_probe(
    feats: torch.Tensor,
    labels: torch.Tensor,
    *,
    embed_dim: int,
    num_classes: int = 1000,
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = "cuda",
    log=print,
) -> nn.Linear:
    """Standard linear-probe loop on cached features.

    Linear head only — backbone is frozen and already-applied.
    """
    head = nn.Linear(embed_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = feats.shape[0]
    feats = feats.to(device)
    labels = labels.to(device)

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        running_loss = 0.0
        running_correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = feats[idx], labels[idx]
            logits = head(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += loss.item() * len(idx)
            running_correct += (logits.argmax(-1) == y).sum().item()
        sched.step()
        log(
            f"epoch {ep + 1:3d}/{epochs}  "
            f"loss={running_loss / n:.4f}  "
            f"top1={running_correct / n:.4f}  "
            f"lr={sched.get_last_lr()[0]:.2e}"
        )
    return head


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    p.add_argument("--model", default="vit_large_patch16_dinov3")
    p.add_argument("--dataset", default="imagenette",
                   choices=("imagenette", "imagenet_val", "imagenet_val_hf",
                            "imagenet_train", "funny_birds", "dsprites"))
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument(
        "--dsprites-target", default="shape",
        choices=("shape", "scale", "orientation", "posX", "posY"),
        help="Latent factor used as the classification label for dsprites.",
    )
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64,
                   help="batch size for FORWARD feature extraction")
    p.add_argument("--probe-batch-size", type=int, default=1024,
                   help="batch size for LINEAR probe training (cached feats)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--force-extract", action="store_true",
                   help="re-extract features even if cached")
    p.add_argument(
        "--out", type=Path, default=None,
        help="path to save head state-dict (default: data/<model>_probe_<dataset>.pt)",
    )
    p.add_argument(
        "--cache", type=Path, default=None,
        help="path to cached features (default: <out_stem>_feats.pt)",
    )
    p.add_argument(
        "--test-frac", type=float, default=0.1,
        help="Fraction of features to hold out from training for the "
             "test-set sanity check (stratified per class).",
    )
    p.add_argument(
        "--min-test-top1", type=float, default=None,
        help="If set, fail (exit non-zero) when the held-out test top-1 "
             "is below this threshold. Per-dataset defaults applied if "
             "this flag is omitted: funny_birds=0.30, dsprites=0.85, "
             "imagenette=0.70, imagenet_val_hf=0.40.",
    )
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if args.out is None:
        args.out = data_dir / f"{args.model}_probe_{args.dataset}.pt"
    if args.cache is None:
        args.cache = args.out.with_name(args.out.stem + "_feats.pt")

    print(f"device  : {device}")
    print(f"model   : {args.model}")
    print(f"dataset : {args.dataset} ({args.split})")
    print(f"out     : {args.out}")
    print(f"cache   : {args.cache}")

    print(f"\nloading backbone")
    model, embed_dim = build_backbone(args.model, device)
    print(f"  embed_dim = {embed_dim}")
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)

    print(f"\nloading dataset")
    if args.dataset == "imagenet_train":
        # Mirrors load_imagenet_val but on the train tree.
        # Will raise SystemExit with setup pointer if not populated.
        dataset = load_dataset(
            "imagenet_val",  # use val backend; user populates train tree via the same loader
            n_per_class=None, transform=transform,
        )
        # Override path to train tree
        from datasets import load_imagenet_val as _liv  # noqa: F401
        # NOTE: real fix is to parameterise load_imagenet_val with a split
        # arg; for now we expect data/imagenet_val/ to actually be the train
        # tree if --dataset=imagenet_train. Re-named for clarity.
        raise SystemExit(
            "TODO: imagenet_train backend not yet wired in datasets.py — "
            "see plan; populate the layout under data/imagenet_train/<wnid>/ "
            "first, then re-run after the loader supports --split=train."
        )
    elif args.dataset == "imagenet_val_hf":
        # HF mirror — auto-downloads, no split arg.
        dataset = load_dataset(
            "imagenet_val_hf", n_per_class=None, transform=transform,
        )
    elif args.dataset == "funny_birds":
        split = "test" if args.split in ("val", "test") else "train"
        dataset = load_dataset(
            "funny_birds", split=split, transform=transform,
        )
    elif args.dataset == "dsprites":
        # dsprites images are 64×64; the timm transform handles upsampling
        # to the model's input resolution. image_mode='RGB' tiles the
        # grayscale channel for ImageNet-pretrained backbones.
        dataset = load_dataset(
            "dsprites", target=args.dsprites_target, transform=transform,
        )
    else:
        dataset = load_dataset(
            args.dataset,
            n_per_class=None,
            split=args.split if args.dataset == "imagenette" else "val",
            transform=transform,
        )
    print(f"  {dataset.name}: {len(dataset)} images, {dataset.num_classes} classes")

    # Auto-derive num_classes from the dataset for non-ImageNet sources, where
    # the natural class count differs from the default 1000. Imagenette also
    # has 10 classes but we keep it at 1000 by convention (probe head is
    # ImageNet-1k-shaped; only 10 logits get trained).
    if args.dataset in ("funny_birds", "dsprites"):
        if args.num_classes != dataset.num_classes:
            print(
                f"  overriding --num-classes {args.num_classes} → "
                f"{dataset.num_classes} (auto-detected from dataset)"
            )
            args.num_classes = dataset.num_classes

    print(f"\nextracting features (or loading cache)")
    feats, labels = cached_features(
        model, dataset, args.cache, device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, force=args.force_extract,
    )
    print(f"  feats: {feats.shape}, labels: {labels.shape}, "
          f"unique labels: {labels.unique().numel()}")

    print(f"\nstratified train/test split (test_frac={args.test_frac})")
    feats_tr, labels_tr, feats_te, labels_te = train_test_split(
        feats, labels, test_frac=args.test_frac, seed=0,
    )
    print(f"  train: {len(feats_tr)} samples, test: {len(feats_te)} samples")

    print(f"\ntraining linear probe (frozen backbone, head only)")
    head = train_probe(
        feats_tr, labels_tr, embed_dim=embed_dim, num_classes=args.num_classes,
        epochs=args.epochs, batch_size=args.probe_batch_size,
        lr=args.lr, weight_decay=args.weight_decay, device=device,
    )

    print(f"\nsanity check on held-out test split")
    test_metrics = evaluate_probe(head, feats_te, labels_te, device=device)
    print(
        f"  test: n={test_metrics['n']}  loss={test_metrics['loss']:.4f}  "
        f"top1={test_metrics['top1']:.4f}  top5={test_metrics['top5']:.4f}"
    )
    train_metrics = evaluate_probe(head, feats_tr, labels_tr, device=device)
    print(
        f"  train: n={train_metrics['n']}  loss={train_metrics['loss']:.4f}  "
        f"top1={train_metrics['top1']:.4f}  top5={train_metrics['top5']:.4f}"
    )
    gen_gap = train_metrics["top1"] - test_metrics["top1"]
    print(f"  generalisation gap (train top1 − test top1): {gen_gap:+.4f}")

    # Sanity-check thresholds. Below these, attribution / concept work
    # would be meaningless (probe didn't learn the task).
    DEFAULT_MIN_TOP1 = {
        "funny_birds": 0.30,
        "dsprites":    0.85,
        "imagenette":  0.70,
        "imagenet_val_hf": 0.40,
    }
    threshold = (
        args.min_test_top1
        if args.min_test_top1 is not None
        else DEFAULT_MIN_TOP1.get(args.dataset, 0.10)
    )
    if test_metrics["top1"] < threshold:
        raise SystemExit(
            f"\nFAIL: test top-1 = {test_metrics['top1']:.4f} < threshold "
            f"{threshold:.2f} for dataset {args.dataset!r}. Probe did not "
            f"learn the task — re-train with more epochs or check the "
            f"feature/label pipeline. Probe checkpoint NOT saved."
        )
    print(f"  PASS: test top-1 ≥ {threshold:.2f} threshold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "num_classes": args.num_classes,
            "embed_dim": embed_dim,
            "dataset": args.dataset,
            "head_state_dict": head.state_dict(),
            "test_top1": test_metrics["top1"],
            "test_top5": test_metrics["top5"],
            "train_top1": train_metrics["top1"],
        },
        args.out,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
