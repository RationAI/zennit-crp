"""Train a linear probe on top of frozen DINOv3 features.

Recipes — copy-paste ready, paths match what the walkthrough notebook
expects (``data/vit_large_patch16_dinov3_probe_<dataset>.pt``):

FunnyBirds (50 synthetic bird classes, ~50k train imgs, 1.5 GB auto-DL):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset funny_birds

dsprites (3 shape classes by default, ~26 MB auto-DL):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset dsprites

ImageNet-1k val (un-gated HF mirror, ~830 MB auto-DL):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset imagenet_val_hf

Imagenette (10-class ImageNet subset, ~98 MB auto-DL — quick dev loop):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset imagenette

Pipeline:

1. Load ``vit_large_patch16_dinov3`` (304 M params), freeze backbone.
2. One pass over the dataset: extract cls-token features per image, cache
   to ``<out_stem>_feats.pt``. Subsequent runs read the cache (instant).
3. Random 90/10 train/val split on cached features.
4. Lightning trains an ``nn.Linear(embed_dim, num_classes)`` head with
   AdamW + cross-entropy. ``ModelCheckpoint(monitor='val_acc')`` keeps
   the best epoch; ``EarlyStopping(patience=5)`` stops when val_acc
   plateaus. ``num_classes`` is auto-detected from the dataset.
5. Save best head state-dict + val metrics to the output ``.pt``.

The head is what the CRP pipeline attribute-conditions on (``y =
class_idx``). Backbone is never updated — DINOv3 weights stay frozen.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import lightning as L
import timm
import torch
import torch.multiprocessing as _torch_mp
import torch.nn as nn
import torch.nn.functional as F
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.classification import MulticlassAccuracy

# Use file-system sharing so DataLoader workers don't need /dev/shm
# (containers typically cap it at 64 MB).
_torch_mp.set_sharing_strategy("file_system")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load as load_dataset  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ── Lightning module ──────────────────────────────────────────────────────────


class LinearProbe(L.LightningModule):
    """nn.Linear head trained on cached DINOv3 cls features."""

    def __init__(
        self, embed_dim: int, num_classes: int,
        lr: float = 1e-3, weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Linear(embed_dim, num_classes)
        kw = dict(num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(**kw)
        self.val_acc = MulticlassAccuracy(**kw)
        self.val_acc5 = MulticlassAccuracy(top_k=5, **kw)

    def forward(self, x):
        return self.head(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self.head(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self.head(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)
        self.val_acc5(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc5", self.val_acc5, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


# ── Feature extraction (one-shot, cached) ────────────────────────────────────


@torch.no_grad()
def extract_features(
    backbone, dataset, *, device: str, batch_size: int, num_workers: int,
):
    """One pass: forward each image through the backbone, take cls
    pre-logits."""
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True,
    )
    feats, labels = [], []
    n = len(dataset)
    seen = 0
    import time
    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = backbone.forward_features(x)
        f = backbone.forward_head(out, pre_logits=True)
        feats.append(f.cpu())
        labels.append(y if torch.is_tensor(y) else torch.as_tensor(y))
        seen += x.shape[0]
        if seen % (50 * batch_size) < batch_size:
            print(f"  extracted {seen}/{n}  ({time.time() - t0:.1f}s)")
    return torch.cat(feats, 0), torch.cat(labels, 0)


def cached_features(
    backbone, dataset, cache_path: Path, *,
    device: str, batch_size: int, num_workers: int, force: bool = False,
):
    if cache_path.exists() and not force:
        d = torch.load(cache_path)
        if d["feats"].shape[0] == len(dataset):
            print(f"loaded cached features from {cache_path}")
            return d["feats"], d["labels"]
        print(f"  cache size mismatch ({d['feats'].shape[0]} vs {len(dataset)}); re-extracting")
    print(f"extracting features for {len(dataset)} images")
    feats, labels = extract_features(
        backbone, dataset, device=device,
        batch_size=batch_size, num_workers=num_workers,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": feats, "labels": labels}, cache_path)
    print(f"saved {cache_path}  (feats={tuple(feats.shape)}, labels={tuple(labels.shape)})")
    return feats, labels


# ── CLI ──────────────────────────────────────────────────────────────────────


_DATASETS = ("funny_birds", "dsprites", "imagenet_val_hf", "imagenette")


@app.command()
def main(
    dataset: str = typer.Option(
        "funny_birds", "--dataset",
        help=f"Source dataset. One of: {', '.join(_DATASETS)}.",
    ),
    model_name: str = typer.Option(
        "vit_large_patch16_dinov3", "--model",
        help="timm model name for the frozen backbone.",
    ),
    epochs: int = typer.Option(
        50, "--epochs",
        help="Max epochs. EarlyStopping usually stops earlier.",
    ),
    patience: int = typer.Option(
        5, "--patience", help="EarlyStopping patience on val_acc.",
    ),
    lr: float = typer.Option(1e-3, "--lr"),
    weight_decay: float = typer.Option(1e-2, "--weight-decay"),
    batch_size: int = typer.Option(
        256, "--batch-size", help="Probe (head) train batch size.",
    ),
    extract_batch_size: int = typer.Option(
        32, "--extract-batch-size",
        help="Backbone forward-pass batch size for feature extraction.",
    ),
    num_workers: int = typer.Option(
        0, "--num-workers",
        help="DataLoader workers for feature extraction. Use 0 in "
             "containers with tiny /dev/shm.",
    ),
    val_frac: float = typer.Option(
        0.1, "--val-frac", help="Held-out validation fraction.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="Probe output .pt path. Default: "
             "data/<model>_probe_<dataset>.pt",
    ),
    cache: Optional[Path] = typer.Option(
        None, "--cache",
        help="Feature cache .pt path. Default: <out_stem>_feats.pt",
    ),
    force_extract: bool = typer.Option(
        False, "--force-extract",
        help="Re-extract features even if cache exists.",
    ),
    dsprites_target: str = typer.Option(
        "shape", "--dsprites-target",
        help="Latent factor used as the classification target for dsprites.",
    ),
    seed: int = typer.Option(0, "--seed"),
):
    """Train a frozen-DINOv3 linear probe with PyTorch Lightning."""
    if dataset not in _DATASETS:
        raise typer.BadParameter(
            f"unknown dataset {dataset!r}; choose from {', '.join(_DATASETS)}"
        )
    L.seed_everything(seed, workers=True)

    if out is None:
        out = REPO_ROOT / "data" / f"{model_name}_probe_{dataset}.pt"
    if cache is None:
        cache = out.with_name(out.stem + "_feats.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device   : {device}")
    print(f"model    : {model_name}")
    print(f"dataset  : {dataset}")
    print(f"out      : {out}")
    print(f"cache    : {cache}")

    # ── backbone ──
    print("\nloading backbone (frozen)")
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval().to(device)
    embed_dim = getattr(backbone, "embed_dim", backbone.num_features)
    cfg = resolve_data_config({}, model=backbone)
    transform = create_transform(**cfg, is_training=False)
    print(f"  embed_dim={embed_dim}")

    # ── dataset ──
    print("\nloading dataset")
    kw = {}
    if dataset == "dsprites":
        kw["target"] = dsprites_target
    elif dataset == "funny_birds":
        kw["split"] = "train"
    elif dataset == "imagenette":
        kw["split"] = "train"
    ds = load_dataset(dataset, transform=transform, **kw)
    num_classes = ds.num_classes
    print(f"  {dataset}: {len(ds)} images, {num_classes} classes")

    # ── features ──
    print()
    feats, labels = cached_features(
        backbone, ds, cache, device=device,
        batch_size=extract_batch_size, num_workers=num_workers,
        force=force_extract,
    )

    # Backbone done — free GPU memory before Lightning takes over.
    del backbone
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── train/val split ──
    n = len(feats)
    n_val = max(1, int(round(n * val_frac)))
    full = TensorDataset(feats, labels)
    train_ds, val_ds = random_split(
        full, [n - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"\nsplit: train={len(train_ds)}, val={len(val_ds)}")

    # In-memory tensors → workers=0 is fine and avoids any IPC overhead.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Lightning training ──
    probe = LinearProbe(
        embed_dim=embed_dim, num_classes=num_classes,
        lr=lr, weight_decay=weight_decay,
    )

    ckpt_dir = out.parent / "_lightning_ckpts" / out.stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best",
        monitor="val_acc", mode="max", save_top_k=1, save_weights_only=True,
    )
    early_cb = EarlyStopping(
        monitor="val_acc", mode="max", patience=patience, verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto", devices=1,
        callbacks=[ckpt_cb, early_cb],
        log_every_n_steps=10,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(probe, train_loader, val_loader)

    # ── Restore best, save in walkthrough-compatible format ──
    print(f"\nbest checkpoint: {ckpt_cb.best_model_path}")
    probe = LinearProbe.load_from_checkpoint(ckpt_cb.best_model_path)
    metrics = trainer.validate(probe, val_loader, verbose=False)[0]
    val_acc = float(metrics["val_acc"])
    val_acc5 = float(metrics["val_acc5"])
    val_loss = float(metrics["val_loss"])

    payload = {
        "model_name": model_name,
        "num_classes": num_classes,
        "embed_dim": embed_dim,
        "dataset": dataset,
        "head_state_dict": probe.head.state_dict(),
        "val_acc": val_acc,
        "val_acc5": val_acc5,
        "val_loss": val_loss,
    }
    torch.save(payload, out)

    print(f"\nbest val_acc  = {val_acc:.4f}")
    print(f"     val_acc5 = {val_acc5:.4f}")
    print(f"     val_loss = {val_loss:.4f}")
    print(f"\nwrote probe to {out}")


if __name__ == "__main__":
    app()
