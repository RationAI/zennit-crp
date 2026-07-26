"""Train a frozen-backbone probe (linear / attentive / block) with Lightning.

Two-step CLI: cache features once, then train any number of heads on
the cache without re-running the backbone.

Heads are pure-PyTorch ``nn.Module``s with vanilla forwards — no LRP
behaviour leaks into training. The walkthrough notebook applies the
AttnLRP composite at attribution time, which rebinds the head's
``BilinearMatmul`` / ``SoftmaxAlongLastDim`` / ``ScaleByConstant``
forwards to LRP-rule-aware variants and restores them on exit. So this
CLI has no ``--matmul-rule`` / ``--alpha`` / ``--beta`` flags — those
belong on the composite, not on the head.

Examples
--------

Cache cls features (cheap; needed for ``linear`` head)::

    uv run python experiments/train_probe.py cache vit_dinov3 funny_birds --kind cls

Cache full token features (~20 GB; needed for ``attentive`` and ``block`` heads)::

    uv run python experiments/train_probe.py cache vit_dinov3 funny_birds --kind tokens

Train heads on the cached features::

    uv run python experiments/train_probe.py train vit_dinov3 linear    funny_birds
    uv run python experiments/train_probe.py train vit_dinov3 attentive funny_birds --num-heads 8
    uv run python experiments/train_probe.py train vit_dinov3 block     funny_birds --num-heads 16 --mlp-ratio 4.0

The ``train`` command auto-loads the right cache for the head's
:attr:`~models.heads.base.Head.input_kind`. If the cache is missing it
prints the exact ``cache`` command to run.

Output ``.pt`` (after best-epoch selection by ``ModelCheckpoint``):

.. code-block:: text

    {
      "base": "vit_dinov3",                    # registry name; build_probe(...)
      "head": "linear" | "attentive" | "block",  # registry name
      "head_kwargs": {...},                    # architecture only, e.g. {"num_heads": 8}
      "head_state_dict": {...},                # head weights only — backbone is frozen
      "num_classes": int,
      "dataset": str,
      "val_acc": float, "val_acc5": float, "val_loss": float,
    }

The walkthrough notebook re-builds the model with
``build_probe(ckpt["base"], ckpt["head"], num_classes=ckpt["num_classes"],
head_kwargs=ckpt["head_kwargs"])`` and loads ``head_state_dict`` into
``probe.head``.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch.multiprocessing as _torch_mp
import torch.nn as nn
import torch.nn.functional as F
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandAugment,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    ToTensor,
)

from experiments.datasets import load as load_dataset
from experiments.models import BASES, HEADS, build_base, build_head, build_probe

# File-system sharing → DataLoader workers don't need /dev/shm
# (containers cap it at 64 MB).
_torch_mp.set_sharing_strategy("file_system")

# Cache extraction (frozen ViT-L forward) and head training (B×T×D matmuls)
# both bottleneck on fp32 matmul on the A100. TF32 keeps fp32 calling
# convention but uses Tensor Cores → ~3-5× faster. Cached features are
# stored at fixed precision (fp32 cls / fp16 tokens) regardless, so this
# does not affect numbers seen at attribution time.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"


# ── Training-dataset registry ────────────────────────────────────────────────
#
# One enum-style entry per named training dataset variant. Each entry maps
# the user-facing CLI choice to `(dataset_name, base_kwargs)` for
# :func:`experiments.datasets.load`. Dataset-specific overrides
# (e.g. dsprites target / n_per_class) come from CLI flags that apply only
# when the matching variant is selected.
#
# The choice string is also used as part of on-disk paths
# (`data/runs/finetune_<base>_<train-ds>/...`) so it must be filesystem-safe
# — kebab-case is fine.

TRAIN_DATASETS: dict[str, tuple[str, dict]] = {
    # FunnyBirds — split the train set by ablation status. The official
    # test split is always zero-ablation; we don't include it here because
    # you don't train on it.
    "funny-birds-train-clean":   ("funny_birds",   {"split": "train", "clean_only": True}),
    "funny-birds-train-full":    ("funny_birds",   {"split": "train", "clean_only": False}),
    # dSprites — default target=shape (3 classes); override via --dsprites-target.
    "dsprites":                  ("dsprites",      {"target": "shape"}),
    # ColoredMNIST — biased train split (0.99 colour↔digit correlation).
    "colored-mnist-train":       ("colored_mnist", {"split": "train"}),
    # Imagenette / ImageNet val — included so the cache & finetune paths
    # work uniformly on these too.
    "imagenette-train":          ("imagenette",    {"split": "train"}),
    "imagenet-val-hf":           ("imagenet_val_hf", {}),
}


# Default `--train-ds` choice per legacy dataset name (used by probe-resume
# flow where the original train-ds isn't recorded in the checkpoint).
_PROBE_TRAIN_DS_DEFAULTS: dict[str, str] = {
    "funny_birds":     "funny-birds-train-clean",
    "dsprites":        "dsprites",
    "colored_mnist":   "colored-mnist-train",
    "imagenette":      "imagenette-train",
    "imagenet_val_hf": "imagenet-val-hf",
}


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_train_ds(
    train_ds: str, *,
    dsprites_target: Optional[str] = None,
    dsprites_n_per_class: Optional[int] = None,
) -> tuple[str, dict]:
    """Look up the registry entry for ``train_ds`` and apply any
    dataset-specific overrides. Returns ``(dataset_name, kwargs)`` ready to
    pass to :func:`experiments.datasets.load`.

    Raises ``typer.BadParameter`` if ``train_ds`` is unknown or if a
    dataset-specific flag is set on the wrong dataset.
    """
    if train_ds not in TRAIN_DATASETS:
        raise typer.BadParameter(
            f"unknown --train-ds {train_ds!r}; choose from "
            f"{sorted(TRAIN_DATASETS)}"
        )
    dataset_name, base_kw = TRAIN_DATASETS[train_ds]
    kw = dict(base_kw)
    if dataset_name == "dsprites":
        if dsprites_target is not None:
            kw["target"] = dsprites_target
        if dsprites_n_per_class is not None:
            kw["n_per_class"] = dsprites_n_per_class
    else:
        if dsprites_target is not None or dsprites_n_per_class is not None:
            raise typer.BadParameter(
                f"--dsprites-* flags are only valid with --train-ds dsprites; "
                f"got --train-ds {train_ds!r}"
            )
    return dataset_name, kw


def cache_path(base: str, train_ds: str, kind: str) -> Path:
    return DATA_DIR / f"{base}_{train_ds}_{kind}_feats.pt"


def probe_path(base: str, head: str, train_ds: str) -> Path:
    return DATA_DIR / f"{base}_{head}_probe_{train_ds}.pt"


# ── cache command ────────────────────────────────────────────────────────────


@app.command("cache")
def cache_cmd(
    base: str = typer.Argument(
        ..., help=f"Base architecture. One of: {', '.join(BASES)}.",
    ),
    train_ds: str = typer.Argument(
        ..., help=f"Training-dataset choice. One of: {', '.join(sorted(TRAIN_DATASETS))}.",
    ),
    kind: str = typer.Option(
        "cls", "--kind",
        help="Feature kind to cache: 'cls' (~hundreds of MB; required by "
             "LinearHead — NOTE: timm applies the model's global_pool, so "
             "for DINOv3/Eva models with global_pool='avg' this is the "
             "patch MEAN-POOL, not the cls token), 'cls_token' (the actual "
             "post-norm CLS token, forward_features(x)[:, 0] — what the "
             "official DINOv3 linear probes use), or 'tokens' (full "
             "sequence, ~20 GB on funny_birds; required by AttentiveHead).",
    ),
    batch_size: int = typer.Option(
        32, "--batch-size",
        help="Backbone forward-pass batch size.",
    ),
    num_workers: int = typer.Option(
        0, "--num-workers",
        help="DataLoader workers. Use 0 in containers with tiny /dev/shm.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-extract even if cache exists.",
    ),
    dsprites_target: Optional[str] = typer.Option(
        None, "--dsprites-target",
        help="(--train-ds dsprites only) Override the latent factor used as "
             "the classification target.",
    ),
    dsprites_n_per_class: Optional[int] = typer.Option(
        None, "--dsprites-n-per-class",
        help="(--train-ds dsprites only) Subsample to this many images per class.",
    ),
):
    """Extract and cache features from a frozen base.

    Cls features are stored fp32 (small; ~200 MB on funny_birds);
    tokens are stored fp16 (~20 GB on funny_birds, mmap'd at load).
    """
    if base not in BASES:
        raise typer.BadParameter(f"unknown base {base!r}; choose from {sorted(BASES)}")
    if kind not in ("cls", "cls_token", "tokens"):
        raise typer.BadParameter(
            f"unknown kind {kind!r}; choose 'cls', 'cls_token' or 'tokens'")
    dataset_name, dataset_kwargs = _resolve_train_ds(
        train_ds, dsprites_target=dsprites_target,
        dsprites_n_per_class=dsprites_n_per_class,
    )

    out = cache_path(base, train_ds, kind)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        d = torch.load(out, map_location="cpu", mmap=True)
        print(f"cache exists at {out}  (feats={tuple(d['feats'].shape)} {d['feats'].dtype})")
        print("→ pass --force to re-extract")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device   : {device}")
    print(f"base     : {base}")
    print(f"train_ds : {train_ds}  ({dataset_name} {dataset_kwargs})")
    print(f"kind     : {kind}")
    print(f"out      : {out}")

    print("\nbuilding base (frozen)")
    base_obj = build_base(base).to(device)
    transform = base_obj.get_transform()
    normalize = base_obj.get_normalize()
    print(f"  embed_dim={base_obj.embed_dim}")

    print("\nloading dataset")
    ds = load_dataset(dataset_name, transform=transform, **dataset_kwargs)
    print(f"  {train_ds}: {len(ds)} images, {ds.num_classes} classes")

    # Dataset yields unnormalized [0,1] tensors. Normalize at the forward
    # boundary so the model sees its expected input distribution while
    # the dataset stays display-ready and reusable.
    if kind == "cls":
        _extract = base_obj.extract_cls
    elif kind == "cls_token":
        # True post-norm CLS token — bypasses forward_head/global_pool
        # (which is 'avg' on timm DINOv3/Eva, i.e. patch mean-pool).
        _extract = lambda x: base_obj.backbone.forward_features(x)[:, 0]
    else:
        _extract = base_obj.extract_tokens
    extract = lambda x: _extract(normalize(x))
    out_dtype = torch.float16 if kind == "tokens" else torch.float32

    print(f"\nextracting features ({kind}, dtype={out_dtype})", flush=True)
    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(ds, **loader_kwargs)

    # Pre-allocate output tensor: avoids the 2× RAM peak from list+torch.cat
    # at the end (matters for ``tokens`` on funny_birds — ~20 GB output, so
    # the cat-peak was ~40 GB and got OOM-killed on this 4-core / shared-RAM
    # box). Discover the per-sample feature shape with one tiny forward.
    print("  probing feature shape with one forward", flush=True)
    with torch.no_grad():
        probe_x, _ = next(iter(loader))
        probe_f = extract(probe_x[:1].to(device, non_blocking=True))
    per_sample_shape = tuple(probe_f.shape[1:])
    n = len(ds)
    feats = torch.empty((n, *per_sample_shape), dtype=out_dtype)
    labels = torch.empty((n,), dtype=torch.long)
    print(f"  pre-allocated feats {tuple(feats.shape)} ({feats.element_size() * feats.numel() / 1e9:.1f} GB)", flush=True)

    cursor, t0 = 0, time.time()
    # Cache extraction is one-shot — no autograd graph needed.
    # (Base.extract_* deliberately *don't* set no_grad themselves so the
    # AttnLRP composite can build its backward graph at attribution time.)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = extract(x).to(out_dtype).cpu()
            b = f.shape[0]
            feats[cursor:cursor + b] = f
            labels[cursor:cursor + b] = (
                y if torch.is_tensor(y) else torch.as_tensor(y)
            ).long()
            cursor += b
            if cursor % (50 * batch_size) < batch_size:
                rate = cursor / max(time.time() - t0, 1e-6)
                eta = (n - cursor) / max(rate, 1e-6)
                print(
                    f"  extracted {cursor}/{n}  "
                    f"({time.time() - t0:.0f}s, {rate:.1f} img/s, "
                    f"ETA {eta:.0f}s)",
                    flush=True,
                )

    print(f"\nsaving cache to {out}", flush=True)
    torch.save(
        {"feats": feats, "labels": labels,
         "base": base, "train_ds": train_ds,
         "dataset": dataset_name, "dataset_kwargs": dataset_kwargs,
         "kind": kind,
         "num_classes": int(ds.num_classes),
         "embed_dim": int(base_obj.embed_dim)},
        out,
    )
    print(f"saved {out}  (feats={tuple(feats.shape)} {feats.dtype})", flush=True)


# ── train command ────────────────────────────────────────────────────────────


class _ProbeLM(L.LightningModule):
    """Lightning wrapper around a head trained on cached features.

    The base is *not* part of this module — features are pre-extracted.
    The saved checkpoint contains only ``self.head``; the backbone is
    re-built from the registry at load time.
    """

    def __init__(
        self, head, num_classes: int, lr: float, weight_decay: float,
        scheduler: str = "none", warmup_epochs: int = 0, max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["head"])
        self.head = head
        kw = dict(num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(**kw)
        self.val_acc = MulticlassAccuracy(**kw)
        # top-5 only well-defined when num_classes >= 5; fall back to
        # top-1 for tiny-class datasets (dsprites: 3, etc.) so the
        # metric does not raise.
        self.val_acc5 = MulticlassAccuracy(top_k=min(5, num_classes), **kw)

    def _step(self, batch, stage, acc, acc5=None):
        x, y = batch
        logits = self.head(x)
        loss = F.cross_entropy(logits, y)
        acc(logits, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        if acc5 is not None:
            acc5(logits, y)
            self.log(f"{stage}_acc5", acc5, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train", self.train_acc)

    def validation_step(self, batch, _):
        return self._step(batch, "val", self.val_acc, self.val_acc5)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
        sched = self.hparams.scheduler
        if sched == "none":
            return opt
        # DINOv2/v3 linear-probe protocol: warmup → cosine annealing.
        # Warmup ramps lr 0 → lr over ``warmup_epochs``; cosine then decays
        # lr → 0 over the remaining epochs.
        if sched == "cosine":
            warmup = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if warmup > 0:
                from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
                warmup_sched = LinearLR(
                    opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup,
                )
                cosine_sched = CosineAnnealingLR(opt, T_max=max(1, total - warmup))
                lr_sched = SequentialLR(
                    opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup],
                )
            else:
                from torch.optim.lr_scheduler import CosineAnnealingLR
                lr_sched = CosineAnnealingLR(opt, T_max=total)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": lr_sched, "interval": "epoch"},
            }
        raise ValueError(f"unknown scheduler {sched!r}; choose 'none' or 'cosine'")


@app.command("train")
def train_cmd(
    base: str = typer.Argument(
        ..., help=f"Base architecture. One of: {', '.join(BASES)}.",
    ),
    head: str = typer.Argument(
        ..., help=f"Head architecture. One of: {', '.join(HEADS)}.",
    ),
    train_ds: str = typer.Argument(
        ..., help=f"Training-dataset choice. One of: {', '.join(sorted(TRAIN_DATASETS))}.",
    ),
    epochs: int = typer.Option(
        50, "--epochs", help="Max epochs (EarlyStopping usually stops earlier).",
    ),
    patience: int = typer.Option(
        5, "--patience", help="EarlyStopping patience on val_acc.",
    ),
    lr: float = typer.Option(1e-3, "--lr"),
    weight_decay: float = typer.Option(1e-2, "--weight-decay"),
    batch_size: int = typer.Option(
        256, "--batch-size", help="Probe (head) train batch size.",
    ),
    val_frac: float = typer.Option(
        0.1, "--val-frac", help="Held-out validation fraction.",
    ),
    num_heads: int = typer.Option(
        8, "--num-heads",
        help="(attentive/block) self-attention heads.",
    ),
    mlp_ratio: float = typer.Option(
        4.0, "--mlp-ratio",
        help="(block only) MLP hidden-dim multiplier (ViT default 4.0).",
    ),
    feature_kind: Optional[str] = typer.Option(
        None, "--feature-kind",
        help="Override the cached feature kind to train on (default: the "
             "head's input_kind). E.g. 'cls_token' trains a linear head on "
             "the true CLS token instead of timm's global_pool ('avg' → "
             "patch mean-pool on DINOv3/Eva). Must be shape-compatible "
             "with the head ((N, D) kinds for 'linear').",
    ),
    scheduler: str = typer.Option(
        "none", "--scheduler",
        help="LR schedule. 'none' (default — constant LR) or 'cosine' "
             "(warmup → cosine annealing, the DINOv2/v3 probe protocol).",
    ),
    cosine_warmup_epochs: int = typer.Option(
        5, "--cosine-warmup-epochs",
        help="(--scheduler cosine only) Linear warmup epochs from lr×1e-6 to lr.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="Output .pt path. Default: data/<base>_<head>_probe_<train-ds>.pt",
    ),
    seed: int = typer.Option(0, "--seed"),
):
    """Train a head on cached features.

    Auto-loads the cache that matches the head's ``input_kind`` (``cls``
    for ``linear``, ``tokens`` for ``attentive``/``block``).
    """
    if base not in BASES:
        raise typer.BadParameter(f"unknown base {base!r}; choose from {sorted(BASES)}")
    if head not in HEADS:
        raise typer.BadParameter(f"unknown head {head!r}; choose from {sorted(HEADS)}")
    if train_ds not in TRAIN_DATASETS:
        raise typer.BadParameter(
            f"unknown --train-ds {train_ds!r}; choose from {sorted(TRAIN_DATASETS)}"
        )

    L.seed_everything(seed, workers=True)

    head_cls = HEADS[head]
    kind = head_cls.input_kind if feature_kind is None else feature_kind
    if feature_kind is not None:
        flat_kinds = ("cls", "cls_token")
        if head_cls.input_kind in flat_kinds and kind not in flat_kinds:
            raise typer.BadParameter(
                f"--feature-kind {feature_kind!r} is not shape-compatible "
                f"with head {head!r} (needs one of {flat_kinds})")
    cache = cache_path(base, train_ds, kind)
    if not cache.exists():
        raise typer.Exit(
            f"\nMissing feature cache at {cache}. Run:\n\n"
            f"    uv run python -m experiments.train_probe cache "
            f"{base} {train_ds} --kind {kind}\n"
        )

    if out is None:
        out = probe_path(base, head, train_ds)
        if feature_kind is not None:
            # Don't collide with the default-kind probe file.
            out = out.with_name(f"{out.stem}_{kind}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"base     : {base}")
    head_info = ""
    if head == "attentive":
        head_info = f" (num_heads={num_heads})"
    elif head == "block":
        head_info = f" (num_heads={num_heads}, mlp_ratio={mlp_ratio})"
    print(f"head     : {head}{head_info}")
    print(f"train_ds : {train_ds}")
    print(f"cache    : {cache}")
    print(f"out      : {out}")

    print(f"\nloading cache (mmap={kind == 'tokens'})")
    d = torch.load(cache, map_location="cpu", mmap=(kind == "tokens"))
    feats, labels = d["feats"], d["labels"]
    embed_dim = int(d["embed_dim"])
    num_classes = int(d["num_classes"])
    print(f"  feats={tuple(feats.shape)} {feats.dtype}, labels={tuple(labels.shape)}")

    # Head construction kwargs — architecture only. The checkpoint
    # records these so the walkthrough notebook can reconstruct the
    # head with identical layout. LRP rules are NOT a constructor
    # concern — they're applied at attribution time by the composite's
    # per-rule canonizers (see :mod:`zennit_ext`), which
    # rebind the head's ``BilinearMatmul`` / ``SoftmaxAlongLastDim`` /
    # ``ScaleByConstant`` forwards inside the composite context.
    head_kwargs: dict = {}
    if head == "attentive":
        head_kwargs.update(num_heads=num_heads)
    elif head == "block":
        head_kwargs.update(num_heads=num_heads, mlp_ratio=mlp_ratio)
    head_obj = build_head(
        head, embed_dim=embed_dim, num_classes=num_classes, head_kwargs=head_kwargs,
    )

    n = len(feats)
    n_val = max(1, int(round(n * val_frac)))
    full = TensorDataset(feats, labels)
    train_split, val_split = random_split(
        full, [n - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"\nsplit: train={len(train_split)}, val={len(val_split)}")

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)

    lm = _ProbeLM(
        head_obj, num_classes=num_classes,
        lr=lr, weight_decay=weight_decay,
        scheduler=scheduler, warmup_epochs=cosine_warmup_epochs, max_epochs=epochs,
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
    trainer.fit(lm, train_loader, val_loader)

    print(f"\nbest checkpoint: {ckpt_cb.best_model_path}")
    lm = _ProbeLM.load_from_checkpoint(ckpt_cb.best_model_path, head=head_obj)
    metrics = trainer.validate(lm, val_loader, verbose=False)[0]

    payload = {
        "base": base,
        "head": head,
        "head_kwargs": head_kwargs,
        "num_classes": num_classes,
        "embed_dim": embed_dim,
        "train_ds": train_ds,
        "dataset": d.get("dataset"),       # bare dataset name from cache
        "dataset_kwargs": d.get("dataset_kwargs"),
        "feature_kind": kind,              # which cached features the head saw
        "head_state_dict": lm.head.state_dict(),
        "val_acc": float(metrics["val_acc"]),
        "val_acc5": float(metrics["val_acc5"]),
        "val_loss": float(metrics["val_loss"]),
    }
    torch.save(payload, out)

    print(f"\nbest val_acc  = {payload['val_acc']:.4f}")
    print(f"     val_acc5 = {payload['val_acc5']:.4f}")
    print(f"     val_loss = {payload['val_loss']:.4f}")
    print(f"\nwrote probe to {out}")


# ── finetune command ────────────────────────────────────────────────────────
#
# Last-resort path past the frozen-backbone probe ceiling: load a trained
# probe checkpoint and continue training with the backbone unfrozen and a
# very small backbone LR. Trades the frozen-backbone invariant the LRP
# composite is built around for a few extra percentage points of accuracy.


def _llrd_param_groups(
    model, *, backbone_lr: float, head_lr: float, weight_decay: float,
    llrd: float,
):
    """Build AdamW param groups with **layer-wise LR decay** on the
    backbone — earlier blocks get lower LR (LR scaled by ``llrd**depth``,
    counted from the *output* end). Standard fine-tune trick: lets the
    head and late blocks adapt fast while keeping early features stable.

    Returns ``[{params, lr, weight_decay}, ...]``. With ``llrd=1.0`` this
    collapses to two groups (backbone, head) — the legacy behaviour.
    """
    backbone = model.backbone
    if llrd >= 1.0 or not hasattr(backbone, "blocks"):
        return [
            {"params": list(backbone.parameters()), "lr": backbone_lr,
             "weight_decay": weight_decay},
            {"params": list(model.head.parameters()), "lr": head_lr,
             "weight_decay": weight_decay},
        ]

    n_blocks = len(backbone.blocks)
    block_param_ids = {
        id(p) for blk in backbone.blocks for p in blk.parameters()
    }
    # Patch embed + cls_token + pos_embed + register tokens — treat as
    # "layer 0" (deepest decay).
    early = [p for p in backbone.parameters() if id(p) not in block_param_ids
             and not _is_final_norm_param(backbone, p)]
    final_norm = [p for p in backbone.parameters() if id(p) not in block_param_ids
                  and _is_final_norm_param(backbone, p)]

    groups = []
    if early:
        groups.append({"params": early, "lr": backbone_lr * (llrd ** n_blocks),
                       "weight_decay": weight_decay})
    for i, blk in enumerate(backbone.blocks):
        depth_from_output = n_blocks - 1 - i
        groups.append({"params": list(blk.parameters()),
                       "lr": backbone_lr * (llrd ** depth_from_output),
                       "weight_decay": weight_decay})
    if final_norm:
        groups.append({"params": final_norm, "lr": backbone_lr,
                       "weight_decay": weight_decay})
    groups.append({"params": list(model.head.parameters()), "lr": head_lr,
                   "weight_decay": weight_decay})
    return groups


def _is_final_norm_param(backbone, p):
    """Detect post-block norm / fc_norm parameters — these sit at the
    output end and stay at full backbone_lr (no decay)."""
    for name in ("norm", "fc_norm"):
        mod = getattr(backbone, name, None)
        if mod is not None and any(p is q for q in mod.parameters()):
            return True
    return False


def _mixup_or_cutmix(x: torch.Tensor, y: torch.Tensor, mixup_alpha: float,
                     cutmix_alpha: float):
    """Draw one of mixup / cutmix (50/50 if both enabled) and apply to
    the batch in-place. Returns ``(x, y_a, y_b, lam)``; loss is then
    ``lam * CE(logits, y_a) + (1 - lam) * CE(logits, y_b)``.
    """
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha == 0 or torch.rand(()) < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if use_cutmix:
        # Random box, area = (1 - lam) of image.
        _, _, H, W = x.shape
        cut_rat = (1.0 - lam) ** 0.5
        cw, ch = int(W * cut_rat), int(H * cut_rat)
        cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        # Re-derive lam from the actual box area.
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        x = lam * x + (1.0 - lam) * x[perm]
    return x, y_a, y_b, lam


class _FinetuneLM(L.LightningModule):
    """Lightning wrapper for end-to-end fine-tuning of a base+head probe.

    Param groups: optional layer-wise LR decay on the backbone + a
    separate head group. Optional cosine schedule with warmup. Optional
    mixup / cutmix and label smoothing applied at the loss level. Runs
    the backbone live every batch — no cache — so the head sees fresh
    augmented views each epoch.
    """

    def __init__(
        self, model, num_classes: int,
        backbone_lr: float, head_lr: float, weight_decay: float,
        scheduler: str = "none", warmup_epochs: int = 0, max_epochs: int = 50,
        llrd: float = 1.0,
        mixup: float = 0.0, cutmix: float = 0.0, label_smoothing: float = 0.0,
        onecycle_pct_start: float = 0.1,
        onecycle_div_factor: float = 25.0,
        onecycle_final_div_factor: float = 1e4,
        normalize=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "normalize"])
        self.model = model
        # Datasets emit unnormalized [0,1] tensors. Apply the per-batch
        # normalize at the forward boundary (canonical Lightning pattern).
        # Defaults to identity if not provided.
        self.normalize = normalize if normalize is not None else (lambda t: t)
        kw = dict(num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(**kw)
        self.val_acc = MulticlassAccuracy(**kw)
        # top-5 only well-defined when num_classes >= 5; fall back to
        # top-1 for tiny-class datasets (dsprites: 3, etc.) so the
        # metric does not raise.
        self.val_acc5 = MulticlassAccuracy(top_k=min(5, num_classes), **kw)

    def _eval_step(self, batch, stage, acc, acc5=None):
        x, y = batch
        logits = self.model(self.normalize(x))
        loss = F.cross_entropy(logits, y, label_smoothing=self.hparams.label_smoothing)
        acc(logits, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        if acc5 is not None:
            acc5(logits, y)
            self.log(f"{stage}_acc5", acc5, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        x, y = batch
        x = self.normalize(x)
        if self.hparams.mixup > 0 or self.hparams.cutmix > 0:
            x, y_a, y_b, lam = _mixup_or_cutmix(
                x, y, self.hparams.mixup, self.hparams.cutmix,
            )
            logits = self.model(x)
            loss = (lam * F.cross_entropy(logits, y_a, label_smoothing=self.hparams.label_smoothing)
                    + (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=self.hparams.label_smoothing))
            # Train accuracy under mixup is meaningless; log against y_a as a proxy.
            self.train_acc(logits, y_a)
        else:
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=self.hparams.label_smoothing)
            self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        return self._eval_step(batch, "val", self.val_acc, self.val_acc5)

    def configure_optimizers(self):
        param_groups = _llrd_param_groups(
            self.model, backbone_lr=self.hparams.backbone_lr,
            head_lr=self.hparams.head_lr, weight_decay=self.hparams.weight_decay,
            llrd=self.hparams.llrd,
        )
        opt = torch.optim.AdamW(param_groups)

        if self.hparams.scheduler == "none":
            return opt
        if self.hparams.scheduler == "cosine":
            warmup = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR, LinearLR, SequentialLR,
            )
            if warmup > 0:
                lr_sched = SequentialLR(
                    opt,
                    schedulers=[
                        LinearLR(opt, start_factor=1e-6, end_factor=1.0,
                                 total_iters=warmup),
                        CosineAnnealingLR(opt, T_max=max(1, total - warmup)),
                    ],
                    milestones=[warmup],
                )
            else:
                lr_sched = CosineAnnealingLR(opt, T_max=total)
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": lr_sched, "interval": "epoch"}}
        if self.hparams.scheduler == "onecycle":
            # Smith's SuperConvergence (1cycle). Each param group's
            # `lr` set above becomes the *peak*; OneCycleLR ramps from
            # peak / div_factor up to peak (pct_start of total steps),
            # then anneals down to peak / final_div_factor.
            # Steps per step (not per epoch) — total_steps comes from
            # the Lightning trainer via dataloader length × epochs.
            from torch.optim.lr_scheduler import OneCycleLR
            total_steps = self.trainer.estimated_stepping_batches
            max_lrs = [g["lr"] for g in opt.param_groups]
            lr_sched = OneCycleLR(
                opt,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self.hparams.onecycle_pct_start,
                anneal_strategy="cos",
                div_factor=self.hparams.onecycle_div_factor,
                final_div_factor=self.hparams.onecycle_final_div_factor,
                three_phase=False,
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}
        raise ValueError(f"unknown scheduler {self.hparams.scheduler!r}; choose 'none', 'cosine', or 'onecycle'")


def _make_train_transform(base_obj, *, randaugment: bool = False,
                          colorjitter_hue: float = 0.0):
    """Augmented training transform: optional ``RandAugment`` + optional
    hue jitter + random crop + flip + rotation + ``ToTensor``. No
    normalize — Lightning's ``_FinetuneLM`` applies normalize at the
    forward boundary so the dataset stays uniform with the eval transform.

    Crop scale leaves at least 70% of the image visible; rotation is
    small (±15°). RandAugment composes 2 random ops at magnitude 9 —
    the timm/AugReg default for ImageNet ViTs. ``colorjitter_hue=0.5``
    randomizes hue across the full circle — used for ColoredMNIST to
    break the colour↔digit shortcut so the model has to learn shape.
    """
    cfg = resolve_data_config({}, model=base_obj.backbone)
    size = cfg["input_size"][-1]
    ops = [
        RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=3),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
    ]
    if colorjitter_hue > 0:
        ops.append(ColorJitter(hue=colorjitter_hue))
    if randaugment:
        # RandAugment runs on PIL images; insert before ToTensor.
        ops.append(RandAugment(num_ops=2, magnitude=9))
    ops.append(ToTensor())
    return Compose(ops)


def _auto_run_dir(name: str) -> Path:
    """Return ``data/runs/<name>/<UTC timestamp>/`` (created)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out = RUNS_DIR / name / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


@app.command("finetune")
def finetune_cmd(
    probe: Optional[Path] = typer.Argument(
        None, help="Path to a probe checkpoint produced by `train` "
                  "(e.g. data/vit_dinov3_attentive_probe_funny_birds.pt). "
                  "Omit and pass --from-scratch to start fresh.",
    ),
    from_scratch: bool = typer.Option(
        False, "--from-scratch",
        help="Skip probe-checkpoint loading; build a fresh head and "
             "fine-tune end-to-end. Requires --base and --train-ds; "
             "--head defaults to 'linear'. Used for vit_small full "
             "fine-tune where there's no prior probe run.",
    ),
    base: Optional[str] = typer.Option(
        None, "--base",
        help=f"(--from-scratch only) Base architecture. One of: {', '.join(BASES)}.",
    ),
    head: str = typer.Option(
        "linear", "--head",
        help=f"(--from-scratch only) Head architecture. One of: {', '.join(HEADS)}.",
    ),
    train_ds: Optional[str] = typer.Option(
        None, "--train-ds",
        help="(--from-scratch only) Training-dataset choice. One of: "
             f"{', '.join(sorted(TRAIN_DATASETS))}. "
             "Encodes both dataset name and variant (clean vs full, etc.).",
    ),
    epochs: int = typer.Option(
        15, "--epochs", help="Max epochs (each epoch is a full backbone forward+backward pass).",
    ),
    patience: int = typer.Option(
        5, "--patience", help="EarlyStopping patience on val_acc.",
    ),
    backbone_lr: float = typer.Option(
        1e-5, "--backbone-lr",
        help="Very small LR for the backbone — large enough to nudge "
             "features toward the task, small enough not to destroy "
             "the pretrained representation.",
    ),
    head_lr: float = typer.Option(
        1e-4, "--head-lr",
        help="LR for the head (~10× backbone-lr). Re-tunes the head "
             "as the backbone moves under it.",
    ),
    weight_decay: float = typer.Option(1e-2, "--weight-decay"),
    batch_size: int = typer.Option(
        8, "--batch-size",
        help="Live backbone forward+backward — keep small to fit GPU memory.",
    ),
    accumulate_grad_batches: int = typer.Option(
        4, "--accumulate-grad-batches",
        help="Effective batch size = batch_size × accumulate_grad_batches.",
    ),
    val_frac: float = typer.Option(0.1, "--val-frac"),
    num_workers: int = typer.Option(
        2, "--num-workers",
        help="DataLoader workers. Use 2 on the 4-core box.",
    ),
    precision: str = typer.Option(
        "bf16-mixed", "--precision",
        help="Mixed precision for the backbone forward+backward. "
             "Use '32-true' to force fp32 for debugging.",
    ),
    augment: bool = typer.Option(
        True, "--augment/--no-augment",
        help="Enable training-time augmentation (RRC + hflip + rotation).",
    ),
    randaugment: bool = typer.Option(
        False, "--randaugment",
        help="Stack RandAugment on top of the basic geometric augs.",
    ),
    colorjitter_hue: float = typer.Option(
        0.0, "--colorjitter-hue",
        help="ColorJitter hue range (0 = off, 0.5 = full circle). For "
             "ColoredMNIST set this to 0.5 to randomise hue at train "
             "time and break the colour↔digit shortcut; without it, "
             "the model latches onto colour and test acc collapses.",
    ),
    mixup: float = typer.Option(
        0.0, "--mixup",
        help="Mixup α (Beta distribution). 0 = off; 0.8 = ImageNet ViT default.",
    ),
    cutmix: float = typer.Option(
        0.0, "--cutmix",
        help="CutMix α. 0 = off; if both mixup and cutmix are non-zero "
             "one is chosen 50/50 per batch.",
    ),
    label_smoothing: float = typer.Option(
        0.0, "--label-smoothing",
        help="Label smoothing ε passed to cross-entropy.",
    ),
    layerwise_lr_decay: float = typer.Option(
        1.0, "--layerwise-lr-decay",
        help="Per-block backbone LR decay rate (was: --llrd). Backbone "
             "block i (counting from input) gets lr × rate^(depth-1-i). "
             "1.0 = off; 0.65–0.7 = common ImageNet ViT fine-tune setting.",
    ),
    scheduler: str = typer.Option(
        "none", "--scheduler",
        help="LR schedule: 'none', 'cosine' (warmup → cosine), or "
             "'onecycle' (Smith SuperConvergence — peak then anneal).",
    ),
    onecycle_pct_start: float = typer.Option(
        0.1, "--onecycle-pct-start",
        help="(--scheduler onecycle) Fraction of total steps spent ramping "
             "to peak LR. Smith default 0.3; 0.1 is fine for short fine-tunes.",
    ),
    onecycle_div_factor: float = typer.Option(
        25.0, "--onecycle-div-factor",
        help="(--scheduler onecycle) initial_lr = peak / div_factor.",
    ),
    onecycle_final_div_factor: float = typer.Option(
        1e4, "--onecycle-final-div-factor",
        help="(--scheduler onecycle) final_lr = initial_lr / final_div_factor.",
    ),
    cosine_warmup_epochs: int = typer.Option(
        0, "--cosine-warmup-epochs",
        help="(--scheduler cosine) Linear warmup epochs from lr×1e-6 to lr.",
    ),
    dsprites_target: Optional[str] = typer.Option(
        None, "--dsprites-target",
        help="(--train-ds dsprites only) Override the latent factor used "
             "as label. Default = 'shape' (3-class).",
    ),
    dsprites_n_per_class: Optional[int] = typer.Option(
        None, "--dsprites-n-per-class",
        help="(--train-ds dsprites only) Subsample to this many images "
             "per class. Default = full 737 k. Use ~5000-10000 for fast "
             "iteration; the 3-class shape task is trivial enough that "
             "30 k images suffice for > 99 % val_acc.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir",
        help="Run directory for best.pt + config.json + metrics.csv. "
             "Default: data/runs/finetune_<base>_<train-ds>/<UTC ts>/.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="(legacy probe-resume) Output .pt path. If --output-dir is "
             "set, this is ignored.",
    ),
    seed: int = typer.Option(0, "--seed"),
):
    """Fine-tune a base+head model end-to-end (backbone unfrozen).

    Two flows:

    * **probe-resume** — pass a probe checkpoint as the first arg; the
      backbone is unfrozen and the trained head continues training. Used
      to push DINOv3 probes past their frozen-backbone ceiling.
    * **--from-scratch** — pass ``--base`` and ``--train-ds`` (and
      optionally ``--head``); a fresh head is built and the whole stack
      is fine-tuned. Used for the vit_small runs. Validated recipe on
      FunnyBirds:

        ``--scheduler onecycle --onecycle-pct-start 0.1
        --backbone-lr 5e-4 --head-lr 5e-3 --layerwise-lr-decay 0.7
        --randaugment --label-smoothing 0.1``

      hits test top-1 0.984 in 25 epochs.

    Always writes ``best.pt`` (model + metadata), ``config.json`` (the
    full Typer params), and ``metrics.csv`` (per-epoch logs) into the
    run directory.
    """
    L.seed_everything(seed, workers=True)

    if from_scratch:
        if base is None or train_ds is None:
            raise typer.BadParameter("--from-scratch requires --base and --train-ds")
        if base not in BASES:
            raise typer.BadParameter(f"unknown base {base!r}; choose from {sorted(BASES)}")
        if head not in HEADS:
            raise typer.BadParameter(f"unknown head {head!r}; choose from {sorted(HEADS)}")
        dataset_name, dataset_kwargs = _resolve_train_ds(
            train_ds, dsprites_target=dsprites_target,
            dsprites_n_per_class=dsprites_n_per_class,
        )
        base_name, head_name = base, head
        head_kwargs: dict = {}
        ds_probe = load_dataset(dataset_name, transform=None, **dataset_kwargs)
        num_classes = int(ds_probe.num_classes)
        ckpt = None
        print(f"from-scratch fine-tune: base={base_name}  head={head_name}  "
              f"train_ds={train_ds}  num_classes={num_classes}", flush=True)
    else:
        if probe is None:
            raise typer.BadParameter("missing probe checkpoint (or pass --from-scratch)")
        print(f"loading probe checkpoint: {probe}", flush=True)
        ckpt = torch.load(probe, map_location="cpu", weights_only=False)
        base_name = ckpt["base"]
        head_name = ckpt["head"]
        head_kwargs = ckpt.get("head_kwargs", {})
        num_classes = int(ckpt["num_classes"])
        # Resolve train_ds from the probe checkpoint: prefer the explicit
        # field (new), fall back to the legacy `dataset` + default-variant
        # lookup (old probe checkpoints have only the bare dataset name).
        ckpt_train_ds = ckpt.get("train_ds")
        if ckpt_train_ds is not None and ckpt_train_ds in TRAIN_DATASETS:
            train_ds = ckpt_train_ds
        else:
            legacy_dataset = ckpt.get("dataset")
            train_ds = _PROBE_TRAIN_DS_DEFAULTS.get(legacy_dataset)
            if train_ds is None:
                raise RuntimeError(
                    f"probe ckpt {probe} has no recognisable train-ds: "
                    f"train_ds={ckpt_train_ds!r}, dataset={legacy_dataset!r}"
                )
        dataset_name, dataset_kwargs = _resolve_train_ds(
            train_ds, dsprites_target=dsprites_target,
            dsprites_n_per_class=dsprites_n_per_class,
        )
        print(f"  base={base_name}  head={head_name}  head_kwargs={head_kwargs}")
        print(f"  num_classes={num_classes}  train_ds={train_ds}  ({dataset_name} {dataset_kwargs})")
        print(f"  starting val_acc (frozen backbone): {ckpt.get('val_acc', '?'):.4f}")

    # Resolve run dir (timestamped) — single source of truth for outputs.
    if output_dir is None:
        if out is not None:
            # Legacy: place artifacts next to --out.
            run_dir = out.parent
            run_dir.mkdir(parents=True, exist_ok=True)
            best_path = out
        else:
            run_dir = _auto_run_dir(f"finetune_{base_name}_{train_ds}")
            best_path = run_dir / "best.pt"
    else:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "best.pt"
    print(f"  run dir: {run_dir}")
    print(f"  best  →  {best_path}")

    # Persist the full Typer config for reproducibility.
    cfg = {
        "base": base_name, "head": head_name,
        "train_ds": train_ds, "dataset": dataset_name,
        "dataset_kwargs": dataset_kwargs,
        "num_classes": num_classes, "epochs": epochs, "patience": patience,
        "backbone_lr": backbone_lr, "head_lr": head_lr,
        "weight_decay": weight_decay, "batch_size": batch_size,
        "accumulate_grad_batches": accumulate_grad_batches,
        "val_frac": val_frac, "precision": precision, "augment": augment,
        "randaugment": randaugment, "colorjitter_hue": colorjitter_hue,
        "mixup": mixup, "cutmix": cutmix,
        "label_smoothing": label_smoothing,
        "layerwise_lr_decay": layerwise_lr_decay,
        "scheduler": scheduler, "cosine_warmup_epochs": cosine_warmup_epochs,
        "onecycle_pct_start": onecycle_pct_start,
        "onecycle_div_factor": onecycle_div_factor,
        "onecycle_final_div_factor": onecycle_final_div_factor,
        "from_scratch": from_scratch, "probe": str(probe) if probe else None,
        "dsprites_target": dsprites_target,
        "dsprites_n_per_class": dsprites_n_per_class,
        "seed": seed,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"\nbuilding model: {base_name} + {head_name}", flush=True)
    model = build_probe(
        base_name, head_name,
        num_classes=num_classes, head_kwargs=head_kwargs,
    )
    if ckpt is not None:
        model.head.load_state_dict(ckpt["head_state_dict"])
    # Unfreeze backbone — train mode + requires_grad.
    for p in model.backbone.parameters():
        p.requires_grad_(True)
    model.backbone.train()
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"  backbone params: {n_backbone:,}")
    print(f"  head params    : {n_head:,}")

    # Datasets — train uses augmentation; val uses the deterministic timm transform.
    print("\nloading dataset", flush=True)
    base_obj = build_base(base_name)
    val_tfm = base_obj.get_transform()
    train_tfm = (
        _make_train_transform(
            base_obj, randaugment=randaugment, colorjitter_hue=colorjitter_hue,
        )
        if augment else val_tfm
    )

    ds_train = load_dataset(dataset_name, transform=train_tfm, **dataset_kwargs)
    ds_val   = load_dataset(dataset_name, transform=val_tfm,   **dataset_kwargs)
    n = len(ds_train)
    n_val = max(1, int(round(n * val_frac)))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    # NB: keep the `train_ds` *string* intact — it goes into the best.pt
    # payload below. (Previously this line shadowed it with the Subset
    # object, so checkpoints pickled the whole dataset into "train_ds".)
    train_subset = Subset(ds_train, train_idx)
    val_ds = Subset(ds_val, val_idx)
    print(f"  split: train={len(train_subset)}, val={len(val_ds)}")
    print(f"  train transform: {train_tfm}")

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

    lm = _FinetuneLM(
        model, num_classes=num_classes,
        backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay,
        scheduler=scheduler, warmup_epochs=cosine_warmup_epochs, max_epochs=epochs,
        llrd=layerwise_lr_decay,
        mixup=mixup, cutmix=cutmix, label_smoothing=label_smoothing,
        onecycle_pct_start=onecycle_pct_start,
        onecycle_div_factor=onecycle_div_factor,
        onecycle_final_div_factor=onecycle_final_div_factor,
        normalize=base_obj.get_normalize(),
    )

    ckpt_dir = run_dir / "_lightning_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best",
        monitor="val_acc", mode="max", save_top_k=1, save_weights_only=True,
    )
    early_cb = EarlyStopping(
        monitor="val_acc", mode="max", patience=patience, verbose=True,
    )
    csv_logger = CSVLogger(save_dir=str(run_dir), name="", version="")

    print(f"\nstarting fine-tune: {epochs} max epochs, "
          f"backbone_lr={backbone_lr}, head_lr={head_lr}, "
          f"layerwise_lr_decay={layerwise_lr_decay}, "
          f"sched={scheduler}{f'+warmup{cosine_warmup_epochs}' if scheduler == 'cosine' else ''}, "
          f"mixup={mixup}, cutmix={cutmix}, label_smoothing={label_smoothing}, "
          f"randaugment={randaugment}, "
          f"effective_bs={batch_size}×{accumulate_grad_batches}={batch_size * accumulate_grad_batches}, "
          f"precision={precision}", flush=True)

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto", devices=1,
        callbacks=[ckpt_cb, early_cb],
        logger=csv_logger,
        log_every_n_steps=10,
        enable_model_summary=False,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        # Note: deterministic=True conflicts with some bf16 ops; rely on seed_everything alone.
    )
    trainer.fit(lm, train_loader, val_loader)

    print(f"\nbest checkpoint: {ckpt_cb.best_model_path}", flush=True)
    lm = _FinetuneLM.load_from_checkpoint(ckpt_cb.best_model_path, model=model)
    metrics = trainer.validate(lm, val_loader, verbose=False)[0]

    payload = {
        "base": base_name,
        "head": head_name,
        "head_kwargs": head_kwargs,
        "num_classes": num_classes,
        "embed_dim": int(model.backbone.embed_dim),
        "train_ds": train_ds,
        "dataset": dataset_name,
        "dataset_kwargs": dataset_kwargs,
        # Both backbone and head weights — the backbone is no longer the
        # stock pretrained one after fine-tuning, so we save it fully.
        "backbone_state_dict": lm.model.backbone.state_dict(),
        "head_state_dict": lm.model.head.state_dict(),
        "val_acc": float(metrics["val_acc"]),
        "val_acc5": float(metrics["val_acc5"]),
        "val_loss": float(metrics["val_loss"]),
        "finetuned_from": str(probe) if probe else None,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "config": cfg,
    }
    torch.save(payload, best_path)

    print(f"\nbest val_acc  = {payload['val_acc']:.4f}", flush=True)
    print(f"     val_acc5 = {payload['val_acc5']:.4f}", flush=True)
    print(f"     val_loss = {payload['val_loss']:.4f}", flush=True)
    print(f"\nwrote fine-tuned model to {best_path}", flush=True)


if __name__ == "__main__":
    app()
