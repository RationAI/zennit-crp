"""Train a frozen-DINOv3 probe (linear or attentive) with PyTorch Lightning.

Two probe types:

* ``--probe-type linear`` *(default; walkthrough-compatible)* — single
  ``nn.Linear`` on the cls-token pre-logits. Cheap, ~50K params.
  Saves to ``data/<model>_probe_<dataset>.pt`` with key
  ``head_state_dict`` (loadable directly into a timm ``model.head``).
* ``--probe-type attentive`` *(SOTA recipe)* — DINOv2/v3-style attentive
  pooling: a learned query attends over the full token sequence
  (cls + register + patch) via :class:`nn.MultiheadAttention`,
  then ``LayerNorm`` + ``Linear``. ~4 M trainable params.
  Saves to ``data/<model>_attn_probe_<dataset>.pt`` with key
  ``attn_state_dict`` (full module state) and ``num_heads``.

Recipes (copy-paste ready):

FunnyBirds, linear probe (the walkthrough's default):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset funny_birds

FunnyBirds, attentive probe (best accuracy; ~20 GB token cache):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset funny_birds --probe-type attentive

dsprites:

    uv run python experiments/train_dinov3_probe.py \\
        --dataset dsprites [--probe-type attentive]

ImageNet-1k val (un-gated HF mirror, ~830 MB auto-DL):

    uv run python experiments/train_dinov3_probe.py \\
        --dataset imagenet_val_hf [--probe-type attentive]

Imagenette:

    uv run python experiments/train_dinov3_probe.py \\
        --dataset imagenette [--probe-type attentive]

Pipeline:

1. Load ``vit_large_patch16_dinov3`` (304 M params), freeze backbone.
2. One pass over the dataset: extract features and cache to disk.
   Linear probe → cls-token pre-logits, ``(N, D)``, fp32, ~200 MB.
   Attentive probe → full token sequence, ``(N, T, D)``, fp16, ~20 GB
   on funny_birds; mmap'd at load so RAM stays small.
3. Random 90/10 train/val split on cached features.
4. Lightning trains the head with AdamW + cross-entropy.
   ``ModelCheckpoint(monitor='val_acc')`` keeps the best epoch;
   ``EarlyStopping(patience=5)`` stops when val_acc plateaus.
5. Save best head + val metrics to the output ``.pt``.

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


# ── Lightning modules ─────────────────────────────────────────────────────────


class _ProbeBase(L.LightningModule):
    """Shared train/val step + metric tracking. Subclasses define `forward`
    and the trainable submodules."""

    def _build_metrics(self, num_classes: int) -> None:
        kw = dict(num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(**kw)
        self.val_acc = MulticlassAccuracy(**kw)
        self.val_acc5 = MulticlassAccuracy(top_k=5, **kw)

    def _step(self, batch, stage: str, acc, acc5=None):
        x, y = batch
        logits = self(x)
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
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class LinearProbe(_ProbeBase):
    """``nn.Linear`` head on the cls-token pre-logits.

    Cheap (~50K params for funny_birds), but ignores all spatial info —
    plateaus around 30–60 % top-1 on FunnyBirds because many classes share
    3 of 5 part variants and the global cls feature can't disambiguate.
    """

    def __init__(
        self, embed_dim: int, num_classes: int,
        lr: float = 1e-3, weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Linear(embed_dim, num_classes)
        self._build_metrics(num_classes)

    def forward(self, x):
        return self.head(x)


class AttentiveProbe(_ProbeBase):
    """Learned-query attention pooling over the full token sequence
    (cls + register + patch tokens), then ``LayerNorm`` + ``Linear``.

    This is the canonical 'attentive probe' from the DINOv2 / DINOv3 eval
    protocols (Oquab et al. 2024; Darcet et al. 2024). One learnable query
    attends over all tokens via :class:`nn.MultiheadAttention` — the
    classifier sees patch-level evidence, which is essential for the
    FunnyBirds part-combination task.

    Trainable params (ViT-L, num_heads=8): ~4.2 M (4 × 1024² MHA
    projections + LN + linear). Backbone stays frozen.
    """

    def __init__(
        self, embed_dim: int, num_classes: int, num_heads: int = 8,
        lr: float = 1e-3, weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._build_metrics(num_classes)

    def forward(self, tokens):
        # tokens: (B, T, D). Cached as fp16 — cast for stable attention.
        tokens = tokens.float()
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)            # (B, 1, D)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))        # (B, D)
        return self.head(pooled)


# ── Feature extraction (one-shot, cached) ────────────────────────────────────


@torch.no_grad()
def extract_features(
    backbone, dataset, *, mode: str, device: str, batch_size: int,
    num_workers: int, dtype: torch.dtype = torch.float32,
):
    """One pass: forward every image through the frozen backbone.

    mode='cls'  → ``(N, D)`` cls pre-logits (cheap; for LinearProbe).
    mode='full' → ``(N, T, D)`` full token sequence (cls + reg + patch);
                  for AttentiveProbe. Cache as fp16 to halve disk + RAM.
    """
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
        out = backbone.forward_features(x)             # (B, T, D)
        if mode == "cls":
            f = backbone.forward_head(out, pre_logits=True)  # (B, D)
        elif mode == "full":
            f = out                                     # (B, T, D)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        feats.append(f.to(dtype).cpu())
        labels.append(y if torch.is_tensor(y) else torch.as_tensor(y))
        seen += x.shape[0]
        if seen % (50 * batch_size) < batch_size:
            print(f"  extracted {seen}/{n}  ({time.time() - t0:.1f}s)")
    return torch.cat(feats, 0), torch.cat(labels, 0)


def cached_features(
    backbone, dataset, cache_path: Path, *, mode: str,
    device: str, batch_size: int, num_workers: int,
    dtype: torch.dtype = torch.float32, force: bool = False,
    mmap: bool = False,
):
    """Load or build the feature cache. ``mmap=True`` uses
    ``torch.load(..., mmap=True)`` so the (potentially multi-GB)
    full-token tensor stays on disk and is paged in lazily by the
    DataLoader."""
    if cache_path.exists() and not force:
        d = torch.load(cache_path, mmap=mmap)
        if d["feats"].shape[0] == len(dataset):
            print(f"loaded cached features from {cache_path}  (mmap={mmap})")
            return d["feats"], d["labels"]
        print(f"  cache size mismatch ({d['feats'].shape[0]} vs {len(dataset)}); re-extracting")
    print(f"extracting features for {len(dataset)} images (mode={mode})")
    feats, labels = extract_features(
        backbone, dataset, mode=mode, device=device,
        batch_size=batch_size, num_workers=num_workers, dtype=dtype,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": feats, "labels": labels}, cache_path)
    print(f"saved {cache_path}  (feats={tuple(feats.shape)} {feats.dtype})")
    return feats, labels


# ── CLI ──────────────────────────────────────────────────────────────────────


_DATASETS = ("funny_birds", "dsprites", "imagenet_val_hf", "imagenette")
_PROBE_TYPES = ("linear", "attentive")


@app.command()
def main(
    dataset: str = typer.Option(
        "funny_birds", "--dataset",
        help=f"Source dataset. One of: {', '.join(_DATASETS)}.",
    ),
    probe_type: str = typer.Option(
        "linear", "--probe-type",
        help=f"Head architecture. {{'linear', 'attentive'}}. "
             "'linear': nn.Linear on cls only — cheap, walkthrough-compatible. "
             "'attentive': MultiheadAttention pooling over the full token "
             "sequence + Linear (DINOv3-paper SOTA recipe; sees patch tokens).",
    ),
    num_heads: int = typer.Option(
        8, "--num-heads",
        help="Attentive probe: heads in the pooling MultiheadAttention. "
             "Ignored for --probe-type linear.",
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
             "data/<model>_probe_<dataset>.pt for linear, "
             "data/<model>_attn_probe_<dataset>.pt for attentive.",
    ),
    cache: Optional[Path] = typer.Option(
        None, "--cache",
        help="Feature cache .pt path. Default depends on probe type "
             "(<out_stem>_feats.pt for linear; <out_stem>_token_feats.pt "
             "for attentive — full token sequence, fp16, ~20 GB on "
             "funny_birds, mmap'd at load time).",
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
    """Train a frozen-DINOv3 probe (linear or attentive) with Lightning."""
    if dataset not in _DATASETS:
        raise typer.BadParameter(
            f"unknown dataset {dataset!r}; choose from {', '.join(_DATASETS)}"
        )
    if probe_type not in _PROBE_TYPES:
        raise typer.BadParameter(
            f"unknown probe-type {probe_type!r}; choose from {', '.join(_PROBE_TYPES)}"
        )
    L.seed_everything(seed, workers=True)

    # Probe-type-specific paths and feature mode.
    is_attentive = probe_type == "attentive"
    out_tag = "attn_probe" if is_attentive else "probe"
    cache_tag = "_token_feats" if is_attentive else "_feats"
    feat_mode = "full" if is_attentive else "cls"
    feat_dtype = torch.float16 if is_attentive else torch.float32

    if out is None:
        out = REPO_ROOT / "data" / f"{model_name}_{out_tag}_{dataset}.pt"
    if cache is None:
        cache = out.with_name(out.stem + cache_tag + ".pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device     : {device}")
    print(f"model      : {model_name}")
    print(f"dataset    : {dataset}")
    print(f"probe-type : {probe_type}" + (f" (heads={num_heads})" if is_attentive else ""))
    print(f"out        : {out}")
    print(f"cache      : {cache}")

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
        backbone, ds, cache, mode=feat_mode, dtype=feat_dtype,
        device=device, batch_size=extract_batch_size,
        num_workers=num_workers, force=force_extract,
        # Attentive cache is large (~20 GB on funny_birds) — mmap so RAM
        # stays small and the DataLoader pages slices in lazily.
        mmap=is_attentive,
    )

    # Backbone done — free GPU memory before Lightning takes over.
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── train/val split ──
    n = len(feats)
    n_val = max(1, int(round(n * val_frac)))
    full = TensorDataset(feats, labels)
    train_ds, val_ds = random_split(
        full, [n - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"\nsplit: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Lightning training ──
    if is_attentive:
        probe = AttentiveProbe(
            embed_dim=embed_dim, num_classes=num_classes,
            num_heads=num_heads, lr=lr, weight_decay=weight_decay,
        )
        ProbeCls = AttentiveProbe
    else:
        probe = LinearProbe(
            embed_dim=embed_dim, num_classes=num_classes,
            lr=lr, weight_decay=weight_decay,
        )
        ProbeCls = LinearProbe

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

    # ── Restore best epoch, save in a deserialisation-friendly format ──
    print(f"\nbest checkpoint: {ckpt_cb.best_model_path}")
    probe = ProbeCls.load_from_checkpoint(ckpt_cb.best_model_path)
    metrics = trainer.validate(probe, val_loader, verbose=False)[0]
    val_acc = float(metrics["val_acc"])
    val_acc5 = float(metrics["val_acc5"])
    val_loss = float(metrics["val_loss"])

    payload = {
        "model_name": model_name,
        "num_classes": num_classes,
        "embed_dim": embed_dim,
        "dataset": dataset,
        "probe_type": probe_type,
        "val_acc": val_acc,
        "val_acc5": val_acc5,
        "val_loss": val_loss,
    }
    if is_attentive:
        # Save the whole attentive head module — query, MHA, norm, linear.
        # Plus enough metadata to rebuild the AttentiveProbe class.
        payload["num_heads"] = num_heads
        payload["attn_state_dict"] = {
            k: v for k, v in probe.state_dict().items()
            if not k.startswith(("train_acc", "val_acc"))
        }
    else:
        # Linear head only — directly loadable into a timm `model.head`,
        # which is what the walkthrough notebook expects.
        payload["head_state_dict"] = probe.head.state_dict()
    torch.save(payload, out)

    print(f"\nbest val_acc  = {val_acc:.4f}")
    print(f"     val_acc5 = {val_acc5:.4f}")
    print(f"     val_loss = {val_loss:.4f}")
    print(f"\nwrote probe to {out}")


if __name__ == "__main__":
    app()
