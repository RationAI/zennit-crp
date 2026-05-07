"""Train a frozen-backbone probe (linear or attentive) with Lightning.

Two-step CLI: cache features once, then train any number of heads on
the cache without re-running the backbone.

Examples
--------

Cache cls features (cheap; needed for ``linear`` head)::

    uv run python experiments/train_probe.py cache vit_dinov3 funny_birds --kind cls

Cache full token features (~20 GB; needed for ``attentive`` head)::

    uv run python experiments/train_probe.py cache vit_dinov3 funny_birds --kind tokens

Train heads on the cached features::

    uv run python experiments/train_probe.py train vit_dinov3 linear    funny_birds
    uv run python experiments/train_probe.py train vit_dinov3 attentive funny_birds --num-heads 8

The ``train`` command auto-loads the right cache for the head's
:attr:`~models.heads.base.Head.input_kind`. If the cache is missing it
prints the exact ``cache`` command to run.

Output ``.pt`` (after best-epoch selection by ``ModelCheckpoint``):

.. code-block:: text

    {
      "base": "vit_dinov3",            # registry name; build_probe(...)
      "head": "linear" | "attentive",  # registry name
      "head_kwargs": {...},            # e.g. {"num_heads": 8}
      "head_state_dict": {...},        # head weights only — backbone is frozen
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

import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch.multiprocessing as _torch_mp
import torch.nn.functional as F
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.classification import MulticlassAccuracy

# File-system sharing → DataLoader workers don't need /dev/shm
# (containers cap it at 64 MB).
_torch_mp.set_sharing_strategy("file_system")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import load as load_dataset  # noqa: E402
from models import BASES, HEADS, build_base, build_head  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
DATASETS = ("funny_birds", "dsprites", "imagenet_val_hf", "imagenette")

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ── Cache paths ──────────────────────────────────────────────────────────────


def cache_path(base: str, dataset: str, kind: str) -> Path:
    return DATA_DIR / f"{base}_{dataset}_{kind}_feats.pt"


def probe_path(base: str, head: str, dataset: str) -> Path:
    return DATA_DIR / f"{base}_{head}_probe_{dataset}.pt"


def _dataset_kwargs(dataset: str, dsprites_target: str) -> dict:
    if dataset == "dsprites":
        return {"target": dsprites_target}
    if dataset in ("funny_birds", "imagenette"):
        return {"split": "train"}
    return {}


# ── cache command ────────────────────────────────────────────────────────────


@app.command("cache")
def cache_cmd(
    base: str = typer.Argument(
        ..., help=f"Base architecture. One of: {', '.join(BASES)}.",
    ),
    dataset: str = typer.Argument(
        ..., help=f"Dataset name. One of: {', '.join(DATASETS)}.",
    ),
    kind: str = typer.Option(
        "cls", "--kind",
        help="Feature kind to cache: 'cls' (~hundreds of MB; required by "
             "LinearHead) or 'tokens' (full sequence, ~20 GB on funny_birds; "
             "required by AttentiveHead).",
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
    dsprites_target: str = typer.Option(
        "shape", "--dsprites-target",
        help="Latent factor used as the classification target for dsprites.",
    ),
):
    """Extract and cache features from a frozen base.

    Cls features are stored fp32 (small; ~200 MB on funny_birds);
    tokens are stored fp16 (~20 GB on funny_birds, mmap'd at load).
    """
    if base not in BASES:
        raise typer.BadParameter(f"unknown base {base!r}; choose from {sorted(BASES)}")
    if dataset not in DATASETS:
        raise typer.BadParameter(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    if kind not in ("cls", "tokens"):
        raise typer.BadParameter(f"unknown kind {kind!r}; choose 'cls' or 'tokens'")

    out = cache_path(base, dataset, kind)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        d = torch.load(out, map_location="cpu", mmap=True)
        print(f"cache exists at {out}  (feats={tuple(d['feats'].shape)} {d['feats'].dtype})")
        print("→ pass --force to re-extract")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device  : {device}")
    print(f"base    : {base}")
    print(f"dataset : {dataset}")
    print(f"kind    : {kind}")
    print(f"out     : {out}")

    print("\nbuilding base (frozen)")
    base_obj = build_base(base).to(device)
    transform = base_obj.get_transform()
    print(f"  embed_dim={base_obj.embed_dim}")

    print("\nloading dataset")
    ds = load_dataset(
        dataset, transform=transform, **_dataset_kwargs(dataset, dsprites_target),
    )
    print(f"  {dataset}: {len(ds)} images, {ds.num_classes} classes")

    extract = base_obj.extract_cls if kind == "cls" else base_obj.extract_tokens
    out_dtype = torch.float32 if kind == "cls" else torch.float16

    print(f"\nextracting features ({kind}, dtype={out_dtype})")
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True,
    )
    feats, labels = [], []
    seen, t0 = 0, time.time()
    # Cache extraction is one-shot — no autograd graph needed.
    # (Base.extract_* deliberately *don't* set no_grad themselves so the
    # AttnLRP composite can build its backward graph at attribution time.)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = extract(x).to(out_dtype).cpu()
            feats.append(f)
            labels.append(y if torch.is_tensor(y) else torch.as_tensor(y))
            seen += x.shape[0]
            if seen % (50 * batch_size) < batch_size:
                print(f"  extracted {seen}/{len(ds)}  ({time.time() - t0:.1f}s)")

    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    torch.save(
        {"feats": feats, "labels": labels,
         "base": base, "dataset": dataset, "kind": kind,
         "num_classes": int(ds.num_classes),
         "embed_dim": int(base_obj.embed_dim)},
        out,
    )
    print(f"\nsaved {out}  (feats={tuple(feats.shape)} {feats.dtype})")


# ── train command ────────────────────────────────────────────────────────────


class _ProbeLM(L.LightningModule):
    """Lightning wrapper around a head trained on cached features.

    The base is *not* part of this module — features are pre-extracted.
    The saved checkpoint contains only ``self.head``; the backbone is
    re-built from the registry at load time.
    """

    def __init__(
        self, head, num_classes: int, lr: float, weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["head"])
        self.head = head
        kw = dict(num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(**kw)
        self.val_acc = MulticlassAccuracy(**kw)
        self.val_acc5 = MulticlassAccuracy(top_k=5, **kw)

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
        return torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )


@app.command("train")
def train_cmd(
    base: str = typer.Argument(
        ..., help=f"Base architecture. One of: {', '.join(BASES)}.",
    ),
    head: str = typer.Argument(
        ..., help=f"Head architecture. One of: {', '.join(HEADS)}.",
    ),
    dataset: str = typer.Argument(
        ..., help=f"Dataset name. One of: {', '.join(DATASETS)}.",
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
        help="(attentive only) heads in the pooling MultiheadAttention.",
    ),
    matmul_rule: str = typer.Option(
        "alpha_beta", "--matmul-rule",
        help="(attentive only) bilinear-matmul LRP rule baked into the "
             "head's q@kᵀ and weights@v ops. {alpha_beta, matmul_factor_2, "
             "passthrough}. Match the composite the walkthrough uses.",
    ),
    alpha: float = typer.Option(
        0.5, "--alpha",
        help="(attentive only) AlphaBeta-rule α. Default matches the "
             "composite's α=β=0.5 working recipe.",
    ),
    beta: float = typer.Option(
        0.5, "--beta",
        help="(attentive only) AlphaBeta-rule β.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="Output .pt path. Default: data/<base>_<head>_probe_<dataset>.pt",
    ),
    seed: int = typer.Option(0, "--seed"),
):
    """Train a head on cached features.

    Auto-loads the cache that matches the head's ``input_kind`` (``cls``
    for ``linear``, ``tokens`` for ``attentive``).
    """
    if base not in BASES:
        raise typer.BadParameter(f"unknown base {base!r}; choose from {sorted(BASES)}")
    if head not in HEADS:
        raise typer.BadParameter(f"unknown head {head!r}; choose from {sorted(HEADS)}")
    if dataset not in DATASETS:
        raise typer.BadParameter(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")

    L.seed_everything(seed, workers=True)

    head_cls = HEADS[head]
    kind = head_cls.input_kind
    cache = cache_path(base, dataset, kind)
    if not cache.exists():
        raise typer.Exit(
            f"\nMissing feature cache at {cache}. Run:\n\n"
            f"    uv run python experiments/train_probe.py cache "
            f"{base} {dataset} --kind {kind}\n"
        )

    if out is None:
        out = probe_path(base, head, dataset)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"base    : {base}")
    print(f"head    : {head}" + (f" (num_heads={num_heads})" if head == "attentive" else ""))
    print(f"dataset : {dataset}")
    print(f"cache   : {cache}")
    print(f"out     : {out}")

    print(f"\nloading cache (mmap={kind == 'tokens'})")
    d = torch.load(cache, map_location="cpu", mmap=(kind == "tokens"))
    feats, labels = d["feats"], d["labels"]
    embed_dim = int(d["embed_dim"])
    num_classes = int(d["num_classes"])
    print(f"  feats={tuple(feats.shape)} {feats.dtype}, labels={tuple(labels.shape)}")

    # Head construction kwargs — declared per head. These are recorded in
    # the checkpoint and replayed at notebook load time, so the
    # walkthrough's attribution-time backward stays consistent with how
    # the head was trained.
    head_kwargs: dict = {}
    if head == "attentive":
        head_kwargs.update(
            num_heads=num_heads,
            matmul_rule=matmul_rule,
            alpha=alpha,
            beta=beta,
        )
    head_obj = build_head(
        head, embed_dim=embed_dim, num_classes=num_classes, head_kwargs=head_kwargs,
    )

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

    lm = _ProbeLM(head_obj, num_classes=num_classes, lr=lr, weight_decay=weight_decay)

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
        "dataset": dataset,
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


if __name__ == "__main__":
    app()
