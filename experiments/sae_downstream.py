"""Downstream-loss SAE for frozen ViTs (preliminary study).

Idea
----
For a frozen ViT, take block ``i``'s OUTPUT ``M`` (the tensor AFTER the residual
add — the block module's output, copied to both the next block and the skip).
Insert an SAE as a pass-through ``M' = decode(encode(M))``. A *vanilla* SAE is
trained to minimise ``||M − M'||``. HERE INSTEAD we minimise the DOWNSTREAM
reconstruction error: the output of the NEXT consuming stage ``Dnext`` must be
preserved::

    LOSS = || Dnext(M) − Dnext(M') ||²  +  λ · ||encode(M)||₁   (M' = decode(encode(M)))

i.e. ``M`` itself may change (``M' ≠ M``) but it must not change what the
consuming stage ``Dnext`` produces. Model params stay FROZEN; only the SAE trains.
The object of study is the DECODED representation ``M'`` (NOT the codes).

★ ITERATIVE-FROM-OUTPUT TRAINING (critical) ★
---------------------------------------------
The per-block SAEs are NOT independent: inserting an SAE at a block changes the
representation that flows DOWNSTREAM to every later block. So we train them
ITERATIVELY FROM THE OUTPUT toward the input. For blocks ``0..N-1`` we SAE blocks
``0..N-2`` (skip the last). Train in DESCENDING order ``i = N-2, …, 1, 0``.

When training ``SAE_i`` the already-trained downstream SAEs (``i+1 … N-2``) are
INSERTED and FROZEN, so the downstream consuming stage is the *deployed* one::

    Dnext = block_{i+1}  followed by  SAE_{i+1}  (if it exists / has been trained)

This is the representation ACTUALLY propagated in the decomposed model — already
changed by the downstream SAE. ``M`` (block ``i`` output) is itself UN-changed by
the downstream SAEs because they sit *after* block ``i`` (we train descending, so
upstream SAEs ``0..i-1`` do not exist yet either). Hence ``M`` is collected from
the clean model; only ``Dnext`` carries the downstream SAE. Iterating from the
output makes the composite self-consistent: each SAE preserves the next layer's
ALREADY-DECODED representation. (We do NOT train all SAEs in parallel against the
clean model — that would ignore the downstream modification.)

We train one SAE per block EXCEPT the last (the last block's output feeds only
the classifier, no "next block"). For ``vit_small`` (12 blocks) → blocks 0..10.

Reuses :class:`experiments.sae.SparseAutoencoder` (untied L1 SAE, unit-norm
decoder, tied pre-decoder bias) and writes its OWN training loop for the
downstream loss.

Two model cases:
  1. ``vit_small`` + linear probe on funny_birds (50 classes).
  2. ``timm`` ``vit_base_patch16_224`` ImageNet-1k pretrained; eval on an
     imagenet-val subset (HF mirror) — fall back to imagenette if unavailable.

CLI::

    python -m experiments.sae_downstream funny_birds --train
    python -m experiments.sae_downstream imagenet --train --eval --reps
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader, Subset

from experiments.sae import SparseAutoencoder

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "sae_downstream"


# ─────────────────────────────────────────────────────────────────────────────
# Model cases
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelCase:
    key: str                      # "funny_birds" | "imagenet"
    model: nn.Module              # frozen, eval, on device
    blocks: nn.ModuleList         # the transformer blocks (post-residual outputs)
    embed_dim: int
    normalize: Callable           # forward-boundary normalize (un-normalized ds → model space)
    forward_logits: Callable      # x(un-normalized) -> logits
    num_classes: int
    note: str = ""


def load_funny_birds_case(device: str) -> ModelCase:
    from experiments.model_io import load_probe, DATASETS, backbone_transforms
    tag = DATASETS["funny_birds"][2]
    model, ck, _ = load_probe(tag, device)
    _, normalize = backbone_transforms(model.backbone)
    blocks = model.backbone.blocks

    def forward_logits(x):
        return model(normalize(x))

    return ModelCase(
        key="funny_birds", model=model, blocks=blocks,
        embed_dim=model.backbone.embed_dim, normalize=normalize,
        forward_logits=forward_logits, num_classes=ck["num_classes"],
        note="vit_small + linear probe, funny_birds (50 classes)",
    )


def load_imagenet_case(device: str) -> ModelCase:
    import timm
    from timm.data import resolve_data_config
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
    model = model.eval().to(device)
    model.requires_grad_(False)
    cfg = resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, -1, 1, 1).to(device)
    std = torch.tensor(cfg["std"]).view(1, -1, 1, 1).to(device)

    def normalize(x):
        return (x - mean.to(x.dtype)) / std.to(x.dtype)

    def forward_logits(x):
        return model(normalize(x))

    return ModelCase(
        key="imagenet", model=model, blocks=model.blocks,
        embed_dim=model.embed_dim, normalize=normalize,
        forward_logits=forward_logits, num_classes=1000,
        note="timm vit_base_patch16_224 ImageNet-1k pretrained",
    )


def load_case(key: str, device: str) -> ModelCase:
    if key == "funny_birds":
        return load_funny_birds_case(device)
    if key == "imagenet":
        return load_imagenet_case(device)
    raise ValueError(f"unknown case {key!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Datasets (un-normalized [0,1] images; normalize applied at forward boundary)
# ─────────────────────────────────────────────────────────────────────────────

def funny_birds_dataset(model, n_max: int):
    from experiments.model_io import DATASETS, backbone_transforms
    from experiments.datasets import load as load_dataset
    transform, _ = backbone_transforms(model.backbone)
    name, kw, _ = DATASETS["funny_birds"]
    ds = load_dataset(name, root=REPO_ROOT / "data", transform=transform, **kw)
    return ds


def imagenet_dataset(case: ModelCase, n_per_class: int, classes: Optional[List[int]]):
    """Try the HF imagenet-val mirror; fall back to imagenette if download fails."""
    import timm
    from timm.data import resolve_data_config
    from torchvision import transforms as T
    cfg = resolve_data_config({}, model=case.model)
    size = cfg["input_size"][-1]
    # un-normalized transform (resize/centercrop/totensor, mean0/std1)
    tfm = T.Compose([
        T.Resize(int(size / cfg.get("crop_pct", 0.9)), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
    ])
    from experiments.datasets import load as load_dataset
    try:
        ds = load_dataset("imagenet_val_hf", root=REPO_ROOT / "data",
                          n_per_class=n_per_class, classes=classes, transform=tfm)
        return ds, "imagenet_val_hf", False
    except Exception as e:  # noqa: BLE001
        print(f"[imagenet] HF mirror unavailable ({type(e).__name__}: {e}); falling back to imagenette")
        ds = load_dataset("imagenette", root=REPO_ROOT / "data", split="val", transform=tfm)
        return ds, "imagenette", True


# ─────────────────────────────────────────────────────────────────────────────
# Block I/O capture (M = block output; B(M) = next block output)
# ─────────────────────────────────────────────────────────────────────────────

def _as_tensor(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def collect_block_pairs(case: ModelCase, ds, n_images: int, device: str,
                        batch_size: int, blocks_idx: List[int],
                        want_labels: bool = False
                        ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Optional[torch.Tensor]]:
    """One clean forward pass; capture each requested block's OUTPUT.

    Returns ``(M, {}, labels)`` where ``M[i]`` = output of block ``i`` flattened
    to ``(n_tokens, d)`` on CPU float32, for every ``i`` in ``blocks_idx``. The
    downstream target ``B(M)`` is applied LIVE by the trainer (``mc.blocks[i+1]``),
    so it is not stored here. Tensors are pre-allocated and filled in place to
    avoid the ~2× peak of a final ``torch.cat`` (which OOM-killed the full-dataset
    run). Only the requested blocks are stored, bounding RAM to
    ``len(blocks_idx) · n_images · N · d · 4 B``.
    """
    blocks = case.blocks
    n_images = min(n_images, len(ds))
    cap: Dict[int, torch.Tensor] = {}

    def _mk(b):
        def hook(mod, i, o):
            cap[b] = _as_tensor(o).detach()
        return hook
    handles = [blocks[b].register_forward_hook(_mk(b)) for b in blocks_idx]
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)
    labels: List[torch.Tensor] = []
    M: Dict[int, torch.Tensor] = {}
    N = d = None
    seen = 0
    try:
        for batch in loader:
            x, y = batch[0], batch[1]
            case.forward_logits(x.to(device))
            if M is None or not M:
                t0 = cap[blocks_idx[0]]
                N, d = t0.shape[1], t0.shape[2]
                M = {b: torch.empty(n_images, N, d, dtype=torch.float32) for b in blocks_idx}
            take = min(x.shape[0], n_images - seen)
            for b in blocks_idx:
                M[b][seen:seen + take] = cap[b][:take].float().cpu()
            if want_labels:
                labels.append(y[:take].clone())
            seen += take
            if seen >= n_images:
                break
    finally:
        for h in handles:
            h.remove()
    M = {b: M[b][:seen].reshape(-1, d) for b in blocks_idx}   # flatten tokens
    lab = torch.cat(labels)[:seen] if want_labels else None
    return M, {}, lab


# ─────────────────────────────────────────────────────────────────────────────
# Downstream-loss SAE training
# ─────────────────────────────────────────────────────────────────────────────

def _block_apply(block: nn.Module, x: torch.Tensor,
                 next_sae: Optional[nn.Module] = None) -> torch.Tensor:
    """Apply the downstream consuming stage ``Dnext`` to a token tensor ``(B,N,d)``
    and return its output (unwrap tuple). ``Dnext = block`` (frozen) followed by
    ``next_sae`` if given — i.e. the DEPLOYED next-block output when a downstream
    SAE has already been trained and inserted (iterative-from-output scheme). The
    block/SAE params are frozen (no grad), but the graph w.r.t. ``x`` is kept so
    the SAE under training receives downstream gradients."""
    out = _as_tensor(block(x))
    if next_sae is not None:
        out = next_sae(out)
    return out


def train_downstream_sae(
    Mb: torch.Tensor, next_block: nn.Module,
    *, n_tokens_per_img: int, m: int, l1_coeff: float, steps: int, lr: float,
    img_batch: int, device: str, seed: int = 0, resample_every: int = 0,
    mode: str = "downstream", next_sae: Optional[nn.Module] = None,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Train one SAE at a block. ``mode='downstream'`` uses the downstream loss
    ``||Dnext(M) − Dnext(decode(encode(M)))||²``; ``mode='standard'`` uses
    ``||M − M'||²`` (control). Both add ``λ·||encode(M)||₁``.

    ``Dnext = next_block`` followed by ``next_sae`` if given (the already-trained,
    frozen downstream SAE — iterative-from-output scheme). The downstream target
    ``Dnext(M)`` is computed ON-THE-FLY from the clean ``M`` so it reflects the
    deployed (SAE-modified) next-block output, NOT the clean next-block output.

    ``Mb`` is the ``(n_img·N, d)`` token matrix of block ``i`` output (token-major
    within image; ``M`` is clean because the downstream SAEs sit AFTER block ``i``).
    ``Dnext`` runs on full ``(B,N,d)`` sequences, so we sample whole IMAGES
    (``img_batch`` of them) per step and reshape to ``(b,N,d)``.

    Activations are centered + globally RMS-scaled before the SAE (well-
    conditioned); the SAE operates in normalised space. ``Dnext`` is applied in RAW
    space (un-normalise the decoded M' before feeding it). The normalisation is
    folded into the deployed params (:func:`experiments.sae._fold`).
    """
    from experiments.sae import _fold, _resample_dead
    g = torch.Generator().manual_seed(seed)
    N = n_tokens_per_img
    n_tok, d = Mb.shape
    n_img = n_tok // N
    # Mb stays on CPU (it can be the full dataset = many GB); only per-step
    # minibatches are moved to the GPU. Normalisation = per-block center + global
    # RMS scale, computed on CPU then applied on the moved batch.
    Mb = Mb.cpu()
    mean = Mb.mean(0).to(device)
    scale = Mb.std().item() + 1e-8
    img_view_raw = Mb.view(n_img, N, d)           # clean M in raw space (CPU)

    sae = SparseAutoencoder(d, m).to(device)
    sae.b_dec.data.zero_()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    fired = torch.zeros(m, dtype=torch.bool, device=device)

    # Small normalised pool on-GPU for dead-feature resampling (avoids holding
    # the full normalised activation matrix on either device).
    pool_imgs = min(n_img, max(64, 8192 // N + 1))
    pool_idx = torch.randint(0, n_img, (pool_imgs,), generator=g)
    Mn_pool = ((img_view_raw[pool_idx].reshape(-1, d).to(device) - mean) / scale)

    for step in range(steps):
        idx = torch.randint(0, n_img, (img_batch,), generator=g)
        a_raw = img_view_raw[idx].to(device)        # (b,N,d) raw M on GPU
        a_n = (a_raw - mean) / scale                # normalised
        f = sae.encode(a_n.reshape(-1, d))
        mp_n = sae.decode(f).view(img_batch, N, d)  # decoded M' (normalised)
        l1 = f.abs().sum(-1).mean()
        if mode == "downstream":
            mp_raw = mean + scale * mp_n            # back to raw space for Dnext
            with_block = _block_apply(next_block, mp_raw, next_sae)
            with torch.no_grad():                   # Dnext(clean M) = deployed target
                target = _block_apply(next_block, a_raw, next_sae)
            recon_loss = F.mse_loss(with_block, target)
        else:
            recon_loss = F.mse_loss(mp_n, a_n)
        loss = recon_loss + l1_coeff * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        fired |= (f.detach() > 0).any(0)
        if resample_every and (step + 1) % resample_every == 0:
            _resample_dead(sae, Mn_pool, fired, g, device)
            fired = torch.zeros(m, dtype=torch.bool, device=device)

    # Final metrics on the full set (chunked over images; batches moved to GPU).
    with torch.no_grad():
        recon_num = recon_den = l0_sum = n_tok_seen = 0.0
        fired_all = torch.zeros(m, dtype=torch.bool, device=device)
        mn_sq = mn_sum = mn_cnt = 0.0              # for var(normalised M)
        dfvu_num = 0.0
        dfvu_tgt_sum = torch.zeros(d, device=device)
        dfvu_tgt_sq = 0.0
        dfvu_n = 0
        chunk = max(1, min(64, n_img))
        for s in range(0, n_img, chunk):
            e = min(n_img, s + chunk)
            a_raw = img_view_raw[s:e].to(device)
            a_n = ((a_raw - mean) / scale).reshape(-1, d)
            f = sae.encode(a_n)
            mp_n = sae.decode(f)
            recon_num += ((mp_n - a_n) ** 2).sum().item()
            recon_den += (a_n.numel())
            mn_sq += (a_n ** 2).sum().item(); mn_sum += a_n.sum().item(); mn_cnt += a_n.numel()
            l0_sum += (f > 0).float().sum().item()
            n_tok_seen += f.shape[0]
            fired_all |= (f > 0).any(0)
            mp_raw = (mean + scale * mp_n).view(e - s, N, d)
            bm = _block_apply(next_block, mp_raw, next_sae)
            tgt = _block_apply(next_block, a_raw, next_sae)
            dfvu_num += ((bm - tgt) ** 2).sum().item()
            tflat = tgt.reshape(-1, d)
            dfvu_tgt_sum += tflat.sum(0)
            dfvu_tgt_sq += (tflat ** 2).sum().item()
            dfvu_n += tflat.shape[0]
        Mn_var = (mn_sq / mn_cnt) - (mn_sum / mn_cnt) ** 2 + 1e-12
        recon_fvu = (recon_num / recon_den) / Mn_var
        l0 = l0_sum / n_tok_seen
        dead = int((~fired_all).sum().item())
        tgt_mean = dfvu_tgt_sum / dfvu_n
        denom = dfvu_tgt_sq - dfvu_n * (tgt_mean ** 2).sum().item() + 1e-12
        downstream_fvu = dfvu_num / denom

    metrics = dict(mode=mode, recon_fvu=recon_fvu, downstream_fvu=downstream_fvu,
                   l0=l0, dead=dead, d=d, m=m, n_tokens=n_tok, n_img=n_img,
                   scale=scale, l1_coeff=l1_coeff, steps=steps, lr=lr,
                   img_batch=img_batch, has_downstream_sae=next_sae is not None)
    return _fold(sae, mean, scale), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Spliceable deployed SAE (raw-space pass-through; encode/decode on raw acts)
# ─────────────────────────────────────────────────────────────────────────────

class DeployedSAE(nn.Module):
    """Raw-space ``a ↦ decode(relu(encode(a)))``. Built from folded ``raw`` params."""

    def __init__(self, raw: Dict[str, torch.Tensor]):
        super().__init__()
        d, m = raw["W_enc"].shape[1], raw["W_enc"].shape[0]
        self.W_enc = nn.Parameter(raw["W_enc"].clone(), requires_grad=False)
        self.b_enc = nn.Parameter(raw["b_enc"].clone(), requires_grad=False)
        self.W_dec = nn.Parameter(raw["W_dec"].clone(), requires_grad=False)
        self.b_dec = nn.Parameter(raw["b_dec"].clone(), requires_grad=False)

    def encode(self, a):
        return F.relu(F.linear(a, self.W_enc, self.b_enc))

    def decode(self, f):
        return F.linear(f, self.W_dec, self.b_dec)

    def forward(self, a):
        return self.decode(self.encode(a))


class BlockSAEWrapper(nn.Module):
    """Wrap a block so its OUTPUT is replaced by ``decode(encode(output))``."""

    def __init__(self, block: nn.Module, sae: DeployedSAE):
        super().__init__()
        self.block = block
        self.sae = sae

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        t = _as_tensor(out)
        rec = self.sae(t)
        if isinstance(out, tuple):
            return (rec, *out[1:])
        return rec


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def sae_ckpt(case_key: str, block: int, mode: str) -> Path:
    return OUT_DIR / case_key / f"sae_{mode}_block{block}.pt"


def reps_path(case_key: str, block: int) -> Path:
    return OUT_DIR / case_key / f"reps_block{block}.npz"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_accuracy(case: ModelCase, ds, n_images: int, device: str,
                      batch_size: int, splice: Optional[List[Tuple[int, DeployedSAE]]] = None,
                      ) -> Dict[str, float]:
    """Top-1 (and top-5) on up to ``n_images``. If ``splice`` given, wrap those
    blocks with their SAE for the whole forward (all spliced simultaneously)."""
    blocks = case.blocks
    wrapped: List[Tuple[int, nn.Module]] = []
    if splice:
        for b, sae in splice:
            orig = blocks[b]
            blocks[b] = BlockSAEWrapper(orig, sae.to(device))
            wrapped.append((b, orig))
    try:
        loader = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)
        top1 = top5 = total = 0
        seen = 0
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = case.forward_logits(x)
            pred = logits.topk(5, dim=-1).indices
            top1 += (pred[:, 0] == y).sum().item()
            top5 += (pred == y.unsqueeze(1)).any(-1).sum().item()
            total += y.numel()
            seen += x.shape[0]
            if seen >= n_images:
                break
    finally:
        for b, orig in wrapped:
            blocks[b] = orig
    return {"top1": top1 / total, "top5": top5 / total, "n": total}


@torch.no_grad()
def _capture_cls(case: ModelCase, ds, n_images: int, device: str, batch_size: int,
                 blocks_idx: List[int], N: int,
                 splice: Optional[List[Tuple[int, "DeployedSAE"]]] = None,
                 ) -> Tuple[Dict[int, np.ndarray], torch.Tensor]:
    """Forward up to ``n_images``; return ``{b: cls_rep (n_img, d)}`` = block ``b``
    OUTPUT's CLS token (token 0), plus per-image labels. If ``splice`` is given the
    listed blocks are wrapped with their SAE for the WHOLE forward, so the captured
    block-``b`` output is the DEPLOYED (decomposed-model) representation at that
    site — i.e. ``decode(encode(M_b^deployed))`` with all SAEs active."""
    blocks = case.blocks
    wrapped: List[Tuple[int, nn.Module]] = []
    if splice:
        for b, sae in splice:
            orig = blocks[b]
            blocks[b] = BlockSAEWrapper(orig, sae.to(device))
            wrapped.append((b, orig))
    caps: Dict[int, List[torch.Tensor]] = {b: [] for b in blocks_idx}

    def _mk(b):
        def hook(mod, i, o):
            t = _as_tensor(o)
            caps[b].append(t[:, 0, :].detach().float().cpu())  # CLS token
        return hook
    handles = [blocks[b].register_forward_hook(_mk(b)) for b in blocks_idx]
    labels: List[torch.Tensor] = []
    seen = 0
    try:
        loader = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)
        for batch in loader:
            x, y = batch[0], batch[1]
            case.forward_logits(x.to(device))
            labels.append(y.clone())
            seen += x.shape[0]
            if seen >= n_images:
                break
    finally:
        for h in handles:
            h.remove()
        for b, orig in wrapped:
            blocks[b] = orig
    out = {b: torch.cat(caps[b], 0)[:n_images].numpy() for b in blocks_idx}
    lab = torch.cat(labels)[:n_images]
    return out, lab


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition-quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def participation_ratio(X: np.ndarray) -> float:
    """Effective dimensionality: (Σλ)² / Σλ² of the covariance eigenvalues."""
    Xc = X - X.mean(0, keepdims=True)
    # eigenvalues of covariance via SVD of centered data
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s ** 2)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))


def knn_purity(X: np.ndarray, y: np.ndarray, k: int = 10, n_max: int = 4000,
               seed: int = 0) -> float:
    """Mean fraction of a point's k nearest neighbours sharing its label."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        idx = rng.choice(len(X), n_max, replace=False)
        X, y = X[idx], y[idx]
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, nbr = nn_.kneighbors(X)
    nbr = nbr[:, 1:]  # drop self
    same = (y[nbr] == y[:, None]).mean()
    return float(same)


def silhouette(X: np.ndarray, y: np.ndarray, n_max: int = 4000, seed: int = 0) -> float:
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        idx = rng.choice(len(X), n_max, replace=False)
        X, y = X[idx], y[idx]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(silhouette_score(X, y))


def linear_probe_acc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Train/test-split logistic-regression accuracy on the rep. Classes with
    <2 samples are dropped (can't be split / stratified — happens for ImageNet
    where reps may have ~1 sample per 1000 classes); falls back to an
    unstratified split if stratification is still infeasible."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    classes, counts = np.unique(y, return_counts=True)
    keep = np.isin(y, classes[counts >= 2])
    X, y = X[keep], y[keep]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte))


def rep_metrics(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {
        "knn_purity": knn_purity(X, y),
        "silhouette": silhouette(X, y),
        "participation_ratio": participation_ratio(X),
        "linear_probe_acc": linear_probe_acc(X, y),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def run(
    case: str = typer.Argument(..., help="funny_birds | imagenet"),
    train: bool = typer.Option(False, "--train"),
    train_standard: bool = typer.Option(False, "--train-standard", help="also train vanilla-recon control SAEs"),
    eval_: bool = typer.Option(False, "--eval"),
    reps: bool = typer.Option(False, "--reps", help="store ORIG/DECODED reps at rep_blocks"),
    expansion: float = typer.Option(4.0, "--expansion", help="dict size m = expansion × embed_dim (may be <1 for an undercomplete bottleneck)"),
    l1_coeff: float = typer.Option(1e-3, "--l1-coeff"),
    steps: int = typer.Option(1500, "--steps"),
    lr: float = typer.Option(1e-3, "--lr"),
    img_batch: int = typer.Option(0, "--img-batch", help="0 = auto-probe"),
    n_train_images: int = typer.Option(800, "--n-train-images"),
    n_eval_images: int = typer.Option(2000, "--n-eval-images"),
    n_rep_images: int = typer.Option(1500, "--n-rep-images"),
    rep_blocks: List[int] = typer.Option([], "--rep-block", help="blocks to store reps for"),
    device: Optional[str] = typer.Option(None, "--device"),
):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    (OUT_DIR / case).mkdir(parents=True, exist_ok=True)
    mc = load_case(case, dev)
    n_blocks = len(mc.blocks)
    train_blocks = list(range(n_blocks - 1))  # all but last
    print(f"[{case}] {mc.note} | blocks={n_blocks} train_blocks={train_blocks} dev={dev}")

    meta_path = OUT_DIR / case / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    # ── datasets ──
    if case == "funny_birds":
        train_ds = funny_birds_dataset(mc.model, n_train_images)
        eval_ds = train_ds  # funny_birds eval uses the same (train, clean) split
        ds_note = "funny_birds train/clean"
    else:
        eval_ds, ds_name, fellback = imagenet_dataset(mc, n_per_class=5, classes=None)
        train_ds = eval_ds
        ds_note = ds_name + (" (FALLBACK from imagenet_val_hf)" if fellback else "")
    meta["dataset"] = ds_note
    meta["note"] = mc.note

    # token count per image (probe one forward)
    with torch.no_grad():
        xb = next(iter(DataLoader(train_ds, batch_size=2)))[0].to(dev)
        cap = {}
        h = mc.blocks[0].register_forward_hook(lambda m, i, o: cap.__setitem__("t", _as_tensor(o)))
        mc.forward_logits(xb)
        h.remove()
        N = cap["t"].shape[1]
    print(f"[{case}] tokens/image N={N} embed_dim={mc.embed_dim}")

    # auto-probe img_batch if not given (downstream loss runs B over (b,N,d))
    if img_batch == 0 and dev == "cuda":
        img_batch = _probe_img_batch(mc, train_blocks[len(train_blocks)//2], N, dev)
        print(f"[{case}] auto img_batch={img_batch}")
    elif img_batch == 0:
        img_batch = 16

    if train:
        m = int(round(expansion * mc.embed_dim))
        # ── DOWNSTREAM mode: ITERATIVE-FROM-OUTPUT (descending block order) ──
        # SAE_i conditioned on the already-trained, frozen downstream SAE_{i+1}.
        # Collect each block's clean output M[b] JUST-IN-TIME over the dataset
        # (one block at a time ≈ n_img·N·d floats), train, then free — bounds RAM
        # to a single block and avoids one ~100 GB allocation (segfaults here).
        # Resumable: a block whose checkpoint already exists is reused (survives
        # pod bounces mid-run).
        trained_ds_saes: Dict[int, DeployedSAE] = {}
        for b in sorted(train_blocks, reverse=True):
            ck_p = sae_ckpt(case, b, "downstream")
            if ck_p.is_file():
                ck_d = torch.load(ck_p, map_location=dev)
                trained_ds_saes[b] = DeployedSAE(ck_d["raw"]).to(dev).eval()
                meta.setdefault("sae", {}).setdefault("downstream", {})[str(b)] = {
                    k: v for k, v in ck_d.items() if k not in ("raw", "case", "block")}
                print(f"[{case}/downstream/b{b}] checkpoint exists → reuse "
                      f"(dFVU={ck_d.get('downstream_fvu', float('nan')):.4f})")
                continue
            print(f"[{case}] collecting block {b} output ({n_train_images} imgs)…")
            Mb = collect_block_pairs(mc, train_ds, n_train_images, dev,
                                     batch_size=max(8, img_batch), blocks_idx=[b])[0][b]
            next_sae = trained_ds_saes.get(b + 1)   # SAE_{i+1} if already trained
            raw, metrics = train_downstream_sae(
                Mb, mc.blocks[b + 1], n_tokens_per_img=N, m=m,
                l1_coeff=l1_coeff, steps=steps, lr=lr, img_batch=img_batch,
                device=dev, mode="downstream", next_sae=next_sae,
                resample_every=max(0, steps // 2))
            del Mb
            ck_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"raw": raw, **metrics, "case": case, "block": b}, ck_p)
            meta.setdefault("sae", {}).setdefault("downstream", {})[str(b)] = metrics
            meta_path.write_text(json.dumps(meta, indent=2, default=float))
            # deploy this SAE (frozen) so the NEXT (more upstream) SAE conditions on it
            trained_ds_saes[b] = DeployedSAE(raw).to(dev).eval()
            print(f"[{case}/downstream/b{b}] dFVU={metrics['downstream_fvu']:.4f} "
                  f"reconFVU={metrics['recon_fvu']:.3f} L0={metrics['l0']:.1f} "
                  f"dead={metrics['dead']}/{m} (cond on SAE_{b+1}={'yes' if next_sae else 'no'})")
        torch.cuda.empty_cache()

    if eval_:
        orig = evaluate_accuracy(mc, eval_ds, n_eval_images, dev, batch_size=max(8, img_batch))
        splice = [(b, DeployedSAE(torch.load(sae_ckpt(case, b, "downstream"),
                                             map_location=dev)["raw"]))
                  for b in train_blocks if sae_ckpt(case, b, "downstream").is_file()]
        dec = evaluate_accuracy(mc, eval_ds, n_eval_images, dev,
                                batch_size=max(8, img_batch), splice=splice)
        meta["eval"] = {"orig": orig, "decomposed_all_saes": dec,
                        "delta_top1": dec["top1"] - orig["top1"]}
        # standard-control decomposed eval if available
        if all(sae_ckpt(case, b, "standard").is_file() for b in train_blocks):
            splice_s = [(b, DeployedSAE(torch.load(sae_ckpt(case, b, "standard"),
                                                   map_location=dev)["raw"]))
                        for b in train_blocks]
            dec_s = evaluate_accuracy(mc, eval_ds, n_eval_images, dev,
                                      batch_size=max(8, img_batch), splice=splice_s)
            meta["eval"]["decomposed_standard"] = dec_s
        meta_path.write_text(json.dumps(meta, indent=2, default=float))
        print(f"[{case}] orig top1={orig['top1']:.4f} top5={orig['top5']:.4f} | "
              f"decomposed top1={dec['top1']:.4f} top5={dec['top5']:.4f} "
              f"Δtop1={dec['top1']-orig['top1']:+.4f}")

    if reps:
        rb = rep_blocks or _default_rep_blocks(n_blocks)
        print(f"[{case}] storing reps for blocks {rb} ({n_rep_images} imgs)…")
        # Fixed SHUFFLED subset so the rep set spans all classes (datasets like
        # FunnyBirds are ordered by class → sequential first-N would capture only
        # the first few classes → degenerate manifolds). Same indices for ORIG and
        # all spliced captures so the points stay paired.
        rep_perm = torch.randperm(len(eval_ds),
                                  generator=torch.Generator().manual_seed(0))[:n_rep_images].tolist()
        rep_ds = Subset(eval_ds, rep_perm)
        # ORIG = clean block-b output. DECODED = the DEPLOYED decomposed model's
        # rep at site b (= post-SAE block-b output WITH all downstream/upstream SAEs
        # active), captured by splicing the WHOLE model and reading block-b output.
        cls_orig, labels = _capture_cls(mc, rep_ds, n_rep_images, dev,
                                        max(8, img_batch), rb, N, splice=None)
        labels_np = labels.numpy()
        # downstream-SAE decomposed model (all blocks 0..N-2 spliced)
        ds_splice = [(bb, DeployedSAE(torch.load(sae_ckpt(case, bb, "downstream"),
                                                 map_location=dev)["raw"]))
                     for bb in train_blocks if sae_ckpt(case, bb, "downstream").is_file()]
        cls_dec, _ = _capture_cls(mc, rep_ds, n_rep_images, dev,
                                  max(8, img_batch), rb, N, splice=ds_splice)
        cls_std = None
        if all(sae_ckpt(case, bb, "standard").is_file() for bb in train_blocks):
            st_splice = [(bb, DeployedSAE(torch.load(sae_ckpt(case, bb, "standard"),
                                                     map_location=dev)["raw"]))
                         for bb in train_blocks]
            cls_std, _ = _capture_cls(mc, rep_ds, n_rep_images, dev,
                                      max(8, img_batch), rb, N, splice=st_splice)
        for b in rb:
            store = {"orig": cls_orig[b], "labels": labels_np,
                     "decoded_downstream": cls_dec[b]}
            if cls_std is not None:
                store["decoded_standard"] = cls_std[b]
            np.savez_compressed(reps_path(case, b), **store)
            print(f"[{case}/b{b}] reps stored: n_img={cls_orig[b].shape[0]} keys={list(store)}")

    # decomposition-quality metrics over stored reps
    quality = {}
    rb = rep_blocks or _default_rep_blocks(n_blocks)
    for b in rb:
        p = reps_path(case, b)
        if not p.is_file():
            continue
        z = np.load(p)
        y = z["labels"]
        qb = {"orig": rep_metrics(z["orig"], y)}
        if "decoded_downstream" in z:
            qb["decoded_downstream"] = rep_metrics(z["decoded_downstream"], y)
        if "decoded_standard" in z:
            qb["decoded_standard"] = rep_metrics(z["decoded_standard"], y)
        quality[str(b)] = qb
        print(f"[{case}/b{b}] quality: " + " | ".join(
            f"{k}:knn={v['knn_purity']:.3f},sil={v['silhouette']:.3f},"
            f"PR={v['participation_ratio']:.1f},lin={v['linear_probe_acc']:.3f}"
            for k, v in qb.items()))
    if quality:
        meta["quality"] = quality
        meta_path.write_text(json.dumps(meta, indent=2, default=float))
    print(f"[{case}] meta → {meta_path}")


def _default_rep_blocks(n_blocks: int) -> List[int]:
    """Early / mid / late representative blocks (all valid SAE'd sites)."""
    last = n_blocks - 2  # last block with an SAE is n_blocks-2 (target is n_blocks-1)
    mid = (n_blocks - 2) // 2
    return sorted({1, mid, last})


def _probe_img_batch(mc: ModelCase, b: int, N: int, dev: str, cap: int = 256) -> int:
    """Double the image-batch until OOM on the downstream fwd+bwd, then back off."""
    d = mc.embed_dim
    block = mc.blocks[b + 1]
    sae = SparseAutoencoder(d, 4 * d).to(dev)
    bs = 4
    best = bs
    while bs <= cap:
        try:
            torch.cuda.empty_cache()
            x = torch.randn(bs, N, d, device=dev, requires_grad=False)
            f = sae.encode(x.reshape(-1, d))
            mp = sae.decode(f).view(bs, N, d)
            out = _block_apply(block, mp)
            loss = out.float().pow(2).mean()
            loss.backward()
            sae.zero_grad(set_to_none=True)
            best = bs
            bs *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    del sae
    torch.cuda.empty_cache()
    return max(4, best // 2 if best > 4 else best)


if __name__ == "__main__":
    app()
