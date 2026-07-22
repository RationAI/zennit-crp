"""Sparse autoencoders (SAEs) for SAE-basis CRP on ViT activations.

Trains one vanilla L1 SAE per ``(probe site, dataset, block)`` on activations
collected from the fine-tuned ``vit_small`` probe, so the concept-flipping
experiment can swap CRP's *axis-aligned* concept basis (``EmbeddingDimConcept``,
one concept per embedding dim) for a *learned, over-complete, ~monosemantic*
dictionary — see ``research/sae_crp_plan.md`` (§3 defines the SAE we implement).

The SAE (Bricken et al. 2023 form, untied weights, tied pre-decoder bias,
unit-norm decoder columns):

    f = ReLU(W_enc (a − b_dec) + b_enc)        # sparse codes, f ∈ ℝ^m, m = α·d
    â = W_dec f + b_dec                          # reconstruction, â ≈ a
    L = ‖a − â‖₂²  +  λ · Σ_i f_i                # decoder columns unit-norm ⇒
                                                 #   weighted L1 = plain L1

``concept_flipping.py --concept sae`` reconstructs the trained SAE from the saved
state-dict and splices it in at the probe site as a reconstruction pass-through
with the feature activations ``f`` exposed as a recordable ``features`` sublayer
(the *CRP move*: logit relevance is decomposed onto the SAE latents).

CLI::

    uv run python -m experiments.sae --datasets dsprites --site proj_drop
    uv run python -m experiments.sae --site residual --expansion 8 --l1-coeff 1e-3
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from timm.data import resolve_data_config, create_transform
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "results" / "sae"

# probe-site key → how to find the per-block module whose OUTPUT is the site
# tensor (B, N, embed_dim). ``proj_drop`` = attention output-projection dropout
# (same site the axis-aligned concept-flipping probes); ``residual`` = the block
# output (full residual stream), the site the SAE-on-ViT literature hooks.
SITES = ("proj_drop", "residual")


def site_modules(model: nn.Module, site: str) -> List[nn.Module]:
    blocks = model.backbone.blocks
    if site == "proj_drop":
        return [blocks[b].attn.proj_drop for b in range(len(blocks))]
    if site == "residual":
        return [blocks[b] for b in range(len(blocks))]
    raise ValueError(f"unknown site {site!r}; pick from {SITES}")


# ─────────────────────────────────────────────────────────────────────────────
# The SAE
# ─────────────────────────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """Untied L1 SAE with tied pre-decoder bias and unit-norm decoder columns.

    ``W_enc``: ``(m, d)``  ·  ``b_enc``: ``(m,)``  ·  ``W_dec``: ``(d, m)``  ·
    ``b_dec``: ``(d,)``. The decoder columns ``W_dec[:, i]`` are the dictionary
    atoms ``d_i`` and are kept unit-norm by :meth:`normalize_decoder`.
    """

    def __init__(self, d: int, m: int):
        super().__init__()
        self.d, self.m = int(d), int(m)
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.b_enc = nn.Parameter(torch.zeros(m))
        # Kaiming-ish init; decoder = encoderᵀ then re-normalised (Bricken init).
        w = torch.randn(m, d) / (d ** 0.5)
        self.W_enc = nn.Parameter(w.clone())
        self.W_dec = nn.Parameter(w.t().clone())
        self.normalize_decoder()

    def encode(self, a: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(a - self.b_dec, self.W_enc, self.b_enc))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return F.linear(f, self.W_dec, self.b_dec)

    def forward(self, a: torch.Tensor):
        f = self.encode(a)
        return self.decode(f), f

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True) + 1e-8)


class FeatureTap(nn.Identity):
    """Named identity whose OUTPUT is the SAE feature tensor ``f`` (B, N, m).
    Record relevance / attach the perturbation hook here. Subclasses
    ``nn.Identity`` so the LRP composite's ``(nn.Identity, Pass)`` entry gives it
    a gradient-identity (conservation-preserving) rule."""


class SAESplice(nn.Module):
    """Reconstruction pass-through that replaces a probe-site module:
    ``a ↦ â = decode(ReLU(encode(a − b_dec)))`` with ``â ≈ a``, exposing ``f`` at
    ``.features``. Encoder/decoder are plain ``nn.Linear`` so the composite's
    ``(nn.Linear, γ)`` rule decomposes the logit's relevance onto ``f``
    conservatively (the CRP move). ``inner`` (if given) runs first — used for the
    *residual* site, where the wrapped block produces ``a`` before reconstruction.
    """

    def __init__(self, raw: Dict[str, torch.Tensor], inner: Optional[nn.Module] = None):
        """``raw`` holds the *deployed* (folded) params — ``W_enc,b_enc`` map raw
        activations straight to the feature codes ``f`` (input centering + scaling
        baked in), ``W_dec,b_dec`` map ``f`` back to raw-scale reconstructions, so
        ``f`` here is bit-identical to the normalised-space ``f`` the SAE trained
        on (sparsity / relevance preserved) while ``â`` lands in raw space."""
        super().__init__()
        self.inner = inner
        d, m = raw["W_enc"].shape[1], raw["W_enc"].shape[0]
        self.encode = nn.Linear(d, m)
        self.encode.weight.data = raw["W_enc"].clone()
        self.encode.bias.data = raw["b_enc"].clone()
        self.act = nn.ReLU()
        self.features = FeatureTap()
        self.decode = nn.Linear(m, d)
        self.decode.weight.data = raw["W_dec"].clone()
        self.decode.bias.data = raw["b_dec"].clone()

    def forward(self, x):
        a = self.inner(x) if self.inner is not None else x
        a = a[0] if isinstance(a, tuple) else a
        f = self.features(self.act(self.encode(a)))
        return self.decode(f)

    @property
    def m(self) -> int:
        return self.decode.weight.shape[1]


def load_sae(site: str, dataset: str, block: int, device: str,
             m: Optional[int] = None) -> "SAESplice":
    """Reconstruct the spliceable SAE (deployed Linear form) from disk. Returns a
    detached :class:`SAESplice` with ``inner=None``; the caller rewires ``inner``
    for the residual site. ``.encode/.decode/.features`` are the real submodules.
    ``m`` selects the dictionary size (None ⇒ legacy m=3072 run)."""
    ck = torch.load(sae_path(site, dataset, block, m=m), map_location=device, weights_only=False)
    raw = {k: ck["raw"][k].to(device) for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
    return SAESplice(raw).to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Activation collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_activations(model, ds, mods: Sequence[nn.Module], n_images: int,
                        device: str, batch_size: int = 64) -> List[torch.Tensor]:
    """Forward up to ``n_images`` dataset images, capture every block's site
    output, flatten tokens → one ``(num_images·N, d)`` matrix per block (CPU).
    A single forward pass feeds all blocks (one hook each)."""
    caps: Dict[int, List[torch.Tensor]] = {b: [] for b in range(len(mods))}

    def _mk(b):
        def hook(mod, i, o):
            t = o[0] if isinstance(o, tuple) else o
            caps[b].append(t.detach().reshape(-1, t.shape[-1]).float().cpu())
        return hook
    handles = [m.register_forward_hook(_mk(b)) for b, m in enumerate(mods)]
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)
    seen = 0
    try:
        for x, _ in loader:
            model(x.to(device))
            seen += x.shape[0]
            if seen >= n_images:
                break
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(caps[b], 0) for b in range(len(mods))]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _fold(sae: "SparseAutoencoder", mean: torch.Tensor, scale: float) -> Dict[str, torch.Tensor]:
    """Bake input normalisation ``a ↦ (a−mean)/scale`` into the trained (normalised-
    space) SAE so the deployed encode/decode operate on RAW activations and
    reconstruct raw-scale ``â``, while the feature codes ``f`` stay identical:

        f = ReLU(W_enc((a−mean)/scale − b_dec) + b_enc)
          = ReLU((W_enc/scale)·a + (b_enc − W_enc·(mean/scale + b_dec)))
        â = mean + scale·(W_dec f + b_dec)
          = (scale·W_dec)·f + (mean + scale·b_dec)
    """
    We, be, Wd, bd = sae.W_enc.detach(), sae.b_enc.detach(), sae.W_dec.detach(), sae.b_dec.detach()
    return dict(
        W_enc=(We / scale).cpu(),
        b_enc=(be - We @ (mean / scale + bd)).cpu(),
        W_dec=(scale * Wd).cpu(),
        b_dec=(mean + scale * bd).cpu(),
    )


def train_sae(acts: torch.Tensor, m: int, *, l1_coeff: float, steps: int,
              lr: float, batch: int, device: str,
              resample_every: int = 0, seed: int = 0) -> tuple:
    """Train one SAE on a ``(n_tokens, d)`` activation matrix. Activations are
    centered + globally scaled to unit RMS before training (so the loss is
    well-conditioned and reconstruction is faithful); the normalisation is then
    folded into the returned deployable ``raw`` params (:func:`_fold`). Returns
    ``(raw, metrics)``. Decoder columns are re-normalised each step; dead latents
    (never active over a window) are re-initialised toward high-error inputs."""
    g = torch.Generator().manual_seed(seed)
    n, d = acts.shape
    acts = acts.to(device)
    mean = acts.mean(0)
    scale = acts.std().item() + 1e-8           # single global RMS scalar
    An = (acts - mean) / scale                  # normalised activations

    sae = SparseAutoencoder(d, m).to(device)
    sae.b_dec.data.zero_()                       # An already centered
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    fired = torch.zeros(m, dtype=torch.bool, device=device)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g).to(device)
        a = An[idx]
        recon, f = sae(a)
        mse = F.mse_loss(recon, a)
        l1 = f.abs().sum(-1).mean()
        loss = mse + l1_coeff * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        fired |= (f.detach() > 0).any(0)
        if resample_every and (step + 1) % resample_every == 0:
            _resample_dead(sae, An, fired, g, device)
            fired = torch.zeros(m, dtype=torch.bool, device=device)

    with torch.no_grad():
        recon, f = sae(An)
        fvu = (F.mse_loss(recon, An) / (An.var() + 1e-12)).item()   # unit-var space
        l0 = (f > 0).float().sum(-1).mean().item()
        dead = int((~(f > 0).any(0)).sum().item())
    metrics = dict(fvu=fvu, l0=l0, dead=dead, d=d, m=m, n_tokens=n,
                   scale=scale, l1_coeff=l1_coeff, steps=steps, lr=lr, batch=batch)
    return _fold(sae, mean, scale), metrics


@torch.no_grad()
def _resample_dead(sae, acts, fired, g, device, max_n: int = 4096):
    dead = (~fired).nonzero(as_tuple=True)[0]
    if dead.numel() == 0:
        return
    sub = acts[torch.randint(0, acts.shape[0], (min(max_n, acts.shape[0]),), generator=g).to(device)]
    recon, _ = sae(sub)
    err = (sub - recon).norm(dim=-1)
    pick = torch.multinomial(err / (err.sum() + 1e-12), dead.numel(), replacement=True)
    v = sub[pick]
    vn = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    sae.W_dec.data[:, dead] = vn.t()
    sae.W_enc.data[dead] = vn * 0.2
    sae.b_enc.data[dead] = 0.0


def sae_path(site: str, dataset: str, block: int, m: Optional[int] = None) -> Path:
    """Checkpoint path for one SAE. ``m`` is encoded in the filename so several
    dictionary sizes coexist for the same (site, dataset, block). ``m=None``
    yields the legacy (size-less) name = the original ``m=3072`` run."""
    suff = "" if m is None else f"_m{m}"
    return OUT_DIR / f"sae_{site}_{dataset}{suff}_block{block}.pt"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    datasets: List[str] = typer.Option([], "--datasets", help="dataset keys (default: all)"),
    site: List[str] = typer.Option(["proj_drop"], "--site", help=f"{SITES}"),
    blocks: List[int] = typer.Option([], "--blocks", help="block subset (default all)"),
    expansion: List[float] = typer.Option([8.0], "--expansion", help="dictionary size(s) "
                                          "m = round(α·d); pass several to sweep (α<1 ⇒ undercomplete)"),
    l1_coeff: float = typer.Option(1e-3, "--l1-coeff", help="L1 sparsity λ"),
    steps: int = typer.Option(4000, "--steps"),
    lr: float = typer.Option(1e-3, "--lr"),
    batch: int = typer.Option(4096, "--batch"),
    n_images: int = typer.Option(1500, "--n-images", help="images for activation collection"),
    resample_every: int = typer.Option(2000, "--resample-every", help="0 disables"),
    device: Optional[str] = typer.Option(None, "--device"),
):
    from experiments.concept_flipping import DATASETS, load_probe
    datasets = datasets or list(DATASETS)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    print(f"device={dev} datasets={datasets} sites={site} expansion={expansion} "
          f"l1={l1_coeff} steps={steps} n_images={n_images}")
    for s in site:
        for key in datasets:
            tag = DATASETS[key][2]
            model, ck, _ = load_probe(tag, dev)
            mods = site_modules(model, s)
            d = model.backbone.embed_dim
            n_blocks = len(mods)
            blk = blocks or list(range(n_blocks))
            # Resume across pod bounces: skip (dim, block) pairs already on disk,
            # and skip the (expensive) activation collection entirely if nothing
            # is left to train for this site.
            ms = [int(round(exp * d)) for exp in expansion]
            todo = [(m, b) for m in ms for b in blk if not sae_path(s, key, b, m=m).is_file()]
            if not todo:
                print(f"[{s}/{key}] all {len(ms)*len(blk)} SAEs present — skip")
                del model
                torch.cuda.empty_cache()
                continue
            transform = create_transform(**resolve_data_config({}, model=model.backbone), is_training=False)
            from experiments.datasets import load as load_dataset
            ds = load_dataset(DATASETS[key][0], root=REPO_ROOT / "data", transform=transform, **DATASETS[key][1])
            print(f"[{s}/{key}] collecting acts ({n_images} imgs, {n_blocks} blocks) for {len(todo)} SAEs…")
            acts = collect_activations(model, ds, mods, n_images, dev)
            for exp in expansion:                      # reuse acts across all dict sizes
                m = int(round(exp * d))
                for b in blk:
                    if sae_path(s, key, b, m=m).is_file():
                        continue
                    raw, metrics = train_sae(acts[b], m, l1_coeff=l1_coeff, steps=steps,
                                             lr=lr, batch=batch, device=dev,
                                             resample_every=resample_every)
                    p = sae_path(s, key, b, m=m)
                    torch.save({"raw": raw, "d": d, "m": m, "expansion": exp,
                                "site": s, "dataset": key, "block": b, **metrics}, p)
                    meta.setdefault(s, {}).setdefault(key, {}).setdefault(str(m), {})[str(b)] = metrics
                    meta_path.write_text(json.dumps(meta, indent=2))
                    print(f"[{s}/{key}/m{m}/b{b}] fvu={metrics['fvu']:.3f} L0={metrics['l0']:.1f} "
                          f"dead={metrics['dead']}/{m} → {p.name}")
            del acts, model
            torch.cuda.empty_cache()
    print(f"meta → {meta_path}")


if __name__ == "__main__":
    app()
