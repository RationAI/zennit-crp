"""Composable ViT base + head modules for explainability experiments.

Adding a new architecture or head:

1. New base — drop a file in ``models/bases/<name>.py`` that subclasses
   :class:`~models.bases.base.Base` and sets ``timm_name``. Register it
   in :data:`BASES` below.
2. New head — drop a file in ``models/heads/<name>.py`` that subclasses
   :class:`~models.heads.base.Head` and sets ``input_kind`` plus any
   constructor parameters. Register it in :data:`HEADS` below.

Then both the training CLI (``experiments/train_probe.py``) and the
walkthrough notebook pick up the new entry automatically — they iterate
over the registry to populate ``--help`` and to dispatch construction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .bases import Base, ViTBase, ViTSmall, DinoV3
from .heads import Head, LinearHead, AttentiveHead, BlockHead
from .probe import Probe


# ── Registries ────────────────────────────────────────────────────────────────

BASES: Dict[str, Type[Base]] = {
    "vit_base":   ViTBase,
    "vit_small":  ViTSmall,
    "vit_dinov3": DinoV3,
}

HEADS: Dict[str, Type[Head]] = {
    "linear":    LinearHead,
    "attentive": AttentiveHead,
    "block":     BlockHead,
}


# ── Builders ──────────────────────────────────────────────────────────────────

def build_base(name: str) -> Base:
    """Build a frozen base by registry name."""
    if name not in BASES:
        raise ValueError(
            f"unknown base {name!r}; choose from {sorted(BASES)}"
        )
    return BASES[name]()


def build_head(
    name: str, *, embed_dim: int, num_classes: int,
    head_kwargs: Optional[Dict[str, Any]] = None,
) -> Head:
    """Build a head by registry name. Extra constructor kwargs go in
    ``head_kwargs`` (e.g. ``{"num_heads": 8}`` for attentive)."""
    if name not in HEADS:
        raise ValueError(
            f"unknown head {name!r}; choose from {sorted(HEADS)}"
        )
    return HEADS[name](
        embed_dim=embed_dim, num_classes=num_classes,
        **(head_kwargs or {}),
    )


def build_probe(
    base: str, head: str, *, num_classes: int,
    head_kwargs: Optional[Dict[str, Any]] = None,
) -> Probe:
    """Convenience: build a base and head and compose them."""
    base_obj = build_base(base)
    head_obj = build_head(
        head, embed_dim=base_obj.embed_dim, num_classes=num_classes,
        head_kwargs=head_kwargs,
    )
    return Probe(base_obj, head_obj)


__all__ = [
    "BASES", "HEADS",
    "Base", "ViTBase", "ViTSmall", "DinoV3",
    "Head", "LinearHead", "AttentiveHead", "BlockHead",
    "Probe",
    "build_base", "build_head", "build_probe",
]
