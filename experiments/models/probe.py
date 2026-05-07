"""Compose a frozen :class:`~models.bases.base.Base` with a trainable
:class:`~models.heads.base.Head` into a single ``nn.Module``.

The backbone is registered at ``self.backbone`` and the head at
``self.head``; ``__getattr__`` falls through to the backbone, so
existing code that does ``model.blocks``, ``model.cls_token``, etc.
keeps working unchanged. That matters for the AttnLRP composite +
concept classes, which look up layers by paths like
``blocks.{i}.attn.head_proj``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .bases.base import Base
from .heads.base import Head


class Probe(nn.Module):
    """Frozen backbone + trainable head.

    Forward path:

    * ``head.input_kind == "cls"``    → ``head(base.extract_cls(x))``
    * ``head.input_kind == "tokens"`` → ``head(base.extract_tokens(x))``

    The base extractors run under ``torch.no_grad()`` (backbone frozen).
    Only ``head.parameters()`` are trainable — the typical optimiser
    setup is ``AdamW(model.head.parameters(), ...)``.
    """

    def __init__(self, base: Base, head: Head) -> None:
        super().__init__()
        self.backbone = base.backbone   # registered submodule
        self.head = head
        # Cache the extractor closure once; depends only on head kind.
        if head.input_kind == "cls":
            self._extract = base.extract_cls
        elif head.input_kind == "tokens":
            self._extract = base.extract_tokens
        else:
            raise ValueError(
                f"unknown head.input_kind {head.input_kind!r}; "
                "expected 'cls' or 'tokens'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._extract(x))

    def __getattr__(self, name: str):
        # Standard nn.Module attribute lookup first — handles parameters,
        # buffers, registered submodules (incl. self.backbone, self.head).
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Fall through to the backbone so existing concept classes that
        # do ``model.blocks``, ``model.cls_token``, etc. resolve.
        backbone = self._modules.get("backbone")
        if backbone is not None:
            return getattr(backbone, name)
        raise AttributeError(name)
