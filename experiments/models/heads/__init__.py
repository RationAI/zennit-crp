"""Trainable classification heads. Each head declares ``input_kind``
to tell the training pipeline which feature cache to load."""
from .base import Head
from .linear import LinearHead
from .attentive import AttentiveHead
from .block import BlockHead

__all__ = ["Head", "LinearHead", "AttentiveHead", "BlockHead"]
