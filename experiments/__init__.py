"""Experiment harness for zennit-crp.

Composable building blocks for explainability work:

* :mod:`experiments.models`   — frozen ViT bases + trainable heads,
  composed via :func:`~experiments.models.build_probe`.
* :mod:`experiments.datasets` — auto-downloading dataset loaders.
* :mod:`experiments.train_probe` — typer CLI (``cache`` + ``train``).

Importable as ``from experiments.<sub> import ...`` once the project is
installed (``uv sync``). No ``sys.path`` manipulation required at
notebook or script entry points.
"""
