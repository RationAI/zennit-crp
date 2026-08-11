# Experiments

Composable building blocks for the explainability work in this repo:

* [`models/`](models/) — frozen ViT bases (`vit_base`, `vit_dinov3`) and
  trainable heads (`linear`, `attentive`), composed via
  `build_probe(...)`. Same registry the CLI and walkthrough notebook
  consume.
* [`datasets/`](datasets/) — auto-downloading dataset loaders
  (`imagenette`, `imagenet_val_hf`, `funny_birds`, `dsprites`).
* [`train_probe.py`](train_probe.py) — typer CLI with two commands:
  `cache` (one-shot feature extraction) and `train` (head training on
  cached features).
* [`funnybirds_part_alignment.py`](funnybirds_part_alignment.py) —
  attribution-on-parts vs background metric, FunnyBirds-only.

All scripts read and write under [`<repo>/data/`](../data) (gitignored).
Run them from the repo root via `uv run python experiments/<script>.py`.

## Training a probe — the two-step recipe

Cache features once per `(base, dataset, kind)`:

```
uv run train-probe cache vit_dinov3 funny_birds --kind cls
uv run train-probe cache vit_dinov3 funny_birds --kind tokens
```

Train heads on top:

```
uv run train-probe train vit_dinov3 linear    funny_birds
uv run train-probe train vit_dinov3 attentive funny_birds --num-heads 8
```

Output paths follow `data/<base>_<head>_probe_<dataset>.pt`. The
walkthrough notebook in
[`tutorials/vit_crp/dinov3_unfolded/`](../tutorials/vit_crp/dinov3_unfolded/)
auto-loads them via the same `build_probe(...)` registry.

## Adding a new base or head

* New base — drop `models/bases/<name>.py` subclassing `Base` and
  setting `timm_name`. Register in `models/__init__.py::BASES`.
* New head — drop `models/heads/<name>.py` subclassing `Head` and
  setting `input_kind` (`"cls"` or `"tokens"`). Register in
  `models/__init__.py::HEADS`. If the head takes constructor params,
  add them to the `train` typer command and forward them via
  `head_kwargs`.

Both the training CLI and the notebook pick up new entries
automatically — they iterate the registry.
