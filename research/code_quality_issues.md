Code-quality cleanups found while building the SAE-downstream experiment (`experiments/sae_downstream.py`) and refactoring the CRP gallery. Each is concrete and independent.

### 1. `DataLoader(num_workers=4)` forkserver stall (data-loss footgun)
On the GPU cluster, a `DataLoader` with `num_workers>0` **hangs at startup** (forkserver worker spawn) once CUDA is initialized in the parent process — the run sits forever in "collecting…" with no error. Hit this repeatedly.
- `experiments/sae.py:170` — `num_workers=4`
- `experiments/head_relevance_by_class.py:79` — `num_workers=2`
- (already fixed to `0` in `experiments/sae_downstream.py:212,450,497`)
Fix: default these collection loaders to `num_workers=0`, or set a `multiprocessing_context="spawn"` / guard. The training loaders in `train_probe.py` expose a `--num-workers` knob (fine); the issue is the hard-coded ones in attribution/collection paths.

### 2. Duplicated dataset registries with divergent key spellings
- `experiments/model_io.py` `DATASETS` (eval): keys `funny_birds`, `dsprites`, `colored_mnist`, each carrying a probe-tag third element.
- `experiments/train_probe.py:116` `TRAIN_DATASETS` (train): keys `funny-birds-train-clean`, `dsprites`, `colored-mnist-train`, …
Two sources of truth, mapping between them implicit (the probe-tag). Unify into one registry (name → loader/kwargs/tag) consumed by both train and eval.

### 3. Duplicated site→layer mapping
`experiments/sae.py` `site_modules(model, site)` and `experiments/model_io.py` `site_layer_names(model, site)` encode the same `proj_drop`/`residual` block-site logic (one returns modules, the other names). Derive one from the other.

### 4. Missing dependency
`experiments/sae_downstream.py` uses `scikit-learn` (kNN purity / silhouette / linear-probe / TSNE) but it was not in `pyproject` (added in passing). Audit `[project.dependencies]` / the `vit` extra so a fresh `uv sync` runs the experiments without `ModuleNotFoundError`.

### 5. Representative-sample selection assumes a shuffled dataset
Sequential first-N capture (e.g. rep/eval subsets) on a **class-grouped** dataset (FunnyBirds train is ordered by class) yields only the first few classes — produced degenerate, few-color manifold plots. Rep/eval subset selection should shuffle or stratify across classes.

_Filed by an automated Claude agent on behalf of @AdamBajger._
