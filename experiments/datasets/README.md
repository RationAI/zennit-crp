# `experiments/datasets/` — per-dataset auto-setup loaders

One Python file per dataset. Each file owns the **download, extract, and
load** logic for a single dataset; nothing is shared across datasets
beyond the lightweight `CuratedDataset` dataclass (`_legacy.py`) and the
unified `load(name, ...)` dispatcher in `__init__.py`.

## Design rules

1. **Auto-setup**. The user must be able to:

   ```python
   from datasets.funny_birds import FunnyBirdsDataset
   ds = FunnyBirdsDataset(root="data", split="train", transform=...)
   ```

   and the class handles download + extract + load on first call.
   No manual setup steps unless technically impossible (e.g., login-gated
   datasets like full ImageNet train).

2. **Idempotent on cache hit**. Subsequent constructions check the local
   cache (under `<root>/<dataset_name>/`) and skip the download.

3. **Resumable downloads** for large files. Use HTTP `Range` requests +
   `.part` files so an interrupted download doesn't force restarting from
   byte zero. See `funny_birds._stream_download` for the reference impl.

4. **One file per dataset.** Layout:

   ```
   datasets/
     __init__.py        # re-exports + load(name, ...) dispatcher
     _legacy.py         # CuratedDataset + Imagenette + ImageNet val (legacy)
     funny_birds.py     # FunnyBirdsDataset
     dsprites.py        # DSpritesDataset
     <new_dataset>.py   # add new datasets here
   ```

5. **Class-API**. Each new dataset module exposes a `<Name>Dataset` class
   that is a `torch.utils.data.Dataset` subclass with the standard
   `__len__`/`__getitem__` interface. The class also exposes
   `num_classes`, `class_indices`, and `name` properties so it's a
   drop-in for code expecting a `CuratedDataset`.

6. **Pathlib for everything.** All paths are `pathlib.Path` instances
   anchored at a `root: Path` constructor argument. No string paths in
   public APIs.

7. **Use torchvision when available.** If a dataset is provided by
   `torchvision.datasets` (e.g. MNIST, CIFAR), wrap it instead of
   re-implementing the download. Only roll our own when torchvision
   doesn't ship it (FunnyBirds, dsprites, etc.).

## Available datasets

| name | size | classes | source | notes |
|---|---:|---:|---|---|
| `funny_birds` | ~1.5 GB | 50 | `download.visinf.tu-darmstadt.de` | Synthetic birds with GT part maps. Hesse et al. ICCV 2023, arXiv:2308.06248. |
| `dsprites` | ~26 MB | 3 (or 6/40/32) | `github.com/google-deepmind/dsprites-dataset` | 2D shapes with controlled latent factors. Higgins et al. ICLR 2017. |
| `imagenet_val_hf` | ~830 MB | 1000 | HF `evanarlian/imagenet_1k_resized_256` | Full ImageNet val, un-gated mirror. |
| `imagenet_val` | ~6.7 GB | 1000 | image-net.org (gated) | Manual setup required — see `_legacy.py`. |
| `imagenette` | ~98 MB | 10 (mapped to ImageNet-1k indices) | fast.ai S3 mirror | 10-class ImageNet subset for quick smoke tests. |

## Adding a new dataset

1. Create `experiments/datasets/<dataset_name>.py`.
2. Define `<DATASET_NAME>_DOWNLOAD_URL` and any layout constants.
3. Subclass `torch.utils.data.Dataset` (use `@dataclass` for clean
   constructor signatures; see `funny_birds.py` for a template).
4. Implement `__post_init__` that:
   - resolves cache paths under `Path(root) / "<dataset_name>"`
   - downloads if missing (use the `_stream_download` resume pattern)
   - extracts if needed
   - reads the manifest into `self.items` (or equivalent)
5. Implement `__len__`, `__getitem__`, plus `num_classes`,
   `class_indices`, `name` properties.
6. Re-export the class from `__init__.py`'s `from .<dataset_name>
   import <Name>Dataset` block.
7. Register the class in `__init__.py`'s `DATASETS` dict so the
   `load(name)` dispatcher knows about it.
8. Test end-to-end by training the probe head with
   `experiments/train_dinov3_probe.py --dataset <name>` (the trainer
   auto-runs a held-out test sanity check; failures abort with a
   threshold message).

## Verifying a dataset / probe end-to-end

The `train_dinov3_probe.py` script doubles as the end-to-end test
harness. It:

1. Loads the dataset via `load(name)` (triggers auto-setup if missing)
2. Extracts DINOv3 features over the full split (caches them)
3. Stratified-splits the features into train/test (default 90/10)
4. Trains the linear head on the train split
5. Evaluates on the held-out test split — top-1, top-5, cross-entropy
6. **Aborts with a non-zero exit code** if test top-1 falls below a
   per-dataset minimum threshold (defaults: funny_birds=0.30,
   dsprites=0.85, imagenette=0.70, imagenet_val_hf=0.40)

This is the contract for "the trained model head is well finetuned":
top-1 above the threshold + a printed train/test gap. The DINOv3
backbone stays frozen throughout (only the linear head is updated).
