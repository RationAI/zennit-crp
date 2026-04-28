"""Generator for ``walkthrough.ipynb`` — keep notebook cells in version-controlled
source so edits are reviewable as plain text.

Run::

    uv run python tutorials/vit_crp/_build_notebook.py

Re-emits ``walkthrough.ipynb`` next to this file.
"""
from __future__ import annotations

import json
from pathlib import Path


_NEXT_ID = [0]


def _id() -> str:
    _NEXT_ID[0] += 1
    return f"cell-{_NEXT_ID[0]:03d}"


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": [l + "\n" for l in lines][:-1] + [lines[-1]] if lines else [""],
    }


def code(*lines: str) -> dict:
    src = list(lines)
    return {
        "cell_type": "code",
        "id": _id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in src[:-1]] + ([src[-1]] if src else []),
    }


CELLS: list[dict] = []


# ─── Title ────────────────────────────────────────────────────────────────────


CELLS.append(md(
    "# Vision-Transformer CRP — End-to-End Walkthrough",
    "",
    "This notebook is a complete tour of the four ViT concept-detector classes "
    "added in this fork:",
    "",
    "| Concept class | Granularity | `attribute()` shape |",
    "|---|---|---|",
    "| `HeadConcept`     | one concept per attention head                       | `(B, num_heads)` |",
    "| `KQVConcept`      | three concepts per block (whole Q / K / V)           | `(B, 3)` |",
    "| `KQVHeadConcept`  | per `(part, head)` — `3 × num_heads`                 | `(B, 3, num_heads)` |",
    "| `HeadDimConcept`  | per `(part, head, dim)` — `3 × num_heads × head_dim` | `(B, 3, num_heads, head_dim)` |",
    "",
    "All four hook the same named tap (`attn.qkv_tap`, an `nn.Identity` injected "
    "between `qkv` Linear and the reshape in timm `Attention.forward`). They differ "
    "only in (a) which slice of the `(B, N, 3·D)` tap they mask, and (b) which axes "
    "they sum over to produce per-concept relevance.",
    "",
    "**You will**:",
    "1. Set up the env with `uv sync`",
    "2. Download an Imagenette subset (real ImageNet images, ten classes, ~98 MB)",
    "3. Build a FeatureVisualization index per concept granularity",
    "4. Pick a target image, find its top-k concepts at a chosen ViT block, and "
    "look at reference samples + conditional heatmaps for each granularity",
    "",
    "**Theory references**: AttnLRP (Achtibat et al., ICML 2024; "
    "[arXiv 2402.05602](https://arxiv.org/abs/2402.05602)) on top of CRP "
    "(Achtibat et al., Nat. MI 2023; [arXiv 2206.03208](https://arxiv.org/abs/2206.03208))."
))


# ─── Setup ────────────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 1. Setup",
    "",
    "From the repo root, install dependencies into a uv-managed virtual env:",
    "",
    "```bash",
    "uv sync --extra vit --extra dev --extra notebook",
    "```",
    "",
    "Then launch this notebook with that env's kernel. If you see import errors "
    "below, you're on the wrong kernel."
))


CELLS.append(code(
    "from __future__ import annotations",
    "import os",
    "import sys",
    "import shutil",
    "import urllib.request",
    "import tarfile",
    "from pathlib import Path",
    "",
    "import numpy as np",
    "import torch",
    "import matplotlib.pyplot as plt",
    "from PIL import Image",
    "",
    "import timm",
    "from timm.data import resolve_data_config, create_transform",
    "from torch.utils.data import Dataset",
    "",
    "from crp.attention_concepts import (",
    "    HeadConcept,",
    "    KQVConcept,",
    "    KQVHeadConcept,",
    "    HeadDimConcept,",
    "    PARTS,",
    ")",
    "from crp.attribution import CondAttribution",
    "from crp.transformer_patches import AttnLRPEpsilonComposite",
    "from crp.visualization import FeatureVisualization",
    "",
    "torch.set_grad_enabled(True)",
    "print('torch', torch.__version__, '| timm', timm.__version__)"
))


# ─── Configuration ────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 2. Configuration",
    "",
    "All run-time knobs in one place. Override here if you have a GPU or want a "
    "bigger model.",
    "",
    "* `MODEL_NAME` — `vit_tiny_patch16_224` (5.7 M params) is CPU-friendly; "
    "`vit_small_patch16_224` (22 M) and `vit_base_patch16_224` (86 M) give "
    "better-localised concepts.",
    "* `NUM_SAMPLES` — how many Imagenette images to index. 64–128 is plenty to "
    "see meaningful per-concept top-k samples; the FV index runs forward + "
    "backward once per image.",
    "* `BLOCK_INDEX` — which ViT block to attribute. Mid-network blocks (5–8 in "
    "a 12-block ViT) typically encode the cleanest object-level concepts."
))


CELLS.append(code(
    "MODEL_NAME = 'vit_base_patch16_224'  # try 'vit_small_patch16_224' / 'vit_tiny_patch16_224' on CPU",
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "",
    "NUM_SAMPLES = 64",
    "BLOCK_INDEX = 6",
    "TOP_K = 4",
    "",
    "TUTORIAL_DIR = Path('tutorials/vit_crp').resolve()",
    "DATA_DIR = TUTORIAL_DIR / 'data'",
    "FV_ROOT = TUTORIAL_DIR / 'FeatureVisualization'",
    "DATA_DIR.mkdir(parents=True, exist_ok=True)",
    "FV_ROOT.mkdir(parents=True, exist_ok=True)",
    "",
    "print(f'device  : {DEVICE}')",
    "print(f'model   : {MODEL_NAME}')",
    "print(f'samples : {NUM_SAMPLES}')",
    "print(f'block   : {BLOCK_INDEX}')",
    "print(f'paths   : data={DATA_DIR}\\n          fv  ={FV_ROOT}')"
))


# ─── Data ─────────────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 3. Data — Imagenette",
    "",
    "[Imagenette](https://github.com/fastai/imagenette) is a 10-class subset of "
    "ImageNet curated by fast.ai. The 160-pixel version is ~98 MB and has the "
    "real ImageNet WordNet IDs in folder names so we can map back to the "
    "1000-class index our pretrained ViT was trained on.",
    "",
    "If you already have the tarball, drop it in `tutorials/vit_crp/data/` and "
    "this cell will skip the download."
))


CELLS.append(code(
    "IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'",
    "ARCHIVE = DATA_DIR / 'imagenette2-160.tgz'",
    "EXTRACTED = DATA_DIR / 'imagenette2-160'",
    "",
    "",
    "def _download_with_progress(url: str, dest: Path) -> None:",
    "    if dest.exists():",
    "        print(f'  exists: {dest}')",
    "        return",
    "    print(f'  fetching {url}')",
    "",
    "    last = [0]",
    "    def report(block, block_size, total):",
    "        if total <= 0:",
    "            return",
    "        pct = int(100 * block * block_size / total)",
    "        if pct >= last[0] + 5:",
    "            last[0] = pct",
    "            print(f'    {pct:3d}%  {block * block_size / 1e6:7.1f} MB / {total / 1e6:.1f} MB')",
    "    urllib.request.urlretrieve(url, dest, reporthook=report)",
    "",
    "",
    "def _extract(archive: Path, target_dir: Path) -> None:",
    "    if (target_dir).exists():",
    "        print(f'  extracted: {target_dir}')",
    "        return",
    "    print(f'  extracting {archive.name}')",
    "    with tarfile.open(archive, 'r:gz') as tf:",
    "        tf.extractall(target_dir.parent)",
    "",
    "",
    "_download_with_progress(IMAGENETTE_URL, ARCHIVE)",
    "_extract(ARCHIVE, EXTRACTED)",
    "print('imagenette ready at', EXTRACTED)"
))


CELLS.append(md(
    "### 3.1 Imagenette → ImageNet-1k label mapping",
    "",
    "Imagenette folder names are ImageNet WordNet IDs. We need the integer "
    "class index that our pretrained ViT outputs."
))


CELLS.append(code(
    "# fast.ai's 10 classes mapped to ImageNet-1k indices",
    "IMAGENETTE_TO_IMAGENET = {",
    "    'n01440764': 0,    # tench",
    "    'n02102040': 217,  # English springer",
    "    'n02979186': 482,  # cassette player",
    "    'n03000684': 491,  # chain saw",
    "    'n03028079': 497,  # church",
    "    'n03394916': 566,  # French horn",
    "    'n03417042': 569,  # garbage truck",
    "    'n03425413': 571,  # gas pump",
    "    'n03445777': 574,  # golf ball",
    "    'n03888257': 701,  # parachute",
    "}",
    "CLASS_NAMES = {",
    "    0:'tench', 217:'English springer', 482:'cassette player',",
    "    491:'chain saw', 497:'church', 566:'French horn', 569:'garbage truck',",
    "    571:'gas pump', 574:'golf ball', 701:'parachute',",
    "}",
))


# ─── Model ────────────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 4. Model + AttnLRP composite",
    "",
    "Idiomatic zennit: no model-time patching. The composite carries a "
    "`TimmViTCanonizer` that, when registered on `composite.context()`, ",
    "1. installs a child `qkv_tap = nn.Identity()` on every timm `Attention` "
    "(the named hook point used by all four concept classes), ",
    "2. swaps `forward` per-instance on `Attention` / `LayerNorm` / `GELU` / "
    "`Dropout` to embed the AttnLRP autograd rules (Q/K/V uniform-rule factors "
    "4, 4, 2; identity rule on activations).  ",
    "All mutations are reversed when `composite.context()` exits — no "
    "process-global state."
))


CELLS.append(code(
    "model = timm.create_model(MODEL_NAME, pretrained=True).eval().to(DEVICE)",
    "",
    "block = model.blocks[BLOCK_INDEX].attn",
    "NUM_HEADS, HEAD_DIM = block.num_heads, block.head_dim",
    "LAYER_NAME = f'blocks.{BLOCK_INDEX}.attn.qkv_tap'",
    "",
    "print(f'layer    : {LAYER_NAME}')",
    "print(f'num_heads: {NUM_HEADS}')",
    "print(f'head_dim : {HEAD_DIM}')",
    "print(f'concept counts:')",
    "print(f'  HeadConcept    -> {NUM_HEADS}')",
    "print(f'  KQVConcept     -> 3')",
    "print(f'  KQVHeadConcept -> {3 * NUM_HEADS}')",
    "print(f'  HeadDimConcept -> {3 * NUM_HEADS * HEAD_DIM}')"
))


# ─── Dataset ──────────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 5. Dataset wrapper",
    "",
    "`FeatureVisualization` expects a `Dataset` whose `__getitem__` returns "
    "`(unpreprocessed_tensor, int_target)`. The preprocessing (mean/std "
    "normalisation matching our timm model) is applied inside FV via the "
    "`preprocess_fn` argument — that way the unpreprocessed tensor can be plotted "
    "directly as an RGB image."
))


CELLS.append(code(
    "import torchvision.transforms as T",
    "",
    "cfg = resolve_data_config({}, model=model)",
    "MEAN, STD, IMG_SIZE = cfg['mean'], cfg['std'], cfg['input_size'][1]",
    "print('mean', MEAN, '| std', STD, '| size', IMG_SIZE)",
    "",
    "# resize/crop only — no normalisation (FV.preprocess_fn handles it).",
    "to_tensor = T.Compose([",
    "    T.Resize(int(IMG_SIZE * 256 / 224)),",
    "    T.CenterCrop(IMG_SIZE),",
    "    T.ToTensor(),",
    "])",
    "",
    "MEAN_T = torch.tensor(MEAN).view(1, -1, 1, 1)",
    "STD_T = torch.tensor(STD).view(1, -1, 1, 1)",
    "",
    "",
    "def preprocess_fn(x: torch.Tensor) -> torch.Tensor:",
    "    return (x - MEAN_T.to(x)) / STD_T.to(x)",
    "",
    "",
    "def denormalize(x: torch.Tensor) -> np.ndarray:",
    "    if x.dim() == 3:",
    "        x = x.unsqueeze(0)",
    "    return x.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()",
    "",
    "",
    "class ImagenetteDataset(Dataset):",
    "    def __init__(self, root: Path, num_samples: int):",
    "        files = []",
    "        targets = []",
    "        for wnid_dir in sorted((root / 'val').iterdir()):",
    "            label = IMAGENETTE_TO_IMAGENET[wnid_dir.name]",
    "            for f in sorted(wnid_dir.glob('*.JPEG')):",
    "                files.append(f)",
    "                targets.append(label)",
    "        # Shuffle deterministically and trim",
    "        rng = np.random.default_rng(0)",
    "        order = rng.permutation(len(files))[:num_samples]",
    "        self.files = [files[i] for i in order]",
    "        self.targets = [targets[i] for i in order]",
    "",
    "    def __len__(self):",
    "        return len(self.files)",
    "",
    "    def __getitem__(self, i):",
    "        img = Image.open(self.files[i]).convert('RGB')",
    "        return to_tensor(img), int(self.targets[i])",
    "",
    "",
    "dataset = ImagenetteDataset(EXTRACTED, NUM_SAMPLES)",
    "print(f'dataset size: {len(dataset)}')",
    "print(f'class distribution: {dict(zip(*np.unique(dataset.targets, return_counts=True)))}')"
))


CELLS.append(md(
    "### 5.1 Sanity check — pretrained model agrees on a sample",
))


CELLS.append(code(
    "sample, target = dataset[0]",
    "sample_pre = preprocess_fn(sample.unsqueeze(0)).to(DEVICE)",
    "with torch.no_grad():",
    "    pred = model(sample_pre)[0].softmax(dim=-1)",
    "top5 = pred.topk(5)",
    "",
    "fig, ax = plt.subplots(figsize=(3, 3))",
    "ax.imshow(denormalize(sample))",
    "ax.set_title(f'true: {CLASS_NAMES.get(target, target)}\\nfile: {dataset.files[0].name}', fontsize=9)",
    "ax.axis('off')",
    "plt.show()",
    "",
    "print('top-5 model predictions:')",
    "for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):",
    "    name = CLASS_NAMES.get(idx, '')",
    "    mark = ' <- target' if idx == target else ''",
    "    print(f'  cls {idx:4d}  p={prob:.3f}  {name}{mark}')"
))


# ─── FV indices ──────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 6. Build a FeatureVisualization index for each concept granularity",
    "",
    "For each of the four concept classes we build a separate index so the "
    "per-layer .npy files don't clash. The index records, for every concept id, "
    "the top-N samples in the dataset that maximise its **relevance** under each "
    "sample's true class. (`run` also tracks activation, but for ViT taps "
    "activation isn't a meaningful proxy — relevance is the operative signal.)",
    "",
    "Each call runs forward + backward through the model once per sample, with "
    "hooks recording the masked relevance at `qkv_tap`. With 64 samples on CPU "
    "this is a few minutes per concept."
))


CELLS.append(code(
    "CONCEPT_DEFS = {",
    "    'head':     HeadConcept,",
    "    'kqv':      KQVConcept,",
    "    'kqv_head': KQVHeadConcept,",
    "    'head_dim': HeadDimConcept,",
    "}",
    "",
    "composite = AttnLRPEpsilonComposite()",
    "attribution = CondAttribution(model, device=torch.device(DEVICE))",
    "",
    "concepts = {}",
    "fvs = {}",
    "for name, cls in CONCEPT_DEFS.items():",
    "    concept = cls()",
    "    concept.register_from_model(model)",
    "    concepts[name] = concept",
    "    fvs[name] = FeatureVisualization(",
    "        attribution,",
    "        dataset,",
    "        layer_map={LAYER_NAME: concept},",
    "        preprocess_fn=preprocess_fn,",
    "        path=str(FV_ROOT / name),",
    "        device=torch.device(DEVICE),",
    "    )",
    "",
    "print('concepts registered for layer', LAYER_NAME)"
))


CELLS.append(code(
    "%%time",
    "for name, fv in fvs.items():",
    "    print(f'\\n=== running FV index for {name!r} ===')",
    "    fv.run(composite, 0, len(dataset), batch_size=8, checkpoint=10000)",
    "print('\\nall four indices built.')"
))


# ─── Inspecting concepts on a target image ───────────────────────────────────


CELLS.append(md(
    "## 7. Inspect concepts on a target image",
    "",
    "Pick one image, compute a per-concept relevance ranking under the image's "
    "true class, take the top-k concepts, and for each one fetch:",
    "1. the **reference samples** — images from the index that most activate "
    "that concept (`get_max_reference`)",
    "2. a **conditional heatmap** — pixel-space attribution under that concept "
    "alone (start the backward pass at `qkv_tap`, masked to the concept)"
))


CELLS.append(code(
    "TARGET_INDEX = 0  # index into the dataset",
    "target_data, target_class = dataset[TARGET_INDEX]",
    "target_pre = preprocess_fn(target_data.unsqueeze(0)).to(DEVICE)",
    "target_pre.requires_grad_(True)",
    "",
    "print(f'target sample: {dataset.files[TARGET_INDEX].name}')",
    "print(f'true class   : {target_class} ({CLASS_NAMES.get(target_class, \"?\")})')"
))


CELLS.append(md(
    "### 7.1 Per-concept top-k under the true class",
    "",
    "Run one backward pass per concept, recording relevance at `qkv_tap`, then "
    "use `concept.attribute()` to aggregate to per-concept-id scores."
))


CELLS.append(code(
    "def per_concept_scores(concept, layer_name: str, data: torch.Tensor, target_class: int):",
    "    conditions = [{'y': [target_class]}]",
    "    result = attribution(",
    "        data,",
    "        conditions,",
    "        composite,",
    "        mask_map=concept.mask,",
    "        record_layer=[layer_name],",
    "    )",
    "    rel = result.relevances[layer_name]",
    "    return concept.attribute(rel, layer_name=layer_name, abs_norm=False)[0]",
    "",
    "",
    "def top_k_flat(scores: torch.Tensor, k: int) -> list[int]:",
    "    flat = scores.flatten()",
    "    k = min(k, flat.numel())",
    "    return torch.topk(flat.abs(), k=k).indices.tolist()",
    "",
    "",
    "def label_for(name: str, flat_id: int) -> str:",
    "    if name == 'head':",
    "        return f'h{flat_id}'",
    "    if name == 'kqv':",
    "        return PARTS[flat_id]",
    "    if name == 'kqv_head':",
    "        p, h = divmod(flat_id, NUM_HEADS)",
    "        return f'{PARTS[p]}/h{h}'",
    "    if name == 'head_dim':",
    "        p, rem = divmod(flat_id, NUM_HEADS * HEAD_DIM)",
    "        h, d = divmod(rem, HEAD_DIM)",
    "        return f'{PARTS[p]}/h{h}/d{d}'",
    "    raise ValueError(name)",
    "",
    "",
    "top_ids: dict[str, list[int]] = {}",
    "for name, concept in concepts.items():",
    "    target_pre.grad = None  # fresh grad for each pass",
    "    scores = per_concept_scores(concept, LAYER_NAME, target_pre, target_class)",
    "    ids = top_k_flat(scores, TOP_K)",
    "    top_ids[name] = ids",
    "    pretty = ', '.join(label_for(name, i) for i in ids)",
    "    print(f'{name:>10s}: {pretty}')"
))


CELLS.append(md(
    "### 7.2 Reference samples per concept",
    "",
    "For each granularity, pull the top-N samples that maximise each top-k "
    "concept's relevance over the indexed dataset."
))


CELLS.append(code(
    "REF_RANGE = (0, 4)  # top-1..top-4 sample per concept",
    "",
    "ref_grids = {}",
    "for name, ids in top_ids.items():",
    "    ref_c = fvs[name].get_max_reference(",
    "        ids, LAYER_NAME, mode='relevance', r_range=REF_RANGE,",
    "        composite=None, plot_fn=None,",
    "    )",
    "    ref_grids[name] = ref_c",
    "print('reference samples loaded for all four concepts.')"
))


CELLS.append(code(
    "def plot_reference_grid(name: str, ref_c: dict, ids: list[int]):",
    "    n_top = REF_RANGE[1] - REF_RANGE[0]",
    "    fig, axes = plt.subplots(",
    "        len(ids), n_top, figsize=(2 * n_top, 2 * len(ids) + 0.4)",
    "    )",
    "    if len(ids) == 1:",
    "        axes = np.array([axes])",
    "    if n_top == 1:",
    "        axes = axes[:, None]",
    "    for r, cid in enumerate(ids):",
    "        samples = ref_c[cid]",
    "        for c in range(n_top):",
    "            ax = axes[r, c]",
    "            if c < samples.shape[0]:",
    "                ax.imshow(denormalize(samples[c]))",
    "            ax.axis('off')",
    "            if c == 0:",
    "                ax.set_ylabel(label_for(name, cid), fontsize=10)",
    "                ax.axis('on')",
    "                ax.set_xticks([]); ax.set_yticks([])",
    "    fig.suptitle(f'{name}: top-{n_top} reference samples per concept', fontsize=11)",
    "    plt.tight_layout()",
    "    plt.show()",
    "",
    "",
    "for name, ids in top_ids.items():",
    "    plot_reference_grid(name, ref_grids[name], ids)"
))


CELLS.append(md(
    "### 7.3 Conditional heatmaps on the target image",
    "",
    "For each top-k concept, run an attribution masked to that concept under the "
    "target class. The heatmap is the input-space relevance — i.e. *where* on "
    "the image this concept is looking."
))


CELLS.append(code(
    "def conditional_heatmap(concept, layer_name: str, concept_id, data: torch.Tensor, target_class: int):",
    "    conditions = [{layer_name: [concept_id], 'y': [target_class]}]",
    "    result = attribution(data, conditions, composite, mask_map=concept.mask)",
    "    hm = result.heatmap[0]",
    "    if hm.dim() == 3:",
    "        hm = hm.sum(dim=0)",
    "    return hm.detach().cpu().numpy()",
    "",
    "",
    "img_np = denormalize(target_data)",
    "n_rows = len(top_ids)",
    "n_cols = max(len(ids) for ids in top_ids.values()) + 1",
    "fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows + 0.4))",
    "if n_rows == 1:",
    "    axes = np.array([axes])",
    "",
    "for r, (name, ids) in enumerate(top_ids.items()):",
    "    axes[r, 0].imshow(img_np)",
    "    axes[r, 0].set_title(name, fontsize=10)",
    "    axes[r, 0].axis('off')",
    "    for c, cid in enumerate(ids, start=1):",
    "        target_pre.grad = None",
    "        hm = conditional_heatmap(concepts[name], LAYER_NAME, cid, target_pre, target_class)",
    "        axes[r, c].imshow(img_np, alpha=0.4)",
    "        axes[r, c].imshow(hm, cmap='bwr', alpha=0.6, vmin=-np.abs(hm).max(), vmax=np.abs(hm).max())",
    "        axes[r, c].set_title(label_for(name, cid), fontsize=9)",
    "        axes[r, c].axis('off')",
    "    for c in range(len(ids) + 1, n_cols):",
    "        axes[r, c].axis('off')",
    "",
    "fig.suptitle(",
    "    f'conditional heatmaps  •  layer={LAYER_NAME}  •  '",
    "    f'target={CLASS_NAMES.get(target_class, target_class)}',",
    "    fontsize=11,",
    ")",
    "plt.tight_layout()",
    "plt.show()"
))


CELLS.append(md(
    "## 8. What's next",
    "",
    "* Try a different `BLOCK_INDEX` — early blocks (0–3) tend to encode "
    "low-level features (edges, color); late blocks (9–11) encode "
    "object-/class-level semantics.",
    "* Swap the target image (`TARGET_INDEX`) — concepts will shift.",
    "* Compare granularities side-by-side: a `kqv_head` concept's heatmap "
    "should be a strict refinement of the `kqv` concept it belongs to "
    "(same part, narrower).",
    "* Use `compute_stats` to find the dataset class for which each concept "
    "is most representative.",
    "* Read [`tutorials/vit_crp/metrics.py`](metrics.py) for the deletion / "
    "insertion AUC faithfulness benchmark across the four granularities.",
    "",
    "**Faithfulness caveat**: `AttnLRPEpsilonComposite` wires in the AttnLRP "
    "uniform rule on Q/K/V and the identity rule on activations, but the "
    "Linear layers in MLPs and the patch-embed Conv use plain ε-LRP rather "
    "than the γ-LRP variant recommended by AttnLRP §3.2.1 (γ ≈ 0.25). On "
    "vit_tiny we see random-concept baselines beat true top-k on deletion AUC "
    "for the finer granularities — adding a γ rule is the next iteration."
))


# ─── Emit ─────────────────────────────────────────────────────────────────────


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "walkthrough.ipynb"
    out.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
