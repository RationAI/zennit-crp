"""Generator for ``walkthrough.ipynb``.

Keep notebook source under version control as plain Python so reviews and
diffs are tractable. Run from the repo root::

    uv run python tutorials/vit_crp/_build_notebook.py

Re-emits ``walkthrough.ipynb`` next to this file. Structure mirrors the
original CRP repo's ``tutorials/{attributions,feature_visualization}.ipynb``:
single attribution → FeatureVisualization indexing → reference samples →
conditional heatmaps — but for ViTs and the four concept granularities in
this fork.
"""
from __future__ import annotations

import json
from pathlib import Path

_NEXT_ID = [0]


def _id() -> str:
    _NEXT_ID[0] += 1
    return f"cell-{_NEXT_ID[0]:03d}"


def md(*lines: str) -> dict:
    if not lines:
        return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": [""]}
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": [l + "\n" for l in lines][:-1] + [lines[-1]],
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
    "# Vision-Transformer CRP — Walkthrough",
    "",
    "End-to-end tutorial for the four ViT concept-detector classes added in "
    "this fork, structured to mirror the original CRP repo's "
    "[`tutorials/attributions.ipynb`](../attributions.ipynb) and "
    "[`tutorials/feature_visualization.ipynb`](../feature_visualization.ipynb).",
    "",
    "## What you'll see",
    "",
    "1. **Setup** — imports, configuration knobs, paths.",
    "2. **Dataset** — Imagenette-160 (10-class ImageNet subset, ~98 MB) with "
    "ImageNet-1k label mapping.",
    "3. **Model + Canonizer + Composite** — load `vit_base_patch16_224`, "
    "build an `AttnLRPGammaComposite` (canonizer pre-bundled, no model-time "
    "patching), inspect what the canonizer does.",
    "4. **Single-image conditional attribution** — pick a configurable target "
    "image, run a `HeadConcept`-conditioned backward pass, plot the heatmap.",
    "5. **Build a `FeatureVisualization` index per concept granularity** — "
    "cached on disk; re-runs are no-ops.",
    "6. **Top-concept identification + reference samples** — for each "
    "granularity (`HeadConcept`, `KQVConcept`, `KQVHeadConcept`, "
    "`HeadDimConcept`), rank concepts under the target class, fetch the "
    "top-N samples that maximise each concept's relevance.",
    "7. **Conditional heatmaps on the target image** — pixel-space "
    "attribution under the most-important concept of each granularity, "
    "side by side.",
    "",
    "**Theory**: AttnLRP (Achtibat et al., ICML 2024; "
    "[arXiv 2402.05602](https://arxiv.org/abs/2402.05602)) on top of CRP "
    "(Achtibat et al., Nature MI 2023; "
    "[arXiv 2206.03208](https://arxiv.org/abs/2206.03208)).",
    "",
    "**Concept-detector cheat sheet**:",
    "",
    "| Class | Granularity | `attribute()` shape |",
    "|---|---|---|",
    "| `HeadConcept`     | one concept per attention head                       | `(B, num_heads)` |",
    "| `KQVConcept`      | three concepts per block (whole Q / K / V)           | `(B, 3)` |",
    "| `KQVHeadConcept`  | per `(part, head)` — `3 × num_heads`                 | `(B, 3, num_heads)` |",
    "| `HeadDimConcept`  | per `(part, head, dim)` — `3 × num_heads × head_dim` | `(B, 3, num_heads, head_dim)` |",
    "",
    "All four hook the same named tap (`attn.qkv_tap`) installed by "
    "`QKVTapCanonizer`."
))


# ─── 1. Setup ─────────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 1. Setup",
    "",
    "From the repo root:",
    "",
    "```bash",
    "uv sync --extra vit --extra dev --extra notebook",
    "```",
    "",
    "then launch this notebook with that env's kernel."
))


CELLS.append(code(
    "from __future__ import annotations",
    "import os",
    "import urllib.request",
    "import tarfile",
    "from pathlib import Path",
    "",
    "import numpy as np",
    "import torch",
    "import torchvision.transforms as T",
    "import matplotlib.pyplot as plt",
    "from PIL import Image",
    "",
    "import timm",
    "from timm.data import resolve_data_config",
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
    "from crp.transformer_patches import (",
    "    AttnLRPEpsilonComposite,",
    "    AttnLRPGammaComposite,",
    "    QKVTapCanonizer,",
    "    TimmViTCanonizer,",
    ")",
    "from crp.visualization import FeatureVisualization",
    "from crp.image import plot_grid, vis_opaque_img",
    "",
    "torch.set_grad_enabled(True)",
    "print('torch', torch.__version__, '| timm', timm.__version__)"
))


CELLS.append(md(
    "### 1.1 Configuration",
    "",
    "All run-time knobs in one place. Override here for a GPU run or a bigger model.",
    "",
    "* `MODEL_NAME` — `vit_base_patch16_224` (86 M) is the AttnLRP-paper default; "
    "`vit_small_patch16_224` (22 M) and `vit_tiny_patch16_224` (5.7 M) are "
    "CPU-friendly.",
    "* `NUM_SAMPLES` — Imagenette images to index. 64–128 is plenty.",
    "* `BLOCK_INDEX` — which ViT block to attribute. Mid-network blocks "
    "(5–8 in a 12-block ViT) carry the cleanest object-level concepts.",
    "* `TARGET_INDEX` — index of the image we'll attribute. `None` → pick "
    "randomly from `RANDOM_SEED`.",
    "* `GAMMA` — γ for the γ-LRP rule on linears (AttnLRP §3.2.1, default "
    "0.25). Set `USE_GAMMA = False` to fall back to ε-LRP."
))


CELLS.append(code(
    "MODEL_NAME = 'vit_base_patch16_224'   # 'vit_small_patch16_224' / 'vit_tiny_patch16_224' on CPU",
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "",
    "NUM_SAMPLES = 64",
    "BLOCK_INDEX = 6",
    "TOP_K = 4",
    "TARGET_INDEX = None  # int → pick that index; None → random under RANDOM_SEED",
    "RANDOM_SEED = 0",
    "",
    "USE_GAMMA = True",
    "GAMMA = 0.25",
    "EPSILON = 1e-6",
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
    "print(f'rule    : {(\"γ-LRP, γ=\" + str(GAMMA)) if USE_GAMMA else \"ε-LRP\"}')"
))


# ─── 2. Dataset ───────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 2. Dataset — Imagenette",
    "",
    "[Imagenette](https://github.com/fastai/imagenette) is a 10-class subset of "
    "ImageNet (fast.ai). The 160-pixel version is ~98 MB and uses real ImageNet "
    "WordNet IDs in folder names so we can map back to the 1000-class indices "
    "the pretrained ViT was trained on."
))


CELLS.append(code(
    "IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'",
    "ARCHIVE = DATA_DIR / 'imagenette2-160.tgz'",
    "EXTRACTED = DATA_DIR / 'imagenette2-160'",
    "",
    "",
    "def _download(url, dest):",
    "    if dest.exists():",
    "        print(f'  exists: {dest}')",
    "        return",
    "    print(f'  fetching {url}')",
    "    last = [0]",
    "    def report(block, size, total):",
    "        if total <= 0: return",
    "        pct = int(100 * block * size / total)",
    "        if pct >= last[0] + 5:",
    "            last[0] = pct",
    "            print(f'    {pct:3d}%  {block*size/1e6:7.1f} MB / {total/1e6:.1f} MB')",
    "    urllib.request.urlretrieve(url, dest, reporthook=report)",
    "",
    "",
    "def _extract(archive, target_dir):",
    "    if target_dir.exists():",
    "        print(f'  extracted: {target_dir}')",
    "        return",
    "    with tarfile.open(archive, 'r:gz') as tf:",
    "        tf.extractall(target_dir.parent)",
    "",
    "",
    "_download(IMAGENETTE_URL, ARCHIVE)",
    "_extract(ARCHIVE, EXTRACTED)",
    "print('imagenette ready at', EXTRACTED)"
))


CELLS.append(md(
    "### 2.1 Class mapping (WordNet ID → ImageNet-1k index)",
))


CELLS.append(code(
    "IMAGENETTE_TO_IMAGENET = {",
    "    'n01440764':   0,  # tench",
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
    "    0: 'tench', 217: 'English springer', 482: 'cassette player',",
    "    491: 'chain saw', 497: 'church', 566: 'French horn',",
    "    569: 'garbage truck', 571: 'gas pump', 574: 'golf ball', 701: 'parachute',",
    "}"
))


# ─── 3. Model + Canonizer + Composite ────────────────────────────────────────


CELLS.append(md(
    "## 3. Model + Canonizer + Composite",
    "",
    "Standard zennit pipeline:",
    "",
    "* **Canonizers** modify the model graph and forward methods so standard "
    "LRP rules can apply. We use `TimmViTCanonizer`, which composes "
    "`QKVTapCanonizer` (adds a named `qkv_tap = nn.Identity()` to every "
    "Attention) with `AttributeCanonizer`s that swap `forward` per-instance "
    "on `Attention`, `LayerNorm`, `GELU`, `Dropout` to embed the AttnLRP "
    "autograd functions in the forward pass.",
    "* **Composite** maps module classes to LRP **Hooks**. We map `Linear` / "
    "`Conv2d` to a gradient×input ε or γ rule, and activations to `Pass` "
    "(the AttnLRP identity rule is already encoded in their forward via the "
    "canonizer).",
    "",
    "All registration is **scoped to `composite.context()`**. No process-global "
    "state, no monkey-patching."
))


CELLS.append(code(
    "model = timm.create_model(MODEL_NAME, pretrained=True).eval().to(DEVICE)",
    "",
    "block = model.blocks[BLOCK_INDEX].attn",
    "NUM_HEADS, HEAD_DIM = block.num_heads, block.head_dim",
    "LAYER_NAME = f'blocks.{BLOCK_INDEX}.attn.qkv_tap'",
    "",
    "if USE_GAMMA:",
    "    composite = AttnLRPGammaComposite(gamma=GAMMA, epsilon=EPSILON)",
    "else:",
    "    composite = AttnLRPEpsilonComposite(epsilon=EPSILON)",
    "",
    "print(f'composite: {type(composite).__name__}')",
    "print(f'layer    : {LAYER_NAME}')",
    "print(f'num_heads: {NUM_HEADS}')",
    "print(f'head_dim : {HEAD_DIM}')",
    "print()",
    "print('concept counts:')",
    "print(f'  HeadConcept    -> {NUM_HEADS}')",
    "print(f'  KQVConcept     -> 3')",
    "print(f'  KQVHeadConcept -> {3 * NUM_HEADS}')",
    "print(f'  HeadDimConcept -> {3 * NUM_HEADS * HEAD_DIM}')"
))


CELLS.append(md(
    "### 3.1 What the canonizer does (inspection cell)",
    "",
    "Sanity-check that `qkv_tap` only exists *inside* `composite.context()`. "
    "Before/after the `with` block, the model is exactly as `timm` constructed it."
))


CELLS.append(code(
    "attn = model.blocks[BLOCK_INDEX].attn",
    "",
    "print('before composite.context():')",
    "print(f'  hasattr(attn, \"qkv_tap\")           = {hasattr(attn, \"qkv_tap\")}')",
    "print(f'  attn.forward is type(attn).forward = {attn.forward.__func__ is type(attn).forward}')",
    "",
    "with composite.context(model) as modified:",
    "    print()",
    "    print('inside composite.context() (canonizer applied):')",
    "    print(f'  hasattr(attn, \"qkv_tap\")           = {hasattr(attn, \"qkv_tap\")}')",
    "    print(f'  attn.forward is timm_attention_forward = '",
    "          f'{attn.forward.__func__.__name__ == \"timm_attention_forward\"}')",
    "",
    "print()",
    "print('after composite.context() exits (canonizer reverted):')",
    "print(f'  hasattr(attn, \"qkv_tap\")           = {hasattr(attn, \"qkv_tap\")}')"
))


# ─── 3.2 Dataset wrapper + preprocess ────────────────────────────────────────


CELLS.append(md(
    "### 3.2 Dataset wrapper + preprocess",
    "",
    "`FeatureVisualization` expects `dataset[i]` to return "
    "`(unpreprocessed_tensor, int_target)`. Mean/std normalisation is applied "
    "by the `preprocess_fn` argument, so unpreprocessed tensors can be plotted "
    "directly as RGB images."
))


CELLS.append(code(
    "cfg = resolve_data_config({}, model=model)",
    "MEAN, STD, IMG_SIZE = cfg['mean'], cfg['std'], cfg['input_size'][1]",
    "",
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
    "def preprocess_fn(x):",
    "    return (x - MEAN_T.to(x)) / STD_T.to(x)",
    "",
    "",
    "def denormalize(x):",
    "    if x.dim() == 3: x = x.unsqueeze(0)",
    "    return x.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()",
    "",
    "",
    "class ImagenetteDataset(Dataset):",
    "    def __init__(self, root, num_samples):",
    "        files, targets = [], []",
    "        for wnid_dir in sorted((root / 'val').iterdir()):",
    "            label = IMAGENETTE_TO_IMAGENET[wnid_dir.name]",
    "            for f in sorted(wnid_dir.glob('*.JPEG')):",
    "                files.append(f); targets.append(label)",
    "        rng = np.random.default_rng(0)",
    "        order = rng.permutation(len(files))[:num_samples]",
    "        self.files = [files[i] for i in order]",
    "        self.targets = [targets[i] for i in order]",
    "",
    "    def __len__(self): return len(self.files)",
    "",
    "    def __getitem__(self, i):",
    "        img = Image.open(self.files[i]).convert('RGB')",
    "        return to_tensor(img), int(self.targets[i])",
    "",
    "",
    "dataset = ImagenetteDataset(EXTRACTED, NUM_SAMPLES)",
    "print(f'dataset size: {len(dataset)}')"
))


# ─── 4. Single-image conditional attribution ─────────────────────────────────


CELLS.append(md(
    "## 4. Single-image conditional attribution",
    "",
    "Pick the target image (`TARGET_INDEX`, or random under `RANDOM_SEED`), run "
    "one `HeadConcept`-conditioned backward pass, and visualise the pixel-space "
    "heatmap. Same pattern as the original `attributions.ipynb` cell — only "
    "the `mask_map` and `composite` are different."
))


CELLS.append(code(
    "rng = np.random.default_rng(RANDOM_SEED)",
    "if TARGET_INDEX is None:",
    "    target_idx = int(rng.integers(0, len(dataset)))",
    "else:",
    "    target_idx = int(TARGET_INDEX)",
    "",
    "target_data, target_class = dataset[target_idx]",
    "target_pre = preprocess_fn(target_data.unsqueeze(0)).to(DEVICE)",
    "target_pre.requires_grad_(True)",
    "",
    "print(f'target sample idx : {target_idx}')",
    "print(f'                    {dataset.files[target_idx].name}')",
    "print(f'true class        : {target_class} ({CLASS_NAMES.get(target_class, \"?\")})')",
    "",
    "with torch.no_grad():",
    "    pred = model(target_pre)[0].softmax(dim=-1)",
    "top5 = pred.topk(5)",
    "print('top-5 model predictions:')",
    "for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):",
    "    name = CLASS_NAMES.get(idx, '')",
    "    mark = ' <- target' if idx == target_class else ''",
    "    print(f'  cls {idx:4d}  p={prob:.3f}  {name}{mark}')"
))


CELLS.append(code(
    "attribution = CondAttribution(model, device=torch.device(DEVICE))",
    "",
    "head_concept = HeadConcept()",
    "head_concept.register_from_model(model)",
    "",
    "# Conditional attribution: HeadConcept head=0 under the target class.",
    "conditions = [{LAYER_NAME: [0], 'y': [target_class]}]",
    "result = attribution(",
    "    target_pre, conditions, composite, mask_map=head_concept.mask,",
    ")",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))",
    "axes[0].imshow(denormalize(target_data))",
    "axes[0].set_title(f'input  •  {CLASS_NAMES.get(target_class, target_class)}')",
    "axes[0].axis('off')",
    "hm = result.heatmap[0].detach().cpu().numpy()",
    "vmax = np.abs(hm).max()",
    "axes[1].imshow(denormalize(target_data), alpha=0.4)",
    "axes[1].imshow(hm, cmap='bwr', alpha=0.7, vmin=-vmax, vmax=vmax)",
    "axes[1].set_title(f'heatmap  •  HeadConcept head=0 @ block {BLOCK_INDEX}')",
    "axes[1].axis('off')",
    "plt.tight_layout(); plt.show()"
))


# ─── 5. FV indexing per granularity ──────────────────────────────────────────


CELLS.append(md(
    "## 5. Build a FeatureVisualization index per concept granularity",
    "",
    "For each of the four concept classes we build a separate FV index — same "
    "tap, different aggregation, different number of concepts. Each index "
    "ranks dataset samples by per-concept relevance under each sample's true "
    "class.",
    "",
    "FV writes its results to `tutorials/vit_crp/FeatureVisualization/<name>/` "
    "and the cell below skips `fv.run()` for any granularity that already has "
    "an index there. Delete that directory to force a rebuild."
))


CELLS.append(code(
    "CONCEPT_DEFS = {",
    "    'head':     HeadConcept,",
    "    'kqv':      KQVConcept,",
    "    'kqv_head': KQVHeadConcept,",
    "    'head_dim': HeadDimConcept,",
    "}",
    "",
    "concepts: dict = {}",
    "fvs: dict = {}",
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
    "    rel_dir = FV_ROOT / name / 'RelMax_sum_normed'",
    "    has_index = rel_dir.is_dir() and any(rel_dir.glob('*.npy'))",
    "    if has_index:",
    "        print(f'[{name}] cached index found at {rel_dir} — skipping fv.run()')",
    "        continue",
    "    print(f'\\n=== running FV index for {name!r} ===')",
    "    fv.run(composite, 0, len(dataset), batch_size=8, checkpoint=10000)",
    "print('\\nall four indices ready.')"
))


# ─── 6. Top-concept identification + reference samples ───────────────────────


CELLS.append(md(
    "## 6. Top-concept identification + reference samples",
    "",
    "For the chosen target image:",
    "",
    "1. Run a backward pass per granularity, recording relevance at "
    "`qkv_tap`, masked by `concept.mask` under the target class.",
    "2. Aggregate via `concept.attribute()` to get one scalar per concept "
    "id.",
    "3. Take the top-K concepts by absolute relevance.",
    "4. From the FV index, fetch the top-N **reference samples** that "
    "maximise each concept's relevance over the dataset.",
    "5. Render with `crp.image.plot_grid`."
))


CELLS.append(code(
    "def per_concept_scores(concept, layer_name, data, target_class):",
    "    conditions = [{'y': [target_class]}]",
    "    result = attribution(",
    "        data, conditions, composite,",
    "        mask_map=concept.mask, record_layer=[layer_name],",
    "    )",
    "    rel = result.relevances[layer_name]",
    "    return concept.attribute(rel, layer_name=layer_name, abs_norm=False)[0]",
    "",
    "",
    "def top_k_flat(scores, k):",
    "    flat = scores.flatten()",
    "    k = min(k, flat.numel())",
    "    return torch.topk(flat.abs(), k=k).indices.tolist()",
    "",
    "",
    "def label_for(name, flat_id):",
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
    "top_ids: dict = {}",
    "for name, concept in concepts.items():",
    "    target_pre.grad = None",
    "    scores = per_concept_scores(concept, LAYER_NAME, target_pre, target_class)",
    "    ids = top_k_flat(scores, TOP_K)",
    "    top_ids[name] = ids",
    "    pretty = ', '.join(label_for(name, i) for i in ids)",
    "    print(f'{name:>9s}: top-{TOP_K}  {pretty}')"
))


CELLS.append(md(
    "### 6.1 Reference samples per concept (one row per granularity)",
    "",
    "`get_max_reference` returns the top-N samples for each requested concept "
    "id. We pass `composite=None, plot_fn=None` to get raw RGB tensors; "
    "conditional heatmaps on the **target** image follow in §7.",
    "",
    "(`get_max_reference`'s built-in heatmap path defaults to "
    "`ChannelConcept.mask` and does not yet accept a `mask_map` override — "
    "for now we render the heatmap separately. Tracked in `FUTURE_STATE.md`.)"
))


CELLS.append(code(
    "REF_RANGE = (0, 4)  # top-1..top-4 reference sample per concept",
    "",
    "fig, axes = plt.subplots(",
    "    len(top_ids) * TOP_K, REF_RANGE[1] - REF_RANGE[0],",
    "    figsize=(2.0 * (REF_RANGE[1] - REF_RANGE[0]),",
    "             1.8 * len(top_ids) * TOP_K),",
    ")",
    "for r, (name, ids) in enumerate(top_ids.items()):",
    "    fv = fvs[name]",
    "    ref_c = fv.get_max_reference(",
    "        ids, LAYER_NAME, mode='relevance', r_range=REF_RANGE,",
    "        composite=None, plot_fn=None,",
    "    )",
    "    for j, cid in enumerate(ids):",
    "        row = r * TOP_K + j",
    "        samples = ref_c[cid]",
    "        for c in range(REF_RANGE[1] - REF_RANGE[0]):",
    "            ax = axes[row, c]",
    "            if c < samples.shape[0]:",
    "                ax.imshow(denormalize(samples[c]))",
    "            ax.axis('off')",
    "            if c == 0:",
    "                ax.set_title(f'{name}: {label_for(name, cid)}',",
    "                             fontsize=9, loc='left', pad=2)",
    "fig.suptitle(",
    "    f'Top-{REF_RANGE[1] - REF_RANGE[0]} reference samples per concept '",
    "    f'(ranked by RelMax over the indexed dataset)',",
    "    fontsize=11,",
    ")",
    "plt.tight_layout(); plt.show()"
))


# ─── 7. Conditional heatmaps on the target image ─────────────────────────────


CELLS.append(md(
    "## 7. Conditional heatmaps on the target image",
    "",
    "For each granularity's most-important concept (rank 0 from §6), compute a "
    "pixel-space attribution conditioned on **just that concept** under the "
    "target class. Side-by-side comparison shows how concept granularity "
    "trades off localisation vs. interpretability:",
    "",
    "- `head` covers the whole head's contribution — broadest support;",
    "- `kqv` covers a whole projection (Q, K, or V) across all heads;",
    "- `kqv_head` is the intersection — narrower;",
    "- `head_dim` is a single feature dimension — sharpest, sometimes noisy."
))


CELLS.append(code(
    "def conditional_heatmap(concept, layer_name, concept_id, data, target_class):",
    "    conditions = [{layer_name: [concept_id], 'y': [target_class]}]",
    "    result = attribution(data, conditions, composite, mask_map=concept.mask)",
    "    hm = result.heatmap[0]",
    "    if hm.dim() == 3:",
    "        hm = hm.sum(dim=0)",
    "    return hm.detach().cpu().numpy()",
    "",
    "",
    "img_np = denormalize(target_data)",
    "fig, axes = plt.subplots(1, 5, figsize=(2.4 * 5, 2.6))",
    "",
    "axes[0].imshow(img_np)",
    "axes[0].set_title(f'input\\n{CLASS_NAMES.get(target_class, target_class)}')",
    "axes[0].axis('off')",
    "",
    "for i, (name, ids) in enumerate(top_ids.items()):",
    "    ax = axes[i + 1]",
    "    cid = ids[0]  # most important",
    "    target_pre.grad = None",
    "    hm = conditional_heatmap(concepts[name], LAYER_NAME, cid, target_pre, target_class)",
    "    vmax = np.abs(hm).max()",
    "    ax.imshow(img_np, alpha=0.4)",
    "    ax.imshow(hm, cmap='bwr', alpha=0.7, vmin=-vmax, vmax=vmax)",
    "    ax.set_title(f'{name}\\n{label_for(name, cid)}')",
    "    ax.axis('off')",
    "",
    "fig.suptitle(",
    "    f'conditional heatmaps  •  layer={LAYER_NAME}  •  '",
    "    f'composite={type(composite).__name__}',",
    "    fontsize=11,",
    ")",
    "plt.tight_layout(); plt.show()"
))


# ─── 8. Notes ─────────────────────────────────────────────────────────────────


CELLS.append(md(
    "## 8. What's next",
    "",
    "* Try a different `BLOCK_INDEX` — early blocks (0–3) tend to encode "
    "low-level features (edges, colour); late blocks (9–11) encode "
    "object-/class-level semantics.",
    "* Re-pick the target image (`TARGET_INDEX = …` or change `RANDOM_SEED`) "
    "— concepts will shift.",
    "* Switch composites (`USE_GAMMA = False` for ε-LRP) and rebuild the FV "
    "indices (delete `tutorials/vit_crp/FeatureVisualization/`) to compare "
    "γ vs. ε qualitatively.",
    "* Use `compute_stats` / `get_stats_reference` (see "
    "[`tutorials/feature_visualization.ipynb`](../feature_visualization.ipynb)) "
    "to find the dataset class for which each concept is most representative.",
    "* Read [`tutorials/vit_crp/metrics.py`](metrics.py) for the deletion / "
    "insertion AUC faithfulness benchmark across the four granularities and "
    "the random-concept baseline.",
    "",
    "**Outstanding work**: see [`FUTURE_STATE.md`](../../FUTURE_STATE.md) — "
    "stability metric, localisation metric, multi-block comparison figure, "
    "broader baselines (gradient-only, Grad-CAM, occlusion)."
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
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "walkthrough.ipynb"
    out.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
