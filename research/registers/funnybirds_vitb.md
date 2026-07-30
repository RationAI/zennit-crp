# FunnyBirds ViT-B (authors' pretrained) + ViT-S full-split retrain

Two deliverables on the same 500-image FunnyBirds **test** split (10 imgs/class,
zero part-ablations). Apples-to-apples baseline: current **M1** ViT-S/16
(`data/runs/finetune_vit_small_funny-birds-train-clean/2026-07-26_160337/best.pt`,
the run referenced at `\label{model:vits-fb}`) measured here at **test top-1
0.980 / top-5 1.000** (490/500). The older M1-recipe run 2026-06-03_000556 is
identical on test (0.980 / 1.000).

FunnyBirds classification splits (verified from `data/funny_birds/FunnyBirds/`):
train 50,000 = 29,330 clean + 20,670 part-ablation (41.3%); test 500, zero
ablations. Reference: Hesse, Schaub-Meyer, Roth, *FunnyBirds: A Synthetic Vision
Dataset for a Part-Based Analysis of Explainable AI Methods*, ICCV 2023
(arXiv:2308.06248); framework repo `visinf/funnybirds-framework`.

---

## Deliverable A — authors' pretrained ViT-B (model record for `model:vits-fb-paper`)

### Journal-ready model record (M1/M2 6-row format)

| field | value |
|---|---|
| architecture | timm `vit_base_patch16_224`: 12 blocks, embed dim 768, 12 heads, patch 16, input $224^2$; classification head $768{\to}50$. State-dict layout is **byte-for-byte identical** to timm's `vit_base_patch16_224` (num_classes=50) — verified zero missing/extra keys — so the authors' vendored Chefer ViT (`models/ViT/ViT_new.py`) loads straight into a stock timm ViT-B/16. Their `ViTModel` wrapper bilinearly downscales the input $256^2{\to}224^2$ before the ViT; reproduced in `FunnyBirdsViTB.forward`. |
| pretrained weights | `vit_base_patch16_224_final_1_checkpoint_best.pth.tar` published by the FunnyBirds authors on the TU Darmstadt visinf mirror (`https://download.visinf.tu-darmstadt.de/data/funnybirds/models/vit_base_patch16_224_final_1_checkpoint_best.pth.tar`, 654 MB). Init was ImageNet ViT-B/16 (rwightman "jx" `jx_vit_base_p16_224-80ecf9dd.pth`), then fine-tuned end-to-end with the head swapped to 50 outputs. Checkpoint metadata: `model='vit_base_patch16_224'`, `epoch=56`, `best_acc1=98.0`; weights under the `['state_dict']` key, raw ViT keys (no prefix). Local copy: `data/funnybirds_models/vit_base_patch16_224_final_1_checkpoint_best.pth.tar`. |
| training data & target | FunnyBirds train split via the authors' `train.py` (`FunnyBirds(data,'train',...)` reading `dataset_train.json`), 50 classes, cross-entropy. No data augmentation and **no ImageNet mean/std normalization** — dataset yields RGB (alpha dropped) resized to $256^2$, `ToTensor` → $[0,1]$. |
| procedure | SGD, lr 0.1, momentum 0.9, weight decay $1{\cdot}10^{-4}$; StepLR (step 60, gamma 0.1); 120 max epochs, batch 64, seed 0; `--pretrained` ImageNet init; checkpoint = best test acc (this file is epoch 56). Source: `visinf/funnybirds-framework/train.py`. Not trained locally — loaded as-is. |
| validation & test | Evaluated locally on the FunnyBirds **test** split (N=500) with the faithful pipeline (resize $256^2$, $[0,1]$, no normalize, interpolate to $224^2$). |
| metrics | test top-1 **0.9800**, top-5 **1.0000** (490/500) — exactly matches the checkpoint's stored `best_acc1=98.0`, confirming a faithful load + preprocessing. For reference, current M1 ViT-S is also 0.980/1.000 on this split. |

### Citation

> Robin Hesse, Simone Schaub-Meyer, and Stefan Roth. "FunnyBirds: A Synthetic
> Vision Dataset for a Part-Based Analysis of Explainable AI Methods." In
> *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*,
> 2023. arXiv:2308.06248. Framework + pretrained models:
> https://github.com/visinf/funnybirds-framework

### Loader (wired into the repo)

`experiments/models/funnybirds_vitb.py`:
- `load_funnybirds_vitb(ckpt_path=DEFAULT_CKPT, device)` → `FunnyBirdsViTB`
  (frozen eval; `.backbone` = full timm ViT-B/16 with head, so
  `backbone.blocks.{i}` attribution paths + the AttnLRP composite work exactly
  like the `imagenet` full-timm model in `experiments/model_io.py`).
- `funnybirds_vitb_transforms()` → `(transform, normalize)`: resize $256^2$ +
  `ToTensor`; normalize is identity (authors use no mean/std norm).
- `FunnyBirdsViTB.forward` interpolates to $224^2$ (reproduces their `ViTModel`).

Reproduce sanity inference:
```python
from experiments.models.funnybirds_vitb import load_funnybirds_vitb, funnybirds_vitb_transforms
from experiments.datasets.funny_birds import FunnyBirdsDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torch
m = load_funnybirds_vitb(device='cuda')
tfm, norm = funnybirds_vitb_transforms()
ds = FunnyBirdsDataset(root='data', split='test', transform=tfm, auto_download=False)
a1 = MulticlassAccuracy(num_classes=50).cuda(); a5 = MulticlassAccuracy(num_classes=50, top_k=5).cuda()
with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=50):
        x, y = x.cuda(), y.cuda(); lo = m(norm(x)); a1(lo, y); a5(lo, y)
print(a1.compute().item(), a5.compute().item())   # 0.98 1.0
```

**Dependency note:** the authors' repo uses a vendored Chefer ViT and a custom
`ViTModel`/`StandardModel` wrapper (their `models/`), plus their intervention
dataset methods — none of which we need to *load* the classifier, because the
weights map 1:1 onto timm. We did not vendor their code; the in-repo timm
equivalent is faithful (metrics match their stored `best_acc1`).

---

## Deliverable B — ViT-S retrained on the FULL train split (clean_only=False)

Hypothesis: the 20,670 part-ablation samples act as training augmentation that
lifts test accuracy above M1 (which drops them, `clean_only=True`).

Recipe: **identical to M1** (AdamW wd 0.05; backbone lr $5{\cdot}10^{-4}$, head lr
$5{\cdot}10^{-3}$, layer-wise lr decay 0.7; OneCycle pct_start 0.1, div 25, final
div $10^4$; 25 epochs; batch 64 × 2 grad-accum; bf16-mixed; RandomResizedCrop
0.7–1.0 + rotation ±15° + RandAugment(2, 9); label smoothing 0.1; seed 0), the
**only** change being `--train-ds funny-birds-train-full` (clean_only=False →
50,000 train imgs, 45,000 train / 5,000 val after the seeded 10% split). This is
the closest our `train_probe.py finetune --from-scratch` allows to the authors'
recipe; their SGD-lr0.1/120-epoch/no-aug recipe is not portable to our
OneCycle+AugReg harness, so we hold M1's recipe fixed to isolate the data change.

Launch command:
```
python -m experiments.train_probe finetune --from-scratch \
    --base vit_small --head linear --train-ds funny-birds-train-full \
    --epochs 25 --patience 25 --scheduler onecycle --onecycle-pct-start 0.1 \
    --backbone-lr 5e-4 --head-lr 5e-3 --layerwise-lr-decay 0.7 \
    --weight-decay 0.05 --randaugment --label-smoothing 0.1 \
    --batch-size 64 --accumulate-grad-batches 2 \
    --num-workers 4 --precision bf16-mixed --seed 0
```

- Run dir: `data/runs/finetune_vit_small_funny-birds-train-full/2026-07-30_224736/`
  (best.pt + config.json + metrics.csv). Trained 2026-07-30, 25 epochs, seed 0.
- **val top-1 0.7461**, top-5 0.8720, val_loss 1.469 (val = seeded 10% of the FULL
  split = 5,000 imgs including ablations). This is *not* comparable to M1's
  clean-only val 0.9715: the full-split val is ~41% ablated birds, many genuinely
  ambiguous between classes, so val_acc is ceilinged well below 1.0. Train top-1
  reached 0.874 (also on the mixed split) — so the low val is the ablation ceiling,
  not underfitting.
- **test top-1 0.9780**, top-5 **1.0000** (489/500) on the clean 500-img test split
  — the honest comparison metric.
- Baseline M1 test top-1: **0.980** (490/500).

**Result: hypothesis NOT supported.** Adding the 20,670 part-ablation samples did
**not** lift clean-test accuracy — the new ViT-S scored 0.978 vs M1's 0.980, a
one-image difference (489 vs 490 / 500), i.e. statistically a tie and slightly
*worse*, not better. The ablated samples do not act as beneficial augmentation for
clean-test classification under this (M1) recipe.

**Recommendation: KEEP M1.** The new ViT-S neither reaches ≥0.99 test top-1 nor
clearly beats M1's 0.980. The current M1 checkpoint
(`.../finetune_vit_small_funny-birds-train-clean/2026-07-26_160337/best.pt`) is
untouched; the new run is recorded above but not adopted.
