# Frozen-backbone DINOv3 FunnyBirds classifier — priority ladder (Adam reprioritization)

_2026-07-26. Goal: FROZEN-backbone DINOv3 FunnyBirds classifier, val top-1 >= 0.95
(reference: M1 end-to-end ViT-S 0.9738). Stop at the first rung reaching the bar._

All runs: `funny-birds-train-clean` (29 330 imgs, 50 classes), features cached from the
frozen timm backbone at 256x256 (bicubic, ImageNet norm), val split = 10% random
(`random_split`, seed 0), python `/home/claude/venvs/zennit-crp/bin/python`, single A40.
`cls_token` = true post-norm CLS `forward_features(x)[:, 0]` — NOT timm's
`forward_head` (global_pool='avg' on DINOv3/Eva = patch mean-pool; the earlier
0.7279 "cls" probe was actually mean-pool). Pipeline extensions added for this:
`train_probe cache --kind cls_token` and `train_probe train --feature-kind cls_token`.

## Rung table

| rung | model (frozen) | head | config | val top-1 | val top-5 |
|---|---|---|---|---|---|
| 0 (pre-existing) | DINOv3-S/16 | linear on mean-pool | lr 1e-3 const | 0.7279 | 0.9718 |
| 1 | DINOv3-S/16 | linear on CLS token | cosine 100ep w5, lr 1e-3, wd 1e-2 | 0.8153 | 0.9896 |
| 1' | DINOv3-S/16 | linear on CLS token | cosine 100ep w5, lr 5e-3, wd 1e-4 | 0.8348 | 0.9944 |
| 2 | DINOv3-S/16 | attentive (8 heads) | cosine 60ep w5, lr 1e-3, wd 1e-2 | 0.9222 | 0.9987 |
| 2' | DINOv3-S/16 | attentive (8 heads) | cosine 120ep w10, lr 2e-3, wd 1e-4 | 0.9278 | 0.9994 |
| **3** | **DINOv3-B/16** | **attentive (8 heads)** | **cosine 60ep w5, lr 1e-3, wd 1e-2** | **0.9530** | **0.9980** |

**Bar reached at rung 3** (0.9530 >= 0.95). Rung 4 (512^2) not run — ladder stops at
the first passing rung.

## Deliverable

- **Checkpoint**: `data/vit_dinov3_base_attentive_probe_funny-birds-train-clean.pt`
  (base `vit_dinov3_base` = `vit_base_patch16_dinov3.lvd1689m`, head `attentive`
  num_heads=8, tokens cache fp16 `data/vit_dinov3_base_funny-birds-train-clean_tokens_feats.pt`).
- Exact command:
  `python -m experiments.train_probe train vit_dinov3_base attentive funny-birds-train-clean
  --epochs 60 --patience 10 --scheduler cosine --cosine-warmup-epochs 5 --lr 1e-3` (seed 0).
- Caches (all seed-free, deterministic eval transform):
  `data/vit_dinov3_{small,base}_funny-birds-train-clean_{cls_token,tokens}_feats.pt`.
- Other rung checkpoints:
  `data/vit_dinov3_small_linear_probe_funny-birds-train-clean_cls_token.pt` (rung 1),
  `..._cls_token_lr5e-3.pt` (rung 1'),
  `data/vit_dinov3_small_attentive_probe_funny-birds-train-clean.pt` (rung 2),
  `..._lr2e-3.pt` (rung 2').

## Side product (kept, deprioritized)

End-to-end finetune Arm A (M1 protocol on DINOv3-S: 25ep, backbone_lr 5e-4, head_lr
5e-3, LLRD 0.7, wd 0.05, bs 64x2, bf16, onecycle 0.1, RandAugment, ls 0.1, seed 0):
**val top-1 0.9702, top-5 1.0000**, run dir
`data/runs/finetune_vit_dinov3_small_funny-birds-train-clean/2026-07-25_200008/`
(best.pt + config.json + metrics.csv). E3 analysis NOT run (deprioritized); the E3
tooling is in place: `experiments/scripts/registers_e3_finetune.py` +
`--checkpoint/--indices-from/--out` extensions to `registers_e1_counts.py collect`.

## Diagnosis notes (why linear-CLS is far from the bar)

FunnyBirds classes share most parts; the global CLS summary of a frozen SSL backbone
under-separates them (0.83 ceiling despite top-5 0.994) — the gap is feature-pooling,
not optimization: moving from CLS to attentive pooling over patch tokens (+0.09) and
to the larger backbone (+0.03) closes it. Domain shift (synthetic renders vs LVD-1689M)
caps the small model below the bar even with attentive pooling.
