# DINOv3 classifier-equipped ViTs for the register-token study

Date: 2026-07-25. Goal: DINOv3 backbones (with 4 register tokens) + trained
classifier heads on ImageNet-1k and FunnyBirds, backbone frozen so its
behavior is preserved for the registers question.

## 1. Checkpoint dig — what exists

Searched: this repo + `data/` + all of `/home/claude/workspaces` (no dinov3
classifier checkpoints anywhere locally; only the timm backbone cache for
`vit_small_patch16_dinov3.lvd1689m`); HuggingFace; timm registry.

| Source | Finding |
|---|---|
| timm `*dinov3*` tags | All pretrained cfgs have `num_classes=0` — backbone-only (`.lvd1689m` / `.sat493m` tags). No IN1k classifier ships with any dinov3 timm tag. |
| Meta `facebook/dinov3-*` + [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | Backbones only on HF. The GitHub repo releases exactly ONE classifier: `dinov3_vit7b16_lc` (ImageNet-1k linear head for ViT-7B/16, torch.hub). Nothing for S/B/L. |
| RationAI org (HF) | Only `LSP-DETR` and `vit-patch16-224-prostate` — no dinov3. |
| **canvit / m2b3 probes** (HF `canvit/dinov3-*-in1k-512x512-linear-clf-probe`, code [m2b3/dinov3-in1k-probes](https://github.com/m2b3/dinov3-in1k-probes), MIT) | **Usable.** Linear IN1k heads on the frozen **CLS token** for ViT-S/16 (81.40), S+/16 (82.89), **B/16 (85.00)**, L/16 (87.44), H+/16 (87.65) — top-1 on full IN1k val at 512×512 input. Plain `nn.Linear` safetensors (`weight`/`bias`); vitb16 config records val top-1 0.8500, top-5 0.9725, ReAL 0.8954; AdamW, Optuna-tuned on cached features, "use_dinov3_init". |

Decision: use the canvit ViT-B/16 head for the ImageNet model (step 4 "found"
branch); train our own FunnyBirds linear probe on DINOv3-S/16 (step 3).

## 2. Base-registry additions

New frozen bases (mirroring `experiments/models/bases/vit_dinov3.py`):

- `experiments/models/bases/vit_dinov3_small.py` — `DinoV3Small`, timm
  `vit_small_patch16_dinov3.lvd1689m` (tag pinned).
- `experiments/models/bases/vit_dinov3_base.py` — `DinoV3Base`, timm
  `vit_base_patch16_dinov3.lvd1689m`.
- Registered in `bases/__init__.py` and `models/__init__.py` `BASES` as
  `vit_dinov3_small` / `vit_dinov3_base` (existing `vit_dinov3` = large,
  unchanged).

Verified via `build_base`: S 21.6M params, embed 384; B 85.6M, embed 768;
both 12 blocks, `num_prefix_tokens = 5` (1 cls + 4 register), 261 tokens at
the default 256² input (RoPE — resolution-flexible), ImageNet mean/std.

### PITFALL — timm dinov3 is `Eva` with `global_pool='avg'`

`forward_head(pre_logits=True)` (our `Base.extract_cls`) returns the **mean
of patch tokens**, NOT the cls token, for all dinov3 bases (S/B/L). The final
`norm` LayerNorm is applied inside `forward_features` (`fc_norm` is
Identity), so Meta's `x_norm_clstoken` == `forward_features(x)[:, 0]`.
Consequences:

- The canvit heads consume the CLS token — feeding them
  `extract_cls` output costs ~30 points top-1 (measured 0.554 vs 0.825 at
  256). The eval script uses `forward_features(x)[:, 0]`.
- Our FunnyBirds probe (below) was trained via the unmodified pipeline, i.e.
  on **mean-patch-token** features. Self-consistent (train and eval use the
  same extractor), but record the feature type.

## 3. FunnyBirds probe — DINOv3-S/16 (trained 2026-07-25)

Pipeline: `python -m experiments.train_probe cache vit_dinov3_small
funny-birds-train-clean --kind cls --batch-size 64` (29,330 imgs, 148 s on
the A40; cache
`data/vit_dinov3_small_funny-birds-train-clean_cls_feats.pt`), then
`train vit_dinov3_small linear funny-birds-train-clean`.

- Attempt 1 (defaults: lr 1e-3, wd 0.01, 50 epochs, no scheduler): val
  top-1 **0.6284**, top-5 0.9382 — ran all 50 epochs, train acc only 0.650
  → underfit.
- Attempt 2 (the one permitted retry: `--lr 0.01 --scheduler cosine
  --epochs 100 --patience 15`): val top-1 **0.7279**, top-5 **0.9718**,
  val_loss 0.886 (train acc 0.805 — still optimization/feature limited).

Checkpoint: `data/vit_dinov3_small_linear_probe_funny-birds-train-clean.pt`
(self-describing: base/head/num_classes/dataset kwargs + head_state_dict +
final metrics). Split: seeded 90/10 of train-clean (26,397/2,933).

Short of the >0.9 hope (M1 end-to-end finetune reaches 0.9758). Plausible
causes, NOT pursued per scope: (a) frozen SSL features on synthetic
renderings are out-of-domain; (b) mean-patch-token features dilute the small
discriminative parts (background dominates); a cls-token or attentive-head
probe, or 512² inputs, are the obvious next knobs if a stronger FB head is
needed.

## 4. ImageNet head — DINOv3-B/16 + canvit probe (wired 2026-07-25)

Loader + eval: `experiments/scripts/eval_dinov3_in1k_probe.py`
(`load_in1k_linear_head(timm_name)` downloads
`canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe`
`model.safetensors` → frozen `nn.Linear(768, 1000)`; logits =
`head(forward_features(x)[:, 0])`). S/16 and L/16 head repos are also mapped
in `HEAD_REPOS`.

Verification on 5,000 imgs (5/class, seeded, `imagenet_val_hf` — NB the
un-gated mirror is pre-resized to 256px, so 512² eval upsamples):

| eval res | top-1 | top-5 |
|---|---|---|
| 512² (head-native) | **0.8336** | 0.9622 |
| 256² (backbone default) | **0.8248** | 0.9592 |

Consistent with the claimed 0.8500 full-val at native resolution given the
256px source cap; head transfers to 256² with only ~1 point loss (RoPE).
No local ImageNet head training needed — the "not found" plan/cost branch is
moot. (If a 256-native or S/16-FB-style head is ever wanted: cls-feature
cache of IN1k train at 256² ≈ 1.28M × 768 × 4B ≈ 3.9 GB, extraction ~2 GPU-h
on the A40 at ~190 img/s, head training minutes.)

## 5. Journal-ready model records (M1/M2 6-row format; NOT yet in the .tex)

### Mx — DINOv3-S/16 FunnyBirds linear probe (trained 2026-07-25)

| field | value |
|---|---|
| architecture | timm `vit_small_patch16_dinov3` (Eva impl): 12 blocks, embed dim 384, 6 heads, patch 16, RoPE, 5 prefix tokens (1 cls + **4 register**), input 256²; linear classification head 384→50 (fresh init) on **mean-patch-token** features (timm Eva `global_pool='avg'`; the cls/register tokens are not read by the head). |
| pretrained weights | timm tag `vit_small_patch16_dinov3.lvd1689m` — DINOv3 self-supervised distillation (Siméoni et al. 2025), LVD-1689M pretrain. Backbone loaded `pretrained=True`, frozen (`experiments/models/bases/vit_dinov3_small.py`); only the head is trained. |
| training data & target | FunnyBirds train split, `clean_only` (29,330 of 50,000 images), 50 classes; features pre-extracted once from the frozen backbone (fp32 cls-feature cache); cross-entropy on class labels. |
| procedure | AdamW lr 0.01, weight decay 0.01, batch 256, cosine schedule with 5 warmup epochs (DINOv3 probe protocol flag), max 100 epochs, EarlyStopping patience 15 on val_acc, seed 0; checkpoint = best val accuracy. First attempt at default lr 1e-3 underfit (0.628); this is the single permitted retry. |
| validation & test | validation = seeded random 10% of train-clean (2,933 imgs), monitored per epoch; FunnyBirds test split untouched. |
| metrics | best val top-1 **0.7279**, top-5 0.9718, val_loss 0.886 (train top-1 0.805 — probe is feature/optimization limited; cf. M1 end-to-end 0.9758). |

Reproduce: cache + train commands above; checkpoint
`data/vit_dinov3_small_linear_probe_funny-birds-train-clean.pt`.

### My — DINOv3-B/16 ImageNet (public canvit linear head, not trained here)

| field | value |
|---|---|
| architecture | timm `vit_base_patch16_dinov3` (Eva impl): 12 blocks, embed dim 768, 12 heads, patch 16, RoPE, 5 prefix tokens (1 cls + **4 register**), default input 256²; linear head 768→1000 on the final-norm **CLS token** (`forward_features(x)[:, 0]`, = Meta's `x_norm_clstoken`; NOT timm's avg-pool pre_logits). |
| pretrained weights | backbone: timm `vit_base_patch16_dinov3.lvd1689m` (DINOv3 LVD-1689M). Head: HF `canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe` (m2b3/dinov3-in1k-probes, MIT; sha 67d9997), plain safetensors `weight`/`bias`. Loader: `experiments/scripts/eval_dinov3_in1k_probe.py::load_in1k_linear_head`. |
| training data & target | head trained upstream (canvit) on ImageNet-1k train, frozen-backbone CLS features at 512², cross-entropy; listed for provenance — no local training of any part. |
| procedure | n/a locally. Upstream (head config.json): AdamW, peak lr 2.8e-4, wd 0.092, batch 1024, 20 epochs, warmup 10%, Optuna-selected. |
| validation & test | upstream full IN1k-val: top-1 0.8500, top-5 0.9724, ReAL 0.8954 (at 512²). Locally verified on a seeded 5/class 5,000-img `imagenet_val_hf` subset (256px-resized mirror): top-1 0.8336 / top-5 0.9622 at 512² (upsampled), 0.8248 / 0.9592 at 256². |
| metrics | see validation row; per-experiment subset accuracies to be reported in the respective entries. |
