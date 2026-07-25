# E2 — which explainability methods highlight the register tokens? (overlap, clean redo)

2026-07-25 · script `experiments/scripts/registers_e2_overlap.py` · arrays
`data/results/registers/e2_overlap_{vit_base_imagenet,vit_small_funny_birds}.npz`,
`e2_summary.json` · figures `figures/registers/e2_overlap/` · paper copies
`crp-paper/iclr2026/journal-figures/e2_examples_p{1,2}.pdf`, `e2_iou_box.pdf`.

## Experiment card

- **RQ**: when an explainability method produces an input saliency map, how strongly
  does its set of "outlier-hot" patches overlap the register set identified from
  activations — and does LRP overlap more than other standard methods?
- **H1**: the LRP (`cp_lrp_baseline`) saliency map's outlier-patch set has substantially
  higher IoU with the activation-identified register set than Chefer's method,
  attention rollout, and occlusion. **H0**: overlaps comparable.
- **Falsified if** LRP's mean IoU is not clearly above the other methods'
  (overlapping bootstrap CIs).
- **Metric rule (symmetric on both sides)**: per-sample `mu + 4*sd`.
  - Activation register set **A**: token L2-norm outliers, union over the **24
    residual-stream sites** (per block *i*: after-attn-add = forward **pre**-hook on
    `blocks[i].norm2`, i.e. its input right after the attention residual add;
    after-mlp-add = `blocks[i]` forward output). mu/sd over the sample's own 196
    patch tokens, CLS excluded; patch in A iff flagged at ≥1 site.
  - Saliency outlier set **S_m**: same rule on the 196 per-patch saliency values of
    that image (mu/sd over the 196 values; flagged iff value > mu+4sd).
  - Per image: IoU(A,S_m), recall |A∩S|/|A|, precision |A∩S|/|S| (NaN when the
    denominator set is empty; nan-aware means).
- **Models/data**: M2 (PRIMARY) ViT-B/16 timm ImageNet val (`n_per_class=10`); M1
  ViT-S/16 FunnyBirds probe (ckpt `data/runs/finetune_vit_small_funny-birds-train-clean/2026-06-03_000556/best.pt`),
  test split. N=64 correctly-classified images each, round-robin class-diverse
  order (seed 0, step-1c scheme), first 64 correct kept. No empty-A images in
  either model (|A| mean 3.95 ViT-B, 5.69 ViT-S).

## Methods (exact hook points; all true-class conditional where applicable)

| method | computation |
|---|---|
| **LRP** | `cp_lrp_baseline` composite, `CondAttribution`, condition `[{"y":[target]}]`, full-model input heatmap; per-patch = sum of \|R\| over the 16×16 pixel patch. |
| **Chefer** (CVPR 2021, grad-weighted rollout variant) | per block `A_bar = I + mean_heads((dlogit/dA ⊙ A)^+)`, rows renormalized, chained over the 12 blocks; CLS row over patches. A captured on the **stock timm attention** (`fused_attn=False`, forward hook on `blocks[i].attn.attn_drop` output — the post-softmax attention, graph kept, no canonizers); `grad_A` via one `torch.autograd.grad` of the summed true-class logits w.r.t. the 12 captured tensors. |
| **attention rollout** (Abnar & Zuidema) | same chaining with the raw attention: `A_bar = row-norm(I + mean_heads(A))` (= 0.5A+0.5I), no gradients; CLS row. |
| **occlusion** | per patch, the 16×16 pixel patch replaced by the per-image mean color (in [0,1] space, before normalize); saliency = `p_clean − p_occluded` of the true-class softmax prob, clamped at 0. 196 forwards/image, batched 98. |

Sanity: per-method maps for the example images inspected before analysis
(`e2_examples_p1.png`) — LRP sharp and register-concentrated, Chefer mixed
object+register, rollout diffuse, occlusion object-focused. Plausible.

## Decision table

**ViT-B/16 · ImageNet val (N=64, |A| mean 3.95, no empty A)**

| method | IoU mean | IoU median | IoU 95% CI (10k bootstrap) | recall | precision | mean \|S_m\| | empty-S imgs |
|---|---|---|---|---|---|---|---|
| **LRP (cp_lrp_baseline)** | **0.637** | **0.667** | **[0.582, 0.693]** | 0.637 | **1.000** | 2.36 | 0 |
| Chefer attribution | 0.332 | 0.250 | [0.256, 0.412] | 0.355 | 0.672 | 1.84 | 7 |
| attention rollout | 0.334 | 0.250 | [0.266, 0.404] | 0.346 | 0.855 | 1.33 | 10 |
| occlusion (Δp⁺) | 0.007 | 0.000 | [0.000, 0.016] | 0.010 | 0.028 | 1.38 | 23 |

**ViT-S/16 · FunnyBirds test (N=64, |A| mean 5.69, no empty A)**

| method | IoU mean | IoU median | IoU 95% CI | recall | precision | mean \|S_m\| | empty-S imgs |
|---|---|---|---|---|---|---|---|
| LRP (cp_lrp_baseline) | 0.013 | 0.000 | [0.004, 0.023] | 0.016 | 0.058 | 2.27 | 0 |
| Chefer attribution | 0.015 | 0.000 | [0.003, 0.030] | 0.016 | 0.060 | 2.64 | 0 |
| attention rollout | 0.006 | 0.000 | [0.000, 0.016] | 0.006 | 0.054 | 0.97 | 27 |
| occlusion (Δp⁺) | 0.011 | 0.000 | [0.003, 0.022] | 0.013 | 0.051 | 2.02 | 0 |

## Verdict

- **ViT-B/16 ImageNet (primary): H1 SUPPORTED.** LRP mean IoU 0.637
  [0.582, 0.693] vs Chefer 0.332 [0.256, 0.412] and rollout 0.334 [0.266, 0.404]
  — CIs clearly disjoint (gap ≈ 0.17 between CI bounds); occlusion is at chance-like
  0.007. LRP's saliency-outlier set is, with **precision 1.000**, a *subset* of the
  activation register set on every image: every patch the mu+4sd rule flags in the
  LRP map is an activation register, and those hot patches catch 64% of the
  registers. The attention-based methods hit registers roughly half as often;
  occlusion — the only purely behavioral method — essentially never does,
  confirming the registers carry no localized class evidence at the input patch.
- **ViT-S/16 FunnyBirds (secondary): all methods ≈ 0, H0 not rejected there** —
  overlapping CIs, means ≤ 0.015. The norm-outlier tokens of the small finetuned
  probe sit on uniform background patches and NO method's outlier-hot set lands on
  them (example panels: every method highlights the bird). So the "LRP paints the
  registers" phenomenon is **specific to the large pretrained model whose LRP
  hotspots we set out to explain**, not a propagation artifact that appears
  wherever high-norm tokens exist.
- Reading: on ViT-B the LRP map is the most faithful *reporter* of register use
  (its outliers are exactly registers), Chefer/rollout partially so, occlusion
  blind to it — consistent with registers being global-information carriers rather
  than local evidence (E-series synthesis, step-3 occlusion).

## Deviations from card

- Selection: first 64 correctly-classified images in the class-diverse order
  (card: "N=64 correctly-classified"); no non-empty-A filter was needed (all 64
  images have |A| ≥ 2 / ≥ 4).
- Occlusion sweeps chunked 32 images per GPU-lock hold (≈2 min each), 2 chunks
  per model; all other stages single holds ≤5 min.
- Figures: FunnyBirds example panels additionally written
  (`e2_examples_vit_small_funny_birds_p{1,2}`); not copied to the paper (card
  lists only the three exact names).
