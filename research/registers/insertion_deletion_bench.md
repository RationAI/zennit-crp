# Insertion-Deletion saliency benchmark (DAPC) — Benchmark run 1

Register / reference note for `exp:insertion-deletion-bench` (crp-paper
experiment journal, section "Insertion-Deletion benchmark for saliency masks").

- **Code:** `experiments/insertion_deletion_bench.py` (module + Typer CLI),
  saliency primitives in `experiments/xai_methods.py`, driver
  `experiments/scripts/run_iddapc_bench.sh`.
- **Arrays:** `data/results/benchmark/iddapc_<model>.npz` (raw curves per
  image/method + per-image DAPC + provenance).
- **Summary:** `data/results/benchmark/iddapc_summary.csv`.
- **Figures:** `figures/benchmark/iddapc_curves_<M>.{png,pdf}`,
  `figures/benchmark/iddapc_bars.{png,pdf}`; PDFs copied to
  `crp-paper/iclr2026/journal-figures/`.

## 1. DAPC metric — exactly as implemented

Per `(method ψ, model M, image x)`:

1. Saliency map `s_{ψ,M}(x)` → per-patch score by **MAX over each
   non-overlapping `patch×patch` patch** (aligned to the model's patch
   embedding: 16 px; grid 14×14 @224 for M1/M2, 16×16 @256 for the DINOv3
   M3/M4). Values kept **signed** (no abs) so an all-negative-relevance patch
   sorts last.
2. Sort patches descending by score: `S_0 ≥ S_1 ≥ … ≥ S_{N-1}`.
3. `M(·)` = **predicted-class softmax probability** (each image conditioned on
   the model's own predicted class).
4. Occlude by **image-mean fill** (patch pixels ← per-image mean colour).
5. Two cumulative curves, each of length `N+1` (index `k` = number of patches
   occluded, `k = 0..N`), **normalised by dividing by `M(x)`** so index 0 = 1.0:
   - **MoRF** — most-salient patches occluded first.
   - **LeRF** — least-salient patches occluded first.
   Index `N` (all patches occluded) is identical for MoRF and LeRF.

```
DAPC(ψ, M, x) = area_under(LeRF curve) − area_under(MoRF curve)
```

with `area = np.trapezoid(curve, dx = 1/N)` over occlusion-fraction ∈ [0,1].
**Positive = good; HIGHER = BETTER.** A faithful map makes MoRF drop fastest
(smallest area) and LeRF slowest (largest area), so `LeRF − MoRF > 0`.

**Sign/normalisation equivalence to the journal equations.** The journal defines
`MoRF/LeRF` as the mean *drop* `M(x) − M(τ)` and `DAPC = LeRF − MoRF`. With the
normalised prediction `p̂ = M(τ)/M(x)`, drop`= 1 − p̂`, so
`area(drop_MoRF) = 1 − area(p̂_MoRF)` and the journal's
`LeRF_drop − MoRF_drop = area(p̂_LeRF) − area(p̂_MoRF)` — **identical** to the
prediction-curve DAPC above. We store the raw normalised prediction curves
(`curve_morf__<m>`, `curve_lerf__<m>`, shape `(n, N+1)`) so any AOPC/AUC/sign
variant is recomputable offline **without rerunning**; per-image DAPC
(`dapc__<m>`) and the image mean are stored alongside.

## 2. Chefer verification against the original repo

**Source of truth:** Chefer, Gur, Wolf, *"Transformer Interpretability Beyond
Attention Visualization"*, CVPR 2021 (`chefer2021transformer`), repo
<https://github.com/hila-chefer/Transformer-Explainability>,
`baselines/ViT/ViT_LRP.py :: VisionTransformer.relprop(method=
"transformer_attribution")`:

```python
for blk in self.blocks:
    grad = blk.attn.get_attn_gradients()   # ∂ y_t / ∂ A  (clean autograd)
    cam  = blk.attn.get_attn_cam()         # LRP relevance R_A of the attn map
    cam  = (grad * cam).clamp(min=0).mean(dim=0)   # over heads
    cams.append(cam)
rollout = compute_rollout_attention(cams)  # ∏ row-norm(I + cam_l), bmm-chained
cam = rollout[:, 0, 1:]                     # CLS→patch row
```

**Did ours match? NO — our previous `chefer_relevance` deviated.** It weighted
the attention *gradient* by the **raw post-softmax attention** `A`
(`cam = (grad ⊙ A)⁺` head-mean), i.e. the **ICCV'21** *"Generic Attention-model
Explainability"* self-attention rule — **not** the CVPR'21 method, which weights
the gradient by the **LRP relevance** `R_A` of the attention map
(`get_attn_cam`). The rollout structure (`I + cam`, row-normalise, matmul-chain,
`[:,0,1:]` read) was already exact.

**Correction (`xai_methods.chefer_transformer_attribution`).** We reproduce the
CVPR'21 algorithm faithfully **without vendoring the authors' bespoke
LRP-instrumented ViT** (per spec). The two ingredients are supplied from our
own stack, in the identical timm `(1, heads, T, T)` head layout:
- `grad` — the clean autograd gradient of the predicted-class logit w.r.t. each
  block's post-softmax attention (`capture_attention(keep_graph=True)` + one
  `autograd.grad`); identical to `get_attn_gradients`.
- `R_A` — the attention-map **LRP relevance**, recorded at each block's
  `attn.softmax` under our **AttnLRP full-bilinear composite (`attnlrp_gamma`)**.
  This is the faithful analogue of `get_attn_cam`: LRP relevance of the softmax,
  computed with our LRP framework instead of theirs. (The benchmark's *LRP*
  saliency row still uses `cp_lrp_baseline`; only Chefer's `R_A` uses
  `attnlrp_gamma`, because CP-LRP StopGradients Q/K → the softmax is a graph
  constant with `R_A ≡ 0`.)

**Quantified difference** (`verify-chefer`, M1, 8 images): the CVPR'21 map
(`grad ⊙ R_A`) and the ICCV'21 map (`grad ⊙ A`) correlate patch-wise
mean 0.92 (min 0.77, max 0.99) — closely related but genuinely distinct methods.
Benchmark run 1's `chefer` row is the **CVPR'21** `chefer_transformer_attribution`.

## 3. RISE vs the E2 single-patch occlusion — answer

**They are NOT the same method.**

- **E2 occlusion** (`xai_methods.occlusion_deltap`): mask **one** 16×16 patch at a
  time with the image-mean colour and record the drop in the predicted-class
  probability, `Δp⁺ = (M(x) − M(x∖p))⁺`. `N` = `grid²` forwards; a *marginal,
  leave-one-out* single-patch effect.
- **RISE** (`xai_methods.rise_saliency`, Petsiuk et al. BMVC'18,
  <https://github.com/eclique/RISE>): draw `N=2000` random `8×8` Bernoulli(0.5)
  masks, bilinearly upsample to `(s+1)·cell` with `cell=⌈H/s⌉` and random-crop
  back to `H×H`; saliency `= (1/(N·p)) Σ_i M(x ⊙ mask_i)·mask_i`. Zero-fill
  masking of **many patches at once**; each pixel's score is its *expected model
  response over random multi-patch subsets* (a correlation / Shapley-flavoured
  estimate), not a single-patch marginal.

So RISE answers a different question than the E2 occlusion (expected joint
contribution vs. individual marginal drop) and uses different masking (random
multi-patch zero-fill vs. one-patch mean-fill). Benchmark run 1 uses **real
RISE** for its `rise` row.

## 4. Results — Benchmark run 1

DAPC mean ± std over `N=64` correctly-classified images (seed 0). **Higher =
better**; **bold** = best method per model. Random ≈ 0 on every model confirms
the protocol is unbiased (floor).

| Method            | M1 ViT-S/FB      | M2 ViT-B/IN      | M3 DINOv3-S/FB   | M4 DINOv3-B/IN   |
|-------------------|------------------|------------------|------------------|------------------|
| LRP (cp_lrp)      | **0.867 ±0.065** | 0.310 ±0.229     | **0.860 ±0.077** | 0.442 ±0.302     |
| Chefer CVPR'21    | 0.828 ±0.087     | **0.467 ±0.216** | 0.855 ±0.064     | **0.585 ±0.264** |
| Attn rollout      | 0.764 ±0.123     | 0.060 ±0.258     | 0.767 ±0.080     | 0.319 ±0.317     |
| RISE              | 0.681 ±0.226     | 0.367 ±0.297     | 0.737 ±0.203     | 0.569 ±0.379     |
| Random (floor)    | −0.000 ±0.116    | −0.002 ±0.056    | 0.007 ±0.101     | −0.004 ±0.060    |

**Ranking per model** (best → worst):
- **M1** ViT-S/FunnyBirds: LRP > Chefer > rollout > RISE > random
- **M2** ViT-B/ImageNet: **Chefer** > RISE > LRP > rollout > random
- **M3** DINOv3-S/FunnyBirds: LRP ≳ Chefer (tied) > rollout > RISE > random
- **M4** DINOv3-B/ImageNet: **Chefer** > RISE > LRP > rollout > random

**Is LRP competitive?** Split by dataset: **on FunnyBirds (M1, M3) LRP is the
single best method**; **on ImageNet (M2, M4) LRP is only 3rd**, clearly behind
Chefer and RISE (LRP 0.31/0.44 vs Chefer 0.47/0.59). This matches the E2
register finding — on the ImageNet ViT-B / DINOv3-B models LRP places relevance
mass on high-norm register/outlier patches, which are not the patches the
classifier most relies on, depressing MoRF faithfulness. Attention rollout is
weakest of the non-random methods on ImageNet (0.06 on M2). Chefer (CVPR'21) is
the most consistent method overall (top-2 on all four models).

Figures: `figures/benchmark/iddapc_curves_{M1..M4}.{png,pdf}` (mean MoRF solid /
LeRF dashed per method), `figures/benchmark/iddapc_bars.{png,pdf}`
(DAPC per method × model).

## 5. Deviations / notes

- **Curve length `N+1`, not `N`.** We store the clean anchor (index 0 = 1.0)
  plus the `N` occlusion steps, so the full curve including endpoints is
  reproducible; the journal text says "length N" (the N occlusion steps). DAPC
  is invariant to this choice (trapezoid over the same [0,1] support).
- **Chefer `R_A` composite.** CVPR'21's `get_attn_cam` is supplied by our
  `attnlrp_gamma` (see §2), not the authors' LRP ViT.
- **M3 attention relevance has tiny magnitude but a faithful ordering.** On the
  finetuned DINOv3-S (M3) the recorded `R_A` at the softmax is orders of
  magnitude smaller than on M1/M4 (registers absorb most relevance), yet the
  *ranking* of patches it induces is still highly faithful — Chefer scores 0.855
  on M3, statistically tied with LRP. So the small magnitude is not a failure;
  DAPC only uses the patch order.
- **ImageNet pool.** M2/M4 select from the HF-val `n_per_class=10` subset
  (10k images), consistent with the rest of the repo; 64 correct images, seed 0.
- **Predicted-class conditioning.** Every method and the perturbation target use
  the model's own predicted class (not the ground-truth label).
