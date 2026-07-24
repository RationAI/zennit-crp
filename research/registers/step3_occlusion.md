# Step 3 (XAI-37): H_B occlusion test — does input content at outlier-token positions matter?

Part of the register/outlier-token study (XAI-34; Darcet et al., arXiv:2309.16588).
**Question:** ViT relevance maps put substantial input relevance on the patches that
become high-norm "outlier"/register tokens. Is that faithful — is the model's
prediction genuinely sensitive to the input content there (H_B)? Or are those
patches selected precisely because their content is redundant (H_A, the registers
paper's account), making the input relevance an artifact?

**Verdict: H_B is rejected; the data support H_A.** LRP assigns a median 23.7% of
total image |R| to outlier patches (~2% of tokens; the single primary outlier patch
is the top-ranked patch of 196 in the median image, carrying 11.1% of |R|), yet
occluding it changes the target probability by a median of only −0.0005 —
statistically indistinguishable from occluding a random low-relevance background
patch — and the per-image relevance mass does not correlate with the causal effect.

## Setup

* Model: timm `vit_base_patch16_224` (ImageNet-1k, full head), via
  `experiments.crp_gallery.load_model` (tag `imagenet`).
* Data: `imagenet_val_hf`, `n_per_class=10` (10k pool), fixed shuffle (seed 0);
  scan pool = first 1024 shuffled images; N=128 correctly-classified images with
  ≥1 outlier patch (846 eligible; first 128 taken).
* Attribution: `crp.attribution.CondAttribution` + `lrp_configs.get("cp_lrp_baseline")`,
  condition `[{"y": [target]}]` on the normalized input; 224×224 heatmap summed
  into the 14×14 patch grid.
* Code: `experiments/scripts/registers_step3_occlusion.py` (phases scan / select /
  lrp / occlude) and `registers_step3_figures.py`. Arrays in
  `data/results/registers/step3_*.npz`; figures in
  `figures/registers/step3_occlusion/`.

### Outlier-token detection (documented choices)

Per-block L2 norms of block-output patch tokens (CLS excluded), forward hooks on
`backbone.blocks.{b}`. Threshold per block = mean + 4·sd over the 1024-image scan
pool. Pool statistics:

| blk | mean | sd | thr | p99.9 | p99.9/median | tok>thr | img>thr |
|----:|-----:|----:|----:|------:|------:|------:|------:|
| 3 | 8.8 | 1.2 | 13.6 | 24.3 | 2.8 | 0.83% | 93% |
| 5 | 12.0 | 6.6 | 38.6 | 79.4 | 7.2 | 1.57% | 100% |
| 6 | 14.2 | 11.0 | 58.2 | 114.8 | **9.0** | 1.76% | 100% |
| 8 | 22.7 | 13.0 | 74.8 | 134.8 | 6.6 | 1.81% | 100% |
| 11 | 64.9 | 18.2 | 137.9 | 144.9 | 2.4 | 0.34% | 59% |

High-norm outliers emerge at blocks 5–6 and the separation (p99.9/median) is
clearest at blocks 6–9; block 11 partially re-normalizes. **Detection set =
union over blocks {6,7,8,9}.** The choice is uncritical: outlier *positions* are
identical across blocks 5–11 (median Jaccard 1.0 between blocks 6/8/9), so the
union equals any single late block. Every pooled image has outliers under this
criterion (median 4 per image, max 6) — consistent with timm ViT-B/16 AugReg
showing the artifact behavior. The *primary* outlier patch = largest
threshold-relative norm. Qualitatively (examples figure) outlier patches sit in
uniform, information-poor regions (sky, bokeh, grass) — exactly the registers
paper's "redundant patch" account.

### Occlusion conditions (pixel space, [0,1], before normalization)

| cond | patch occluded | occluder |
|------|----------------|----------|
| a | primary outlier | constant fill = mean color of its non-outlier 8-neighbors |
| a_all | ALL outlier patches (median 4) | neighbor-mean, per patch |
| b | primary outlier | Gaussian noise matched to per-channel image mean/std, clipped |
| c | random non-outlier, non-outlier-adjacent, below-median-|R| patch | neighbor-mean |
| d | top-|R| patch outside the outlier 8-neighborhoods | neighbor-mean |

Metric: Δp = p(target | occluded) − p(target | clean), softmax over 1000 classes.
Median clean p(target) = 0.848.

## Results

### Δp per condition (N=128; Wilcoxon signed-rank vs 0)

| cond | median Δp | IQR | p |
|------|----------:|-----|---:|
| a (primary outlier, neighbor-mean) | **−0.0005** | [−0.0037, +0.0017] | 0.10 |
| b (primary outlier, matched noise) | **−0.0006** | [−0.0033, +0.0025] | 0.23 |
| a_all (all ~4 outlier patches) | **−0.0017** | [−0.0082, +0.0023] | 0.0013 |
| c (random background control) | −0.0007 | [−0.0044, +0.0024] | 0.05 |
| d (top-relevance control) | −0.0018 | [−0.0058, +0.0045] | 0.15 |

* Occluding the primary outlier patch is **indistinguishable from the random
  background control**: paired |Δp_a| vs |Δp_c|, Wilcoxon p = 0.12.
* The top-relevance control patch moves the prediction significantly more than
  the outlier patch (paired |Δp_d| vs |Δp_a|, p = 1.6e−7) — even though LRP ranks
  the outlier patch *above* patch d within the same image in the median case.
* Prediction (argmax) preserved in 100% of images for a/b/c/d and 127/128 for
  a_all.
* Ceiling caveat: single-patch occlusion is generally weak on ViT-B (even the
  top-relevance patch only gives −0.0018 median), so the informative contrasts
  are the *paired* a-vs-c (null) and a-vs-d (outlier ≪ object patch) comparisons,
  not absolute magnitudes.

### Relocation guard (hydra effect) — condition a, clean thresholds

* **48% (61/128)** of images show ≥1 outlier at a *new* position after occlusion
  (median 0 new positions overall; the count distribution is roughly half 0, half 1+).
* **58%** of occluded primary patches are *still* outlier tokens after their
  content is replaced by a flat neighbor-mean color — the scratch-pad location
  frequently survives content replacement outright, which is itself direct
  evidence that the *content* does not determine the storage site.
* Outlier count is conserved (median 4 → 4).
* Crucially, the null result is **not explained by relocation**: Δp_a is
  negligible both with relocation (median −0.0001, n=61) and without (−0.0008,
  n=67); Mann–Whitney p = 0.47. Prediction is preserved in 100% of both groups.

### Faithfulness contrast (LRP mass vs causal effect)

* Median |R| fraction on outlier patches: **23.7%** (vs 1.8% uniform share, ≈13×
  over-representation). Median |R| fraction on the single primary patch: **11.1%**
  (uniform 0.51%); the primary outlier patch is the **rank-1 patch of 196** by |R|
  in the median image.
* Correlation of per-image LRP mass with measured effect:
  * frac(primary) vs |Δp_a|: Spearman ρ = −0.09 (p = 0.33) — no relationship.
  * frac(primary) vs |Δp_b|: ρ = +0.07 (p = 0.43) — no relationship.
  * frac(all outliers) vs |Δp_a_all|: ρ = **−0.22** (p = 0.012) — weakly
    *negative*: images where LRP piles more relevance onto outlier patches show,
    if anything, *smaller* causal effects.

## Conclusion

The input relevance that cp_lrp_baseline assigns to outlier-token patches is not
matched by causal content-sensitivity: occluding the patch (two different
occluders) does no more than occluding a random background patch, far less than a
matched top-relevance object patch, the effect is uncorrelated (or weakly
anti-correlated) with the assigned relevance mass, and the high-norm token often
persists at the same position with entirely different content. The hydra-effect
confound was measured and ruled out as the explanation. **H_B rejected; supports
H_A: outlier-patch relevance is an attribution artifact of the scratch-pad
mechanism, not evidence that the model reads image content there.**

Follow-ups this suggests: (i) check whether the relevance arriving at outlier
patches travels through the value-path of late-block attention (links to step 2's
relevance-flow decomposition); (ii) repeat Δp with logit instead of prob (softmax
saturation at clean p≈0.85 compresses effects, though the a-vs-d contrast already
controls for this); (iii) same protocol on a registers-trained ViT where the
artifact should be absent.

## Files

* `experiments/scripts/registers_step3_occlusion.py` — phased experiment (scan/select/lrp/occlude)
* `experiments/scripts/registers_step3_figures.py` — figures
* `data/results/registers/step3_scan.npz` — pool norms (1024×12×196), preds, probs
* `data/results/registers/step3_selection.npz` — the 128 images, outlier masks, thresholds
* `data/results/registers/step3_lrp.npz` — 14×14 patch relevance per image
* `data/results/registers/step3_occlusion.npz` — per-condition probs/preds, relocation masks
* `data/results/registers/step3_analysis.npz` — derived Δp, fractions, relocation flags
* `figures/registers/step3_occlusion/{dp_by_condition,faithfulness_scatter,relocation,examples}.{png,pdf}`
