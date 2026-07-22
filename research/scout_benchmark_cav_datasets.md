# Scout report: datasets / occlusion protocols / flipping baselines / auto-CAV

*Sonnet literature-scout output, 2026-07-22, for XAI-21 paper planning. Verbatim
findings below; decisions distilled at top.*

## Distilled decisions

- **Localization datasets**: ImageNet-S300 primary (ImageNet-1k images, object
  masks, val 4,097 densely annotated; S50 val 752 for fast iteration) +
  PartImageNet secondary for part granularity (~24k imgs, 158 IN classes, part
  masks). CUB/Pets/PASCAL-Part rejected (domain shift / no parts).
- **Occlusion protocol**: Chefer et al. CVPR 2021 pos/neg token-perturbation AUC
  (mean-value baseline) as comparability anchor; ROAD noisy-imputation as
  leakage-robustness check; cite SaCo (CVPR 2024) critique of cumulative AUC.
- **Flipping baselines**: activation-magnitude ranking IS standard (CRP's own
  ActMax-vs-RelMax framing; Yeom 2021 Pruning-by-Explaining; 2024 "Pruning by
  Explaining Revisited" does it on ViTs — closest prior to our flipping setup,
  must-read for novelty). Random: no fixed K convention; TCAV uses N≥10; state
  own K + variance justification. Closed-form expected random deletion curve
  exists: Hama/Mase/Owen JMLR 2023 — citable to replace/validate Monte-Carlo.
- **Auto-CAV**: Text2Concept (ICML 2023), LG-CAV (NeurIPS 2024), TextCAVs =
  CLIP-text-derived CAVs without curated sets. Probe-set banks: Broden, DTD-47,
  PartImageNet. **No published pipeline** deriving CAVs from CRP representants +
  held-out retrieval generalization → genuine gap, novel contribution if
  circularity controlled (discover label via RelMax → build probe set from
  INDEPENDENT source (Broden/CLIP-retrieval), exclude representants → evaluate
  direction alignment + held-out retrieval).
- CAV robustness citation: Pahde et al. ICLR 2025 pattern-CAVs (same lab).

## Full report

### 1. Datasets with localization GT

| Dataset | Size | Annotation | IN domain match |
|---|---|---|---|
| PartImageNet | ~24k imgs, 158 IN classes | per-part instance seg masks | direct (IN subset) |
| CUB-200-2011 | 11,788, 200 birds | 15 part keypoints + bbox + attrs | partial (birds only) |
| Oxford-IIIT Pets | ~7,4k, 37 breeds | trimap whole-object | partial, no parts |
| ImageNet-S | S50: 64k/752/1,682; S300: 385k/4,097/9,088; full 919-class | semantic seg masks | exact (IS ImageNet) |
| PASCAL-Part | ~10k VOC2010 | part seg masks | weak |

Refs: PartImageNet He et al. ECCV 2022 arXiv:2112.00933 github.com/TACJu/PartImageNet;
ImageNet-S Gao et al. TPAMI 2022 arXiv:2106.03149 github.com/LUSSeg/ImageNet-S;
CUB vision.caltech.edu/datasets/cub_200_2011; Pets robots.ox.ac.uk/~vgg/data/pets.

### 2. Occlusion faithfulness protocols

1. **Chefer et al. CVPR 2021** (arXiv:2012.09838): pos perturbation = remove
   most-relevant patches first, track top-1 acc, AUC lower=better; neg = remove
   least-relevant first, AUC higher=better; up to 90% removal; predicted+target
   class variants. Most-cited ViT protocol.
2. **AOPC** Samek et al. TNNLS 2017 (arXiv:1509.06321): MoRF region perturbation,
   ancestor metric.
3. **RISE insertion/deletion** Petsiuk BMVC 2018 (arXiv:1806.07421): deletion AUC
   (lower better) + insertion AUC (higher better) pairing now default.
4. **ROAD** Rong et al. ICML 2022 (arXiv:2202.00449): noisy linear imputation
   fill — avoids occlusion-shape leakage; more consistent MoRF/LeRF.

Baseline fill values, prevalence order: per-image mean > zero > blur > dataset
mean > ROAD imputation. 2025 systematic comparison: arXiv:2512.11433 ("Back to
the Baseline"). ViT-native SOTA critique: **SaCo** CVPR 2024 (arXiv:2404.01415)
— pairwise salience-vs-confidence-drop over patch groups instead of cumulative
AUC. Also Vision DiffMask (arXiv:2304.06391).

### 3. Flipping-curve baselines

- ActMax vs RelMax = CRP paper's own framing (Achtibat NMI 2023).
- **Yeom et al. Pattern Recognition 2021** (arXiv:1912.08881) Pruning by
  Explaining: LRP-relevance vs weight-magnitude vs activation/Taylor vs random
  unit-deletion curves. Direct precedent.
- **Pruning by Explaining Revisited, ECCV-W 2024** (arXiv:2408.12568): the same
  on ViTs with CRP relevance. CLOSEST EXISTING WORK to our concept-flipping —
  read fully before claiming novelty of the benchmark.
- Random K: no convention; TCAV (arXiv:1711.11279) N≥10 + t-test is the citable
  precedent; deletion papers use 3–10 orderings, mean±std.
- **Analytic expected random deletion curve: Hama, Mase & Owen JMLR 2023**
  (arXiv:2205.12423) — closed-form via anchored decomposition.

### 4. Auto-CAV

- Probe banks: Broden (CVPR 2017, ~60k imgs, 468 scenes/585 objects/234 parts/
  32 materials/47 textures/11 colors), DTD-47, PartImageNet (underused,
  IN-native).
- CLIP-derived CAVs: **Text2Concept** ICML 2023 (arXiv:2305.06386) — map any
  encoder's features to CLIP space, CAV from text embedding, example-free;
  **LG-CAV** NeurIPS 2024 (arXiv:2410.10308); **TextCAVs** (concept-name-only).
- Pattern-CAVs robustness: Pahde et al. ICLR 2025 (OpenReview Q95MaWfF4e).
- **(iii) CRP-representant-seeded CAV + held-out generalization: NOT FOUND in
  literature — genuine gap.** Anti-circularity design: RelMax discovers the
  concept LABEL only → probe set built from independent source (Broden /
  CLIP-retrieval), representants excluded → evaluate (a) direction alignment
  with CRP-relevant subspace, (b) held-out retrieval vs masks on ImageNet-S/
  PartImageNet. First circularity-controlled CRP-vs-CAV benchmark.
