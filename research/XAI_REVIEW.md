# XAI for Vision Transformers — review

A literature snapshot of post-hoc explainability methods, libraries, and evaluation
protocols relevant to Vision Transformers (ViTs). Bias toward 2023–2024 results, with
older landmarks included when they are the foundation that newer methods build on.

---

## 1. SOTA attribution methods

| Method | Year | Paper | Code | ViT-ready? | One-line idea |
|---|---|---|---|---|---|
| Attention Rollout / Attention Flow | 2020 | [Abnar & Zuidema, ACL 2020](https://arxiv.org/abs/2005.00928) | [samiraabnar/attention_flow](https://samiraabnar.github.io/articles/2020-04/attention_flow) | Yes (any transformer) | Multiply attention matrices across layers (rollout) or run max-flow on the attention DAG. |
| Partial-LRP for attention | 2019 | Voita et al., ACL 2019 | - | Hand-implemented per arch | Apply epsilon-LRP only inside attention; treat softmax as identity. |
| Chefer (Transformer Interpretability Beyond Attention) | 2021 (CVPR) | [arXiv 2012.09838](https://arxiv.org/abs/2012.09838) | [hila-chefer/Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) | ViT, BERT | LRP + gradient-weighted attention, aggregated via rollout. Strong faithfulness, ViT-only. |
| Chefer (Generic Attention-model Explainability) | 2021 (ICCV oral) | [arXiv 2103.15679](https://arxiv.org/abs/2103.15679) | [hila-chefer/Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability) | Self-attn, co-attn, encoder-decoder | Extends T-Attr to bi-modal & encoder-decoder via per-attention rollout rule. |
| Conservative LRP for Transformers | 2022 (ICML) | [Ali, Schnake et al., arXiv 2202.07304](https://arxiv.org/abs/2202.07304) | [ameenali/xai_transformers](https://github.com/ameenali/xai_transformers) | Yes | Identifies LayerNorm & attention as the unreliability source; "detach" tricks restore the conservation property of LRP. Theoretical predecessor of AttnLRP. |
| AttnLRP (Attention-Aware LRP) | 2024 (ICML) | [Achtibat et al., arXiv 2402.05602](https://arxiv.org/abs/2402.05602) | [rachtibat/LRP-eXplains-Transformers](https://github.com/rachtibat/LRP-eXplains-Transformers) | LLaMA, Mixtral, Flan-T5, ViT, CLIP | Adds Deep-Taylor-derived rules for softmax & matmul. O(1)-pass, attributes both inputs *and* latents. Current SOTA on faithfulness benchmarks. |
| Beyond Intuition (TokenTM / IG-on-last-attention) | 2023 (TMLR) | [Chen et al., OpenReview](https://openreview.net/forum?id=rm0zIzlhcX) | - | BERT, ViT, CLIP | Integrated gradients applied to the last attention map with noise-decreasing baseline. |
| ViT-CX | 2023 (IJCAI) | [arXiv 2211.03064](https://arxiv.org/abs/2211.03064) | [vaynexie/CausalX-ViT](https://github.com/vaynexie/CausalX-ViT) | ViT | Mask-and-score on *patch embeddings* (not raw pixels); explicitly models causal overdetermination among redundant patches. |
| Transformer Input Sampling (TiS) | 2023 (ICCVW) | [Englebert et al.](https://openaccess.thecvf.com/content/ICCV2023W/NIVT/papers/Englebert_Explaining_Through_Transformer_Input_Sampling_ICCVW_2023_paper.pdf) | - | ViT, DeiT | Drop a random subset of patches at the embedding stage, run forward, weight masks by the resulting logit. Like a ViT-native RISE. |
| Attention Guided CAM | 2024 (AAAI) | [arXiv 2402.04563](https://arxiv.org/abs/2402.04563) | [leemsaebom/Attention-Guided-CAM](https://github.com/leemsaebom/attention-guided-cam-visual-explanations-of-vision-transformer-guided-by-self-attention) | ViT | Selective gradient aggregation weighted by *sigmoid-normalised* self-attention (replaces softmax peakiness). |
| ViT-ReciproCAM | 2023 | [arXiv 2310.02588](https://arxiv.org/abs/2310.02588) | - | ViT | Gradient-free and attention-free: masks tokens, looks at output reciprocity. Cheap and surprisingly competitive. |
| VTranM (Vector Transformation Measurement) | 2024 | [OpenReview b5LJVjwOsB](https://openreview.net/forum?id=b5LJVjwOsB) | - | ViT | Tracks vector-length and directional change through attention, aggregates across layers. |
| FViT (Faithful ViT via DDS) | 2024 (ICML) | [arXiv 2311.17983](https://arxiv.org/abs/2311.17983) | [kaustpradalab/FViT](https://github.com/kaustpradalab/FViT) | ViT | Trains a smoother ViT via Denoised Diffusion Smoothing; raw attention then becomes a more faithful explanation. |
| Concept Relevance Propagation (CRP) | 2023 (Nat Mach Intell) | [Achtibat et al., arXiv 2206.03208](https://arxiv.org/abs/2206.03208) | [rachtibat/zennit-crp](https://github.com/rachtibat/zennit-crp) | CNN; ViT via extensions (the host repo) | Conditional LRP: restrict the backward pass to a chosen concept (filter / head / latent), get *concept-conditioned* heatmaps. |
| Integrated Gradients on ViT | 2017 base / ongoing | [Sundararajan et al.](https://arxiv.org/abs/1703.01365) | Captum | Works but flawed | Path-integrated gradient w.r.t. input pixels. Baseline choice matters; saturation issues are well-documented for ViTs. |
| LIME on patches | 2016 base | [Ribeiro et al.](https://arxiv.org/abs/1602.04938) | [marcotcr/lime](https://github.com/marcotcr/lime) | Slow, model-agnostic | Treat each ViT patch as a super-pixel; fit a sparse linear surrogate. |

### Survey papers worth reading

- **Explainability of Vision Transformers: A Comprehensive Review and New Perspectives** — [arXiv 2311.06786](https://arxiv.org/abs/2311.06786). Builds the taxonomy used by most newer ViT-XAI papers.
- **The Explainability of Transformers: Current Status and Directions** — [MDPI Computers 2024](https://www.mdpi.com/2073-431X/13/4/92).
- **Explainability for Vision Foundation Models: A Survey** — [arXiv 2501.12203](https://arxiv.org/html/2501.12203v1). Covers SAEs, CRP, AttnLRP, and the foundation-model angle (DINOv2, SAM, CLIP).
- **Explainability and Evaluation of Vision Transformers: An In-Depth Experimental Study** — [MDPI Electronics 13/175](https://www.mdpi.com/2079-9292/13/1/175). Side-by-side numbers across most of the methods in the table above.

---

## 2. Off-the-shelf libraries

### Captum (Meta / PyTorch official)

- **Repo:** [meta-pytorch/captum](https://github.com/meta-pytorch/captum). Maturity: actively maintained, PyTorch-blessed.
- **What works on ViTs:** generic gradient methods (IntegratedGradients, Saliency, DeepLIFT, GradientShap, NoiseTunnel/SmoothGrad). They all run on a `timm`/HF ViT out of the box because they only need a callable.
- **What it does *not* ship:** transformer-specific LRP (no AttnLRP, no Chefer rollout, no conservative-LRP). LayerLRP exists but uses the generic ε rule and is unreliable on ViTs (the Conservative-LRP paper documents this).
- **Install:** `pip install captum`
- **Minimal example (10 lines):**
  ```python
  import timm, torch
  from captum.attr import IntegratedGradients
  model = timm.create_model("vit_base_patch16_224", pretrained=True).eval()
  x = torch.randn(1, 3, 224, 224, requires_grad=True)
  ig = IntegratedGradients(model)
  attr = ig.attribute(x, target=243, n_steps=32)   # 243 = "mastiff"
  heatmap = attr.squeeze().sum(0).detach().cpu()
  ```

### pytorch-grad-cam (jacobgil)

- **Repo:** [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam). Maturity: very active, ships dedicated ViT tutorial.
- **ViT support:** Yes via a `reshape_transform` that drops the CLS token and reshapes the 196 patch tokens to 14×14. Supports `GradCAM`, `GradCAMPlusPlus`, `ScoreCAM`, `AblationCAM`, `EigenCAM`, `EigenGradCAM`, `LayerCAM`.
- **Recommended target layer:** `model.blocks[-1].norm1`. The very last block can't be used because gradients flow only into the CLS token.
- **Install:** `pip install grad-cam`
- **Minimal example:**
  ```python
  from pytorch_grad_cam import GradCAM
  def reshape_transform(t, h=14, w=14):
      r = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2))
      return r.permute(0, 3, 1, 2)
  cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1],
                reshape_transform=reshape_transform)
  ```

### LRP-eXplains-Transformers (LXT, the AttnLRP repo)

- **Repo:** [rachtibat/LRP-eXplains-Transformers](https://github.com/rachtibat/LRP-eXplains-Transformers). Maturity: official ICML 2024 release, actively maintained.
- **ViT support:** First-class. Patches HF / timm transformers via `torch.fx` symbolic tracing so a single `monkey_patch(...)` call covers most architectures.
- **What it ships:** AttnLRP (default), CP-LRP (Conservative-LRP), gradient-based baselines, attention rollout. Faithful relevances *for latent representations* — needed for concept-level work.
- **Install:** `pip install lxt`
- **Minimal example (efficient input×grad path):**
  ```python
  from lxt.efficient import monkey_patch
  import timm, torch
  model = timm.create_model("vit_base_patch16_224", pretrained=True).eval()
  monkey_patch(model)
  x = torch.randn(1, 3, 224, 224, requires_grad=True)
  logit = model(x)[0, 243]
  logit.backward()
  relevance = (x * x.grad).sum(1).detach()
  ```

### Chefer's Transformer-Explainability

- **Repo:** [hila-chefer/Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) (ViT) and [Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability) (multimodal).
- **Maturity:** Reference implementation of the CVPR 2021 / ICCV 2021 papers. Read-only since ~2022; works but uses a forked ViT (`ViT_LRP.py`).
- **ViT support:** Yes, but you must instantiate their fork of the model. Not drop-in for arbitrary `timm` / HF checkpoints.
- **Install:** clone repo, `pip install -r requirements.txt`.
- **Minimal example:** see `Transformer_explainability.ipynb` in the repo. Pattern is `model.relprop(...)` after a forward pass.

### zennit (we use it)

- **Repo:** [chr5tphr/zennit](https://github.com/chr5tphr/zennit).
- **What it ships:** LRP rule zoo (Epsilon, AlphaBeta, Gamma, Flat, ZBox, ZPlus, WSquare), Composites (`EpsilonAlpha2Beta1`, `EpsilonGammaBox`, `EpsilonPlus`, …), Canonizers for ResNets, gradient-based methods, occlusion.
- **ViT-readiness:** No built-in transformer composite — and this is exactly the gap our project fills. The recipes that work today: `EpsilonAlpha2Beta1` over MLP blocks + AttnLRP-style bilinear rule on attention via a custom Composite/Canonizer (our `attention_unfolded` module).
- **Install:** `pip install zennit zennit-crp`

### Quantus (evaluation, not attribution)

- **Repo:** [understandable-machine-intelligence-lab/Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus). Maturity: actively maintained, JMLR 2023 paper.
- **30+ metrics** across faithfulness, robustness, localisation, complexity, randomisation, axiomatic. Pairs naturally with any of the attribution libraries above.
- **Install:** `pip install quantus`

### iNNvestigate

- **Repo:** [albermax/innvestigate](https://github.com/albermax/innvestigate). Maturity: mature but TF/Keras only, requires TF2 eager-execution disabled.
- **ViT support:** Effectively none — the codebase predates ViTs and TF transformer support is limited. Not recommended for new ViT work in 2024+.

### Interpreto (new, transformer-focused)

- **Paper:** [arXiv 2512.09730](https://arxiv.org/pdf/2512.09730) (late-2025 release, Hugging Face-style API for transformer explainability). Worth tracking; too young to recommend as a daily driver but it bundles AttnLRP, rollout, occlusion and Quantus metrics under one API.

---

## 3. Evaluation protocols

| Protocol | Year | Idea | Reference |
|---|---|---|---|
| Sanity Checks (model & data randomisation) | 2018 | Re-randomise weights or labels; a *faithful* attribution should change. Many saliency methods fail this. | [Adebayo et al., NeurIPS 2018](https://arxiv.org/abs/1810.03292) |
| Revisiting Sanity Checks | 2021 | Refines the protocol; shows some negatives in Adebayo et al. were measurement artefacts. | [Yona & Greenfeld, arXiv 2110.14297](https://arxiv.org/abs/2110.14297) |
| Insertion / Deletion AUC | 2018 | Sort pixels by relevance, progressively mask/reveal them, integrate the logit curve. Low DAUC + high IAUC = faithful. | [Petsiuk et al. (RISE), BMVC 2018](https://arxiv.org/abs/1806.07421) |
| Pointing Game | 2016 | Binary: is the argmax of the heatmap inside the GT bounding box? | Zhang et al. |
| Perturbation tests (Positive / Negative) | 2017 | Replace top-k% relevant pixels by their mean; measure logit drop (positive) or rise (negative). Used in Chefer. | Samek et al. |
| FunnyBirds | 2023 (ICCV) | Synthetic-bird dataset where parts can be *cleanly removed* in the rendering pipeline → ground-truth part importance. Six protocols across completeness / correctness / contrastivity. | [Hesse et al., arXiv 2308.06248](https://arxiv.org/abs/2308.06248), [github visinf/funnybirds](https://github.com/visinf/funnybirds) |
| Segmentation AUC on ImageNet-Seg | 2021 | Heatmap → binarised mask → mIoU / pixel-AP vs ImageNet segmentation labels. Used by Chefer. | [Chefer 2021](https://github.com/hila-chefer/Transformer-Explainability) |
| Faithfulness (ViT-specific) — SaCo | 2024 (CVPR) | Salience-guided Faithfulness Coefficient: pairwise compares salience ranks against actual influence drops. Authors show every "advanced" ViT method is barely distinguishable from random under previous metrics. | [Wu et al., arXiv 2404.01415](https://arxiv.org/abs/2404.01415) |
| Quantus toolkit | 2023 | 30+ metrics in one library, six categories. | [Hedström et al., JMLR 2023](https://www.jmlr.org/papers/v24/22-0142.html) |
| MetaQuantus | 2023 | Meta-evaluation: which Quantus metrics are themselves *reliable*? | [Hedström et al., TMLR 2023](https://github.com/annahedstroem/MetaQuantus) |

---

## 4. Related work / mechanistic & concept-level

- **Concept Relevance Propagation (CRP)** — [Achtibat et al., Nat Mach Intell 2023](https://www.nature.com/articles/s42256-023-00711-8). Conditional LRP that answers both "where" and "what". Direct foundation of this repository.
- **Mechanistic Interpretability of Fine-Tuned ViTs** — [arXiv 2503.18762](https://arxiv.org/abs/2503.18762). Per-head behavioural analysis on distorted images; finds early-layer heads are noise-suppressors, middle-layer heads are concept-monosemantic.
- **Seeing Through Circuits: Faithful Mechanistic Interpretability for Vision Transformers** — [arXiv 2604.14477](https://arxiv.org/html/2604.14477). Automatic Visual Circuit Discovery (Vi-CD), per-class subgraph extraction.
- **Sparse Autoencoders for Vision** — [osu-nlp-group/saev](https://osu-nlp-group.github.io/saev/) plus [arXiv 2502.06755](https://arxiv.org/html/2502.06755) ("Interpretable and Testable Vision Features via SAEs"). Decompose ViT residual stream into monosemantic features; the vision analogue of the Anthropic "Scaling Monosemanticity" work.
- **Causal Feature Explanation (CaFE) for vision SAEs** — [arXiv 2509.00749](https://arxiv.org/html/2509.00749v1). Effective-receptive-field-based attribution to isolate the patches that *cause*, not just co-occur with, an SAE feature firing.
- **PatchSAE for CLIP** — sparse-autoencoder lens specifically for CLIP-ViT, recovers spatial+semantic concept attributions per patch.
- **R-Cut (Relationship Weighted Out and Cut)** — [PMC paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11085337/). Graph-cut framing of ViT explanations.
- **Anthropic Transformer Circuits work** (NLP-focused but doctrinally relevant) — [transformer-circuits.pub](https://transformer-circuits.pub). Induction heads, QK / OV circuits, monosemanticity.

---

## 5. What's directly usable in OUR project

Ranking by *plug-in effort* to the unfolded-walkthrough notebook today.

1. **AttnLRP via the existing `crp/` composites.** Already wired. Nothing to add. Continue as the default.
2. **Chefer T-Attr as a sanity baseline.** ~50 lines: clone the repo, instantiate their `vit_base_patch16_224` fork, call `model.relprop`. Plot side-by-side with our AttnLRP heatmap to demonstrate qualitative agreement.
3. **pytorch-grad-cam Grad-CAM / EigenCAM on `blocks[-1].norm1`.** Two cells. Useful as the "what classical XAI sees" reference. The recipe above is copy-pasteable.
4. **Captum IntegratedGradients on the input.** One cell. Slow but model-agnostic — gives a third independent baseline.
5. **Quantus evaluation cell.** One import, one wrapper call per metric. Adds insertion/deletion AUC + pointing game + faithfulness number per method. Most leverage per LOC.
6. **FunnyBirds quantitative protocol.** Higher effort (~half-day): FunnyBirds dataset already in `experiments/datasets.py` per the task log; need their `evaluation_protocols/` scripts to compute the six dimensions. Gives publishable numbers.
7. **LXT (`pip install lxt`) as a second-opinion AttnLRP.** Useful for cross-checking our implementation against the reference. ~10 lines.
8. **CP-LRP (Conservative-LRP) baseline.** Already implemented in LXT and in Chefer's fork. Reproduces the 2022 paper's numbers; would close a methodological loop.
9. **SaCo metric** for the writeup. Worth a single cell because it specifically targets the "random baseline confound" that other faithfulness metrics suffer from on ViTs.

---

## 6. Open questions / gaps the field hasn't closed

- **Latent-relevance evaluation.** Insertion/deletion only measures *input* attributions. Almost nothing measures whether the *intermediate* relevances that CRP / AttnLRP produce are themselves faithful. Our register-token / Q-leaf / K-leaf decomposition is exactly this granularity — there is room to define a new evaluation here.
- **Per-head / per-Q-vs-K attribution.** Mechanistic-interpretability ViT papers analyse heads behaviourally (ablation, activation patching), but no published method gives a *faithful relevance score per Q×K interaction* the way our unfolded-attention rule does. This is a genuine novelty axis.
- **CLS-token bypass.** Every gradient-based method on ViT routes through CLS in the final block. Recent register-token architectures (DINOv2 / DINOv3) muddy this further. Most XAI methods silently break on them; we already had to write `RegisterTokenConcept`.
- **SAE × LRP fusion.** SAE features are the right *vocabulary* (monosemantic, sparse); LRP/CRP is the right *propagation*. No paper has yet routed CRP backward through an SAE-decomposed residual stream. Looks like a natural extension.
- **Trust calibration of "advanced" methods.** SaCo 2024 already flagged: under proper metrics, many SOTA ViT explanations are statistically close to random. AttnLRP wins on faithfulness benchmarks but the *interpretation* of its attributions (e.g. negative relevance to a foreground patch) is under-studied.
- **Multi-resolution / hierarchical ViTs (Swin, MaxViT).** All evaluation work above is on plain ViT-B/L. Hierarchical models, where receptive fields are non-uniform, are an open frontier.
- **Faithful explanations under fine-tuning vs prompting.** Most XAI is evaluated on fully-supervised ImageNet ViTs. Whether the same methods stay faithful for CLIP-ViT, DINOv2/v3 linear probes, or LoRA-fine-tuned models is largely empirical and contested.
