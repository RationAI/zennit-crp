# Related-work brief: AttnLRP (2402.05602) & CRP original (2206.03208)

Careful full-text read (ar5iv HTML, both papers incl. appendices/supplement). For the paper "CRP concept detectors for ViTs". Equation numbers are the papers' own.

---

## 1. AttnLRP — Achtibat, Hatefi, Dreyer, Jain, Wiegand, Lapuschkin, Samek. ICML 2024, arXiv:2402.05602

### 1.1 Method core

- LRP framing: additive decomposition f_j(x) ∝ R_j = Σ_i R_{i←j} (Eq 1), lossless aggregation R_i = Σ_j R_{i←j} (Eq 2), layer-wise conservation Σ_i R_i^{l-1} = Σ_j R_j^l (Eq 3). Rules derived via Deep Taylor Decomposition: first-order expansion at reference point x̃ (Eq 4), generic rule R_i = Σ_j J_ji x_i R_j/(f_j(x) + ε·sign(f_j)) (Eq 5), ε ≈ 1e-6.
- Linear layers: ε-rule R_i = Σ_j W_ji x_i R_j/(z_j+ε) (Eq 8); bias b_j absorbs part of the relevance. Element-wise nonlinearities: identity rule R_i^{l-1} = R_i^l (Eq 9).
- **Softmax** (Prop 3.1, Eq 13): Taylor at x *with* bias term → R_i^l = x_i (R_i^{l+1} − s_i Σ_j R_j^{l+1}). The hidden bias (≈ softmax(0)=1/N) deliberately absorbs relevance ⇒ strict conservation is intentionally relaxed at softmax. Alternatives (Voita bias-distribution; Ding/Chefer identity rule) shown numerically unstable (App A.2.1, Fig 5: relevances explode to ±1e15).
- **Matrix multiplication A·V** (Props 3.2–3.3): decompose bilinear summands A_ji·V_ip; uniform rule R_{i←j} = R_j/N for N-ary multiplication (Eq 14; provably identical under Shapley with zero baseline); combined with ε-rule gives R_ji(A_ji) = Σ_p A_ji V_ip R_jp/(2·O_jp + ε) (Eq 15) — each operand (A and V) gets **half** the relevance; conservation strictly holds, no bias.
- **LayerNorm/RMSNorm** (Prop 3.4, Eq 19): Taylor at reference point **0** ⇒ identity rule through the x_j/g(x) normalization; affine parts (γ, β, mean subtraction) handled by ε-rule; normalization removed from the backprop graph. This *derives* Ali et al. (2022)'s heuristic from DTD.
- **γ-rule for ViTs** (Eq 24; z+ rule Eq 25 = γ→∞ case): applied *only in ViTs* (gradient shattering; LLMs are noise-free with ε everywhere), to conv + linear FFN layers *outside* the attention module; optionally z+ on softmax (App A.2.3, Fig 12 — propagating relevance through softmax with z+ beats blocking it).
- **ViT composite** (App B.4, Table 4), found by per-layer-type perturbation sweep (Figs 12–16): Convolution γ=0.25; Linear γ=0.05; LinearInputProjection (W_q, W_k, W_v) ε; LinearOutputProjection (W_o) ε.
- CP-LRP (Ali et al. 2022) contrast (App A.2.2, Table 3): treats softmax output as constant ⇒ attention becomes linear in V, relevance flows only through the value path; "the query and key matrices can no longer be attributed".

### 1.2 What they demonstrate, exactly

- **Table 1 (faithfulness + plausibility)**: metric = area between LeRF and MoRF perturbation curves (Blücher et al. 2024; ΔA^F/ΔA^I, App B.2 Eqs 30–31), baseline value zero. Models/datasets: ViT-B-16 (torchvision) / ImageNet cls (3200 samples, **input-pixel** flipping); Llama 2-7b / IMDB (fine-tuned head, 93% acc) and Wikipedia next-word (context 512), token-embedding flipping; Flan-T5-XL / SQuAD v2 (+ IoU vs answer mask). AttnLRP wins everywhere: ViT 6.19 vs γCP-LRP 6.06 vs CP-LRP-all-ε 2.53; Llama2-Wiki 10.93 vs CP-LRP 7.85 vs G×AttnRoll 9.79; SQuAD 1.76 vs 1.74 (CP-LRP); IoU 0.840 vs 0.830.
- Per-dataset breakdowns: Tables 5–8 (AttnLRP all-ε on ViT is only 2.79 — the γ composite, not the attention rules, provides most of the ViT gain: 6.19 "best" vs 2.79 "all epsilon").
- **Table 2 + App B.6 (efficiency)**: LRP w/ checkpointing O(1) compute / O(√N_L) memory vs O(N_T) perturbation; cost/time/memory curves on Llama 2-13b/70b (Fig 8; 70b exceeds 160 GB for LRP).
- **§4.3 + App B.7 (latent understanding, LLM only)**: Phi-1.5; ActMax reference sentences from Wikipedia-summary dataset for FFN "knowledge neurons" (last nonlinearity, GELU output); AttnLRP heatmaps *on* those reference samples; weight-row logit-lens projection. Neuron interventions: deactivate #3948 (L17 "cold"), amplify #5687 (L18 "candy") / #4104 (L17 "dry") flips 'Arctic'→'candy store'/'Desert' (Figs 2, 3, 9–11).

### 1.3 What they do NOT do (evidence for our positioning)

- **No CRP apparatus**: no conditional (concept-masked) backward passes, no RelMax, no reference-image galleries. Latent analysis = relevance *ranking* of neurons + **ActMax** sample retrieval (§3.4 explicitly: "researchers rely on Activation Maximization (ActMax)"; strategy = "(1) Collect prompts that lead to the highest activation of a unit. (2) Explain the unit's activation using AttnLRP"). CRP (Achtibat et al. 2023) is cited only for the idea of ranking latent relevance.
- **No ViT latent concepts**: §3.4/§4.3 restrict latent analysis to LLM FFN knowledge neurons ("In this work, we concentrate on knowledge neurons … situated at the last non-linearity in FFN layers"); the ViT appears only for input-space faithfulness (§4.1). No embedding-dim / head / value / query concept sites in ViTs anywhere. Fig 6 caption defers exactly this: "the detailed investigation of the processes within the attention model can be investigated with AttnLRP only … We leave these further explorations for future work."
- **No latent flipping benchmark**: faithfulness is input-space (pixel/token) perturbation only (§4.1, App B.2). Neuron manipulation in §4.3 is anecdotal (3 neurons, 2 prompts), not a systematic MoRF/LeRF protocol over latent concepts.
- **No SAE / alternative bases**: unit of analysis is the raw neuron throughout.

### 1.4 Quotables (verbatim)

1. "While partial solutions exist, our method is the first to faithfully and holistically attribute not only input but also latent representations of transformer models with the computational efficiency similar to a singular backward pass." (Abstract)
2. "Consequently, the query and key matrices can no longer be attributed, which reduces the faithfulness and makes latent explanations in query and key matrices infeasible." (App A.2.2, on CP-LRP)
3. "Moreover, the detailed investigation of the processes within the attention model can be investigated with AttnLRP only, while it is not possible with CP-LRP. We leave these further explorations for future work." (App B, Fig 6 caption)

### 1.5 Technical details we must stay consistent with

- Rule table (their Table 3): AttnLRP = softmax Taylor-at-x-with-bias + matmul ε&uniform (Eq 15) + LayerNorm identity; CP-LRP = softmax constant + value-path ε + LayerNorm identity.
- Conservation caveats: (a) softmax bias absorbs relevance (conservation deliberately broken there); (b) Eq 15 splits relevance 1/2 to A-path and 1/2 to V-path. If we report per-site relevance sums/conservation checks, account for both.
- ViT composite γ values (0.25 conv, 0.05 linear, ε on all four attention projections) were *tuned by perturbation grid search* (Pahde et al. 2023 style); "Tuning the γ-parameter in ViTs to obtain faithful attributions is necessary" is a stated open problem.
- Their ViT perturbation flips **pixels** (not patch tokens) to zero, relevance from input-pixel heatmaps; LLMs flip whole token embeddings.
- Faithfulness metric of record: Blücher et al. 2024 area between LeRF/MoRF curves — same family as our AOPC-style area-between-curves score; cite for lineage.

### 1.6 Pitfalls for our setup

- **CP-LRP is two baselines, not one.** Their Table 1/5 shows CP-LRP(all-ε) is catastrophically noisy on ViT (2.53) while CP-LRP(+γ composite) is nearly AttnLRP-level (6.06 vs 6.19). Our `cp_lrp_baseline` must state which variant it is; comparing against ε-only CP-LRP would be a strawman by their own data.
- Their own numbers show AttnLRP's ViT input-faithfulness edge over γCP-LRP is small (≈2%); the *qualitative* differentiator they claim is Q/K attributability. If our query/key concept sites work, that directly instantiates what they "leave for future work" — strong positioning, but conditional heatmaps at Q/K sites *require* AttnLRP-style (not CP-LRP) propagation; under CP-LRP those sites receive no relevance by construction.
- If we use z+ on softmax (their App A.2.3 option) vs plain Taylor softmax rule, faithfulness differs (Fig 12); document the choice per lrp_config.
- Their softmax rule's non-conservation means concept relevances upstream vs downstream of softmax are not comparable in absolute terms.

---

## 2. CRP original — Achtibat, Dreyer, Eisenbraun, Bosse, Wiegand, Samek, Lapuschkin. Nat. Mach. Intell. 5:1006–1019 (2023), arXiv:2206.03208

### 2.1 Method core

- LRP base: z_ij = a_i w_ij, z_j = Σ_i z_ij (Eqs 2–4); R_{i←j} = (z_ij/z_j)·R_j (Eq 5); R_i = Σ_j R_{i←j} (Eq 6); conservation Eq 26 (supplement).
- **CRP conditional message** (Eq 7 = supp Eq 27): R_{i←j}^{(l-1,l)}(x|θ∪θ_l) = (z_ij/z_j) · Σ_{c_l∈θ_l} δ_{j c_l} · R_j^l(x|θ). θ = condition set of network-element ids per layer; implemented as binary masking of relevance tensors in a single backward pass. Layers without conditions: δ ≡ 1 (unconstrained flow). Conditions within a layer act as logical OR, across layers as AND (Supp B.2).
- Conv layers: mask on the **channel axis only**, voxel indexing (p,q,j) (Eq 8 = supp Eq 28); justification = spatial weight sharing ⇒ channel ≈ concept; for dense layers, per-neuron.
- Concept relevance = Σ_i R_i^l(x|θ) (Eq 9) "in any layer l where θ has taken full effect"; local/regional variant Σ_{i∈I} (Eq 10); concept atlas = per-superpixel argsort of R_I(x|θ_c) (supp Eq 36); hierarchical concept composition via message aggregation R_{i←j}^{(l-1,l)} = Σ_{u,v}Σ_{p,q} R_{(u,v,i)←(p,q,j)} (Eqs 11–12 = supp 32–33) → concept composition graphs.
- **RelMax** (Eq 17 = supp Eq 43): maximization targets T^rel_sum(x) = Σ_i R_i(x|θ) and T^rel_max(x) = max_i R_i(x|θ), vs ActMax T^act_sum = Σ_i z_i (Eq 13), T^act_max = max_i z_i (Eq 14); reference set X*_k = top-k of argsort-desc T (Eqs 15–16). Class-/concept-conditional via θ; class-conditional reference sets (Supp D.2).
- Reference-sample post-processing (Supp C.4–C.5): crop to receptive field computed with the LRP ♭(flat)-rule (supp Eq 45, rule Eq 52); heatmap-mask crops at 40% of max relevance; concept localization by initializing the backward pass at channel activations R_(p,q,j) = z_(p,q,j)·δ_jc.
- Channel similarity ρ_qp = symmetrized mean cosine similarity of ReLU'd channel activation maps over each other's reference sets (Eqs 18–19), t-SNE on d = 1−ρ.
- **LRP composite** (Supp L.2): LRP_{ε-z+-♭} — ♭ first conv layer, z+ (Eq 53) remaining conv layers, ε (Eq 51) dense layers — after BatchNorm canonization; initialization R_j^L = f_j(x) (logit, not softmax); optional per-layer relevance normalization R/Σ|R| (Eq 54) for cross-dataset reference retrieval.

### 2.2 What they demonstrate, exactly

- Models/datasets (Supp L.1) — **all CNNs, no transformer anywhere**: VGG-16 (±BN) / ImageNet; ResNet34 / CUB-200 (fine-tuned, 76%); VGG-16 / ISIC 2019 (82.15%); VGG-16 / Adience age+gender; 3-block 1D-CNN / MIT-BIH ECG; LeNet-5 / Fashion-MNIST.
- Fig 3: conditional heatmaps, masked reference samples, concept atlas (layer3.0.conv2), concept composition graph (features.24→26→28, "animal on branch") for bird classification.
- Fig 4 + Supp H.3: Clever Hans watermark filter 361 (features.30, VGG-16 BN); zeroing the 20 most relevant filters, confidence tracking; inverse search: ≥7 ImageNet classes contaminated.
- Fig 5 + Supp H.4: activation-similar channel clusters (features.40) whose *relevances* diverge per class — fine-grained laptop vs remote decisions; "similarly activating channels do not necessarily encode redundant information".
- Fig 6 + §3.4/Supp G: MTurk user study (25/group, between-subject, Sept 2022), border-artifact detection: CRP TPR 89.1±2.4% / TNR 72.6±3.4%, beats LRP, SHAP, Grad-CAM, IG (p < 8e-4); CRP scores *lowest* on perceived clarity.
- Supp D.1 (Case studies 1–3 + Figs 10–16): ActMax vs RelMax divergence (polysemantic car/pattern filter; watermark filter; T_sum vs T_max sensitivity); set-intersection robustness analysis — RelMax sets more stable to sum-vs-max target choice.
- **Supp H.1 "filter flipping"**: rank channels by spatially sum-aggregated relevance, zero activation maps successively (most-relevant-first and least-relevant-first), measure relative confidence; VGG-16 features.28/.14/.0, 250 ImageNet samples, restricted to predictions >50% softmax. Findings: relevance power-law distributed; 14/16/20 top filters suffice for 85/90/95% confidence drop; ~300/512 filters uninvolved; ResNet34 layer-wise relevance flow + per-layer flipping (Supp Fig 38).
- Supp H.2: ISIC band-aid concept **insertion/replacement** by α-blended activation-tensor transplantation with spatial masks (supp Eqs 46–48) at features.0/.10/.28; class-score and per-channel relevance tracked.
- Supp I: gender-classification bias concepts (Adience); Supp J: ECG time-series concepts; Supp F: runtimes (preprocessing 21–50 min for 50k ImageNet val over all 13 VGG conv layers; per-sample explanation <1 s; full 512-concept atlas layer ≈8–12 s).

### 2.3 What they do NOT do (evidence for our positioning)

- **No transformers/ViTs**: architecture list is exhaustively CNN (Supp L.1). §5.1 closes with "an adaptation of the CRP approach beyond CNN, e.g., to recurrent [10] or graph [115] neural networks, is possible" — transformers are not named even as future work; attention appears only in the related-work paragraph on *self-explainable architectures* (Supp A). No attention/softmax/LayerNorm propagation rules exist in the paper — the ε-z+-♭ composite (Supp L.2) is undefined for a ViT.
- **Concept unit = conv channel or dense neuron.** The channel≈concept assumption is *justified via spatial weight sharing of conv filters* (§5.1, Supp B.3.1) — an argument that does not transfer to ViT embedding dims. Subspaces are acknowledged ("In principle, a concept can also refer to a set of filters … or spanning a concept-defining subspace. In general, CRP is applicable without restrictions in such a case", Supp F.1) but only explored via post-hoc activation clustering (§3.3/H.4) — **no learned dictionaries / SAEs**.
- **Latent flipping exists but is not a propagation-method or basis benchmark**: Supp H.1's filter flipping is a workload/sparsity analysis ("how much explanation is needed"), on CNN channels only, single composite, confidence-relative metric, confident samples only. No comparison across attribution variants, layers as sites, or concept bases; no AOPC-style scoring of competing recipes.
- Faithfulness of CRP itself is argued by inheritance from LRP + the user study (plausibility); there is no input- or latent-perturbation faithfulness comparison against other attribution methods.

### 2.4 Quotables (verbatim)

1. "In this work we introduce the Concept Relevance Propagation (CRP) approach, which combines the local and global perspectives and thus allows answering both the 'where' and 'what' questions for individual predictions." (Abstract)
2. "Following the LRP methodology, an adaptation of the CRP approach beyond CNN, e.g., to recurrent [10] or graph [115] neural networks, is possible." (§5.1 — useful to show ViTs were out of scope)
3. "Simply because an image leads to high activation does not mean that the image is representative of the neuron's function in a larger inference context. Adversarial examples are a prime example of this." (Supp C.3 — motivates RelMax galleries)

### 2.5 Technical details we must stay consistent with

- Conditional-heatmap definition = Eq 7 masking semantics (OR within layer, AND across layers; unconditioned layers pass all relevance). Our conditional ViT heatmaps should be describable in exactly this formalism.
- RelMax target definitions: T^rel_sum / T^rel_max (Eq 17) and ActMax T^act_sum / T^act_max (Eqs 13–14); their figures default to X*_8 with T^rel_sum ("X*_8 rel_sum"), receptive-field cropping, 40%-of-max heatmap masking. If our galleries use different k/target/crop, say so.
- Relevance initialization at the logit R_j^L = f_j(x) (explicitly recommended; multi-output init discouraged, Supp Fig 1); per-layer normalization (Eq 54) only for extended/OOD reference datasets.
- Conservation statement: concept relevance Σ_i R_i valid "at or below the lowest layer where the concept is expressed" — matches how we sum conditional relevances.
- Filter-flipping protocol (H.1) is the direct precedent for our concept flipping: rank by spatially aggregated relevance, zero activations, MoRF + LeRF orders, confidence relative to initial value, ≥50%-confidence samples. Cite it; differentiate: ours is a systematic benchmark across ViT sites (embedding-dim/head/value/query/block-output), propagation recipes, and raw-vs-SAE bases with AOPC-style scoring — theirs is one CNN model-composite pair used to argue explanation sparsity.

### 2.6 Pitfalls for our setup

- **Cannot claim "first latent concept flipping" or "first latent intervention"**: Supp H.1 (filter flipping) and H.2 (activation transplantation, Eqs 46–48) already flip/edit latent CNN concepts; AttnLRP §4.3 edits LLM neurons. Safe claim: first *systematic concept-flipping faithfulness benchmark for CRP-style concept detectors at ViT internal sites*, and first to compare raw-latent vs SAE concept bases under it.
- Their z+-heavy CNN composite discards negative conv-layer relevance; our ViT composites (ε/γ) keep signed relevance — sign conventions of "concept relevance" differ across the two papers' practice; be explicit which we use before ranking/flipping.
- The channel=concept justification (spatial invariance) does not carry to ViT embedding dims; we must give our own grouping argument (token-axis sharing of the channel/feature axis) — and note this is precisely where SAE bases enter (their acknowledged polysemanticity: Supp C.3/K.1).
- CRP's atlas/composition machinery assumes spatial activation tensors (p,q,j); ViT token axis substitutes for (p,q) — fine, but say so rather than implying their equations apply unchanged.
- RelMax rankings inherit model confidence (init at logit); if our gallery collection normalizes per layer (their Eq 54) behavior changes materially (their Supp Fig 57) — pin the choice.

---

## 3. Cross-paper positioning summary (for the intro)

- CRP (2023) supplies the concept-conditional relevance + RelMax gallery apparatus, but only for CNNs, with channel=concept, and evaluates via a user study + exploratory CNN filter flipping.
- AttnLRP (2024) supplies faithful transformer LRP rules (softmax/matmul/norm + ViT γ composite) and input-space faithfulness benchmarks, but its latent analysis is ActMax-based, LLM-FFN-only, anecdotal, and explicitly defers attention-internal investigation to future work.
- Neither paper: (a) runs the full CRP apparatus (conditional heatmaps + RelMax galleries) at ViT internal sites; (b) benchmarks latent-concept flipping on ViTs systematically; (c) considers any basis other than raw neurons/channels (no SAEs). Our paper composes (CRP conditioning) ∘ (AttnLRP/CP-LRP propagation) at ViT sites and adds the missing faithfulness benchmark + basis comparison — differentiable without overclaiming novelty on either ingredient or on latent intervention per se.
