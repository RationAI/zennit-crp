# SAE-basis CRP for Vision Transformers — experiment plan

**Status:** EXECUTED 2026-06-09 — both sites × {embed_dim, sae} × 3 datasets × 12 blocks; 72 SAEs (FVU 0.005–0.044); results at zennit-flip. See `slack_tasks.md` DONE entry + `experiments/sae.py`. Kept as design rationale.
**Owner:** zennit dev session. **Branch:** `transformer-multi-concept`.
**Origin:** Tomáš Brázdil's question — our concept detectors are *axis-aligned*
(`ChannelConcept`/`HeadConcept`/`EmbeddingDimConcept`/`TokenConcept`), but
superposition says features are *non-axis-aligned*. This plan tests whether
swapping CRP's concept basis from raw channels/dims to a **learned sparse
autoencoder (SAE) dictionary** gives more faithful, more interpretable
concept-relevance explanations.

---

## 1. Research question (the one worth publishing)

> **Does decomposing a ViT's prediction with CRP onto a trained SAE dictionary —
> instead of raw embedding dimensions / heads — yield more faithful (concept-
> flipping / AOPC) and more human-interpretable concept-relevance explanations,
> and does it resolve the superposition objection while preserving LRP relevance
> conservation?**

Two sub-questions:
- **Q1 (faithfulness).** Is the concept-flipping score `AOPC_most − AOPC_least`
  on the SAE basis **higher** than on the `EmbeddingDimConcept` axis-aligned
  basis at the same probe site (same model, same images)? A monosemantic basis
  should concentrate relevance into fewer, cleaner detectors → faster MoRF
  collapse → larger score.
- **Q2 (conservation).** Does relevance stay conserved when propagated through
  the spliced SAE (Σ feature-relevance ≈ relevance at the probe site, up to the
  reconstruction residual)? This is what separates us from gradient attribution.

## 2. Why it is publishable / novelty positioning

CRP (Achtibat et al., *Nature Machine Intelligence* 2023, arXiv **2206.03208**)
is faithful and conservation-respecting but its concepts are axis-aligned.
Superposition (Elhage et al., *Toy Models of Superposition*, arXiv **2209.10652**)
says the real features are non-axis-aligned; SAEs recover an approximately
monosemantic basis (Cunningham et al., ICLR 2024, arXiv **2309.08600**; Bricken
et al., *Towards Monosemanticity*, Transformer Circuits 2023). The crisp gap:

| Prior art | What it does | Why we differ |
|---|---|---|
| **CaFE** — Han, Kim, Kwak, arXiv **2509.00749** (2025) | AttnLRP from one SAE feature **→ input patches** (feature→input attribution, "what made this feature fire") | We go **output→SAE-basis**: decompose the *class logit's* relevance onto SAE latents (CRP direction), with concept-conditional masking. Opposite direction. |
| **Sparse Feature Circuits** — Marks et al., arXiv **2403.19647** (ICLR 2025) | Gradient / attribution-patching on SAE features, LMs | We use **LRP-rule propagation with conservation**, not gradient×act; ViT not LM. |
| **CRP** (2206.03208) | Output→concept relevance, axis-aligned (channels/heads) | We replace the axis-aligned basis with a **learned SAE basis**. |

So the headline "LRP through an SAE on a ViT" is **not** itself novel (CaFE did
the feature→input version). Our defensible contribution is the **CRP move**:
output-relevance *decomposition / conditioning* onto an SAE basis, with a
quantitative faithfulness comparison (concept-flipping/AOPC) against axis-aligned
CRP, plus a relevance-conservation guarantee. Position the paper explicitly
against CaFE and SFC on these two axes.

## 3. What an SAE is (definition we implement)

Standard untied SAE with pre-encoder bias (Bricken et al. 2023 form). For a
ViT activation `a ∈ ℝ^d` at the probe site, overcomplete dictionary of
`m = α·d` features (`α` = 8–32):

```
f   = ReLU(W_enc (a − b_dec) + b_enc)        # f ∈ ℝ^m, sparse codes
â   = W_dec f + b_dec  =  b_dec + Σ_i f_i d_i  # d_i = unit-norm columns of W_dec
L   = ‖a − â‖₂²  +  λ · Σ_i f_i ‖d_i‖₂        # recon + L1 sparsity
```

Weights untied (`W_enc ≠ W_decᵀ`), decoder columns unit-normalized, tied
`b_dec` subtracted before encoding. **Default variant: vanilla L1 SAE** for the
first pass (simplest, matches the canonical definition); keep **TopK** (Gao et
al., arXiv **2406.04093**) as a drop-in if L1 shrinkage hurts reconstruction.
JumpReLU / Gated are later options. Train per probe site (one SAE per block we
study), on activations collected from the trained probe's own data.

Vision precedent that SAEs on ViT activations are interpretable: Stevens et al.
(`saev`, arXiv **2502.06755**), Joseph et al. (arXiv **2504.08729**), PatchSAE
(arXiv **2412.05276**) — all hook the **residual stream / per-patch tokens**.

## 4. Implementation

### 4.1 Probe site (decision needed from team)
Proposed: the **same `proj_drop` block-output site** the concept-flipping
experiment already probes (`backbone.blocks.{b}.attn.proj_drop`), so the SAE
basis is directly comparable to `EmbeddingDimConcept` at that site. Activations
are `(B, N, d=384)` on vit_small. **Alternative** (closer to the SAE literature):
the block **residual stream** output. *Team: confirm `proj_drop` vs residual.*

### 4.2 Collect activations
Forward the trained probe over its dataset (dsprites / colored_mnist /
funny_birds), capture `proj_drop` outputs with a forward hook (the experiment
already does exactly this in `concept_flipping.py:286`), flatten tokens →
`(num_images·N, 384)` matrix. Store as a parquet/`.npy` under
`data/results/sae/acts_{dataset}_block{b}.npy`.

### 4.3 Train the SAE
New module `experiments/sae.py`: a tiny `SparseAutoencoder(nn.Module)` (the
equations above) + a `train_sae()` loop (Adam, normalize decoder each step,
resample dead latents). CLI mirrors `concept_flipping.py` (typer). Save
`data/results/sae/sae_{dataset}_block{b}.pt` + a metrics json (recon MSE,
fraction-variance-unexplained, mean L0, dead-feature count).

### 4.4 Splice the SAE into the graph as a recordable concept layer  *(the CRP move)*
A canonizer `SAEProbeCanonizer` inserts, at the probe site, the trained SAE as a
**reconstruction pass-through** so the forward output is preserved:
`a ↦ â = W_dec ReLU(W_enc(a − b_dec) + b_enc) + b_dec`, with the **feature
activation `f` exposed as a named `nn.Module` sublayer** (`sae.features`). Give
the decoder linear an **ε-LRP rule** (and the encoder a Pass/Epsilon rule) so
CRP relevance propagating back from the logit lands on `f` with conservation.
Because `â ≈ a`, splicing perturbs the forward pass minimally (report the recon
error as a control). Relevance recorded at `sae.features` is `(B, N, m)`.

> This is what makes it CRP, not CaFE: relevance originates at the **output
> logit** (`relevance_init` on the target class, as in `concept_flipping.py`)
> and is **decomposed onto the SAE latents**, not originated at a feature.

### 4.5 `SAEConcept(Concept)`  — `crp/concepts.py`
Mirror `EmbeddingDimConcept` exactly, but in feature space: one concept id = one
SAE feature; `attribute()` sums the recorded `(B, N, m)` relevance over the
token filter → `(B, m)` per-feature relevance; `mask()` zeroes the gradient for
all but the selected features (concept-conditional propagation). `n_grid =
1..m`. Drops straight into `concept_detectors()` in `concept_flipping.py` as a
new `--concept sae` branch (with the matching `D` feature-membership matrix).

### 4.6 Evaluation (reuse existing machinery)
- **Faithfulness:** run the existing concept-flipping experiment with
  `--concept sae` at the studied block(s); compute `AOPC_most − AOPC_least` and
  compare to `--concept embed_dim` at the same site (Q1). Same notebook cells,
  bootstrap-CI band + Wilcoxon.
- **Conservation control (Q2):** assert `Σ_i R(f_i) ≈ R(probe)` up to recon
  residual, per image.
- **Interpretability (qualitative):** CRP Relevance-Maximization reference
  images per top SAE feature (the `fv` machinery in the walkthrough), to show
  features look monosemantic.

### 4.7 Parallel notebook  `tutorials/vit_crp/sae_walkthrough.ipynb`
Same 8-section skeleton and style as `walkthrough.ipynb`, SAE added:
1. Setup · 2. Configuration (probe + composite + **SAE checkpoint**) ·
3. Hookable layers (**show the spliced `sae.features` layer**) ·
4. Load dataset + focal image · 5. **Train/load SAE + reconstruction sanity
(recon MSE, L0)** · 6. Reference samples per SAE feature (Relevance-Maximization
on the SAE basis vs `embdim_at_proj`) · 7. Conditional propagation cascade on
top SAE features · 8. Notes + faithfulness (SAE vs embed_dim AOPC).

## 5. Milestones (smoke → full)
1. **Smoke:** dsprites, block 11, α=8 L1 SAE, 1 epoch → recon MSE sane, `f`
   sparse, splice preserves accuracy within ε, conservation holds on 1 image.
2. **Single-block faithfulness:** dsprites block 11, SAE vs embed_dim AOPC.
3. **All blocks / 3 datasets** if Q1 looks positive.
4. Notebook + figures (png+pdf, `figures/sae_crp/...`), then paper only **if
   explicitly asked** (do not touch the paper unsolicited).

## 6. Risks / open decisions
- **Probe site** `proj_drop` vs residual stream — *team decision*.
- **Splice faithfulness:** if recon error is large, CRP explains a different
  function than the model. Mitigation: report recon FVU; only proceed if small.
- **LRP rule for the SAE linears:** ε vs γ — settle empirically (conservation +
  heatmap quality), add as an `lrp_configs` knob.
- **Compute:** SAE training is cheap (one linear layer, GPU-minutes/block);
  flipping on `m = 8·384 = 3072` features is ~8× the embed_dim cost per block —
  start single-block.
- **Novelty risk:** must frame vs CaFE (2509.00749) precisely (direction +
  conservation), or a reviewer rejects as incremental.

## 7. References
- Achtibat et al. CRP — arXiv 2206.03208 (Nat. Mach. Intell. 2023)
- Elhage et al. Toy Models of Superposition — arXiv 2209.10652 (2022)
- Cunningham et al. SAEs Find Interpretable Features — arXiv 2309.08600 (ICLR 2024)
- Bricken et al. Towards Monosemanticity — transformer-circuits.pub 2023 (no arXiv)
- Templeton et al. Scaling Monosemanticity — transformer-circuits.pub 2024 (no arXiv)
- Gao et al. TopK SAEs — arXiv 2406.04093 (2024)
- Rajamanoharan et al. Gated / JumpReLU — arXiv 2404.16014 / 2407.14435 (2024)
- Marks et al. Sparse Feature Circuits — arXiv 2403.19647 (ICLR 2025)
- Han, Kim, Kwak. CaFE (LRP feature→input on SAE, ViT) — arXiv 2509.00749 (2025)
- Stevens et al. saev — arXiv 2502.06755 · Joseph et al. — arXiv 2504.08729 · PatchSAE — arXiv 2412.05276
