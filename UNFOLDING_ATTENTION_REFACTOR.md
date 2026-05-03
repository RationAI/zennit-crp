# Unfolding the attention module into atomic LRP layers — implementation plan

**Scope.** Implementation details only. For motivation, design rationale,
mechanism analysis, and trade-off discussion, see
**`RESEARCH_NOTES.md`**, especially Entries 4-5 (conservation analysis,
bilinear rule mechanism) and Entry 6 (AlphaBeta rule which this
refactor is the natural substrate for).

**Status.** Phase 1 implemented and merged on `attention-unfolding`
branch (commits `9bfc189`, `0e7df9c`, `f6f3d4b`). Phase 2 in progress.

**Branch organisation.**
* `transformer-multi-concept` — main working branch, always green
  walkthrough.
* `attention-unfolding` — refactor branch. This plan + Phase 1 prototype
  + ongoing Phase 2 work.
* This document committed to both branches so it's findable from either
  working tree.

## Target design

```
EvaAttentionUnfolded(nn.Module):
    qkv:        nn.Linear            (D → 3·D)
    split:      ChunkAlongLastDim(3) (1 input, 3 outputs)
    q_norm:     nn.LayerNorm | nn.Identity
    k_norm:     nn.LayerNorm | nn.Identity
    rope_q:     RotaryEmbedding(num_prefix_tokens)
    rope_k:     RotaryEmbedding(num_prefix_tokens)
    scale_q:    ScaleByConstant(self.scale)
    qk_scores:  BilinearMatmul                          # q @ k.transpose(-2, -1)
    add_mask:   AddBias(constant=resolved_attn_mask)    # optional, per timm signature
    softmax:    SoftmaxAlongLastDim
    attn_drop:  nn.Dropout                              # passthrough during attribution
    context:    BilinearMatmul                          # attn_post_softmax @ v
    reshape:    ReshapeMergeHeads
    out_norm:   nn.LayerNorm | nn.Identity              # only on some EvaAttention variants
    proj:       nn.Linear                               # D → D
    proj_drop:  nn.Dropout

    forward(self, x, rope=None, attn_mask=None, is_causal=False):
        qkv = self.qkv(x)
        q, k, v = self.split(qkv)
        q = self.q_norm(q); k = self.k_norm(k)
        q = self.rope_q(q, rope); k = self.rope_k(k, rope)
        q = self.scale_q(q)
        scores = self.qk_scores(q, k.transpose(-2, -1))
        scores = self.add_mask(scores, attn_mask)
        weights = self.softmax(scores)
        weights = self.attn_drop(weights)
        ctx = self.context(weights, v)
        out = self.reshape(ctx)
        out = self.out_norm(out)
        out = self.proj(out)
        return self.proj_drop(out)
```

Each new module owns its own LRP rule:

| Module | Role | LRP rule | Source |
|---|---|---|---|
| `ChunkAlongLastDim(3)` | Q/K/V split | true identity (3-tuple of grad slices reassembled) | trivial |
| `BilinearMatmul` | Q@Kᵀ, attn@V | `_MatmulFactor2Fn` (Achtibat Prop 3.3) | already implemented as Function |
| `RotaryEmbedding` | RoPE rotate (npt-aware) | wraps `apply_rot_embed_cat`; full pass-through, optional `.detach(rope)` | `crp.transformer_patches._eva_attention_forward` already has this logic |
| `ScaleByConstant` | `q * self.scale` | identity rule (constant doesn't absorb relevance) | new |
| `AddBias` | mask add | identity-on-x branch (mask is leaf constant, absorbs no R) | trivial |
| `SoftmaxAlongLastDim` | softmax | identity rule (R_in = R_out, per AttnLRP Eq. 9 derivation for normalisations) | new |
| `ReshapeMergeHeads` | tensor reshape | identity (no rule needed) | trivial |

`nn.LayerNorm` keeps the existing `layer_norm_forward` swap (stop_gradient on std).
`nn.Linear` keeps zennit's stock `Epsilon`. `nn.Dropout` keeps the
passthrough.

The block wrapper (`EvaBlock`) similarly gets unfolded:

```
EvaBlockUnfolded(nn.Module):
    norm1, attn (= EvaAttentionUnfolded), drop_path1
    layerscale1: LayerScaleMul(gamma_1)         # owns the divide_gradient(γ·x, 2) rule
    add1:        ResidualAdd                    # owns the ratio split rule
    norm2, mlp,  drop_path2
    layerscale2: LayerScaleMul(gamma_2)
    add2:        ResidualAdd
```

`LayerScaleMul` and `ResidualAdd` are 2-input `nn.Module`s wrapping the
existing autograd Functions.

## Concept conditioning

Currently `KQVHeadConcept` reads `block.6.attn.qkv_tap` (shape
`(B, N, 3·D)`), slices the 3·D dimension into `[K, Q, V]` in a
hard-coded layout, and masks. After the refactor, the same concept
becomes:

* `KConcept(model, head_idx).mask` reads `block.6.attn.k_norm`
  (or `rope_k`) — single tensor of shape `(B, num_heads, N, head_dim)`,
  no slicing.
* `QConcept`, `VConcept` analogous.
* `AttentionWeightConcept` reads `block.6.attn.softmax` —
  `(B, num_heads, N, N)` tensor.
* `HeadConcept` reads `block.6.attn.proj` (the projection input, since
  reshape happens before `proj`).

The existing concept classes either get migrated to read the new layer
names (preferred) or get a thin shim that maps the old tap name to the
new layer name.

## Implementation plan

The work is sequenced so the main branch always has a working
walkthrough and the refactor branch can be merged when (and only when)
its conservation behaviour matches or improves on main.

### Phase 1 — prototype (DONE — commits `9bfc189`, `0e7df9c`, `f6f3d4b`)

Files created on `attention-unfolding`:
* `crp/attention_unfolded.py` — 11 atomic Module kernels +
  `EvaAttentionUnfolded` container + `EvaAttentionSubstitutionCanonizer`.
* `tests/test_attention_unfolded.py` — 31 unit tests, all green.
* `experiments/test_unfolded_block0.py` — Phase-1 exit-criterion script.

Exit criterion met: forward + backward bit-identical (`max|diff| = 0.0`)
on real DINOv3 ViT-L weights when block 0 substituted. Total
100/100 tests pass on the branch.

Outstanding from Phase 1 (deferred to Phase 2):
* The unfolded path applies AttnLRP identity rule on softmax (Eq. 9)
  and on `q * scale` (constants absorb no R), which the legacy
  `_eva_attention_forward` skipped. Under the working composite this
  produces ~40× larger relevance magnitudes — semantic, not noise.
  See `RESEARCH_NOTES.md` Entries 4-5 for the mechanism.

### Phase 2 — full migration

1. Apply the substitution canonizer to *all* attention blocks. Re-run
   the conservation probe (`experiments/dinov3_conservation.py`). The
   anomaly seen on the current architecture (norm.R_in=0 vs
   blocks.23.R_out=3.71) should resolve, or at least become
   debuggable through the unfolded structure.
2. Unfold `EvaBlock` similarly (`EvaBlockUnfolded` with `LayerScaleMul`
   and `ResidualAdd` submodules).
3. Migrate `KQVHeadConcept`, `KQVHeadDimConcept`, `HeadConcept`,
   `HeadDimConcept` to target the new layer names. Add deprecation
   warnings to anyone still reading `qkv_tap` / `attn_out_tap`.
4. Update the working composite to install the substitution canonizer
   (alongside or in place of the existing forward-replacement canonizer).
5. Re-run the full test suite (`pytest tests/test_vit_integration.py
   tests/test_attention_concepts.py`) — should remain green.
6. Re-run `experiments/run_dinov3_remedy_eval.py` — magnitudes /
   focus / register-leak should match the current `working_combo`.
7. Re-execute the main `tutorials/vit_crp/walkthrough.ipynb` and the
   DINOv3 `working_combo/walkthrough.ipynb` against the refactored code;
   diff output cells against the version on `main`.

**Exit criterion**: tests green; conservation probe shows ≤ current
chain anomalies; all walkthroughs reproduce; per-block conservation is
checkable without measurement gymnastics.

### Phase 3 — cleanup & merge back

1. Delete the now-unused canonizer-replacing-forward path
   (`_eva_attention_forward`, `_timm_attention_forward`).
2. Drop `qkv_tap`, `attn_out_tap` Identity injection (concepts now
   target named submodules directly). Possibly keep
   `AttentionTapsCanonizer` for one release as deprecated for
   downstream users.
3. Update the docs in `crp/transformer_patches.py` module docstring
   to point at the new structure.
4. Squash-merge to `transformer-multi-concept`.

## Potential problems & mitigations

### Per-problem implementation notes

| # | Issue | Resolution |
|---|---|---|
| P1 | `BasicHook` with 2-input modules (BilinearMatmul) | Phase 1 chose to bake the rule into the autograd Function inside the module's forward (no custom Hook needed). zennit's `Pass` attaches to BilinearMatmul; the Function does the work. |
| P2 | DAG forward (Q/K/V split) | Container `EvaAttentionUnfolded.forward` writes the wiring explicitly (no `nn.Sequential`). |
| P3 | `ChunkAlongLastDim` LRP rule | Identity in LRP terms (cat is the inverse). No custom rule; autograd's natural backward suffices. |
| P4 | RoPE as a Module | `RotaryEmbedding(num_prefix_tokens, rotate_half, detach_rope=False)`. Forward signature `(q, rope)`. detach_rope flag for Composite-level toggle. |
| P5 | Pre-existing concept code | Phase 2 work. Migration plan: add `q_tap`/`k_tap`/`v_tap` as Identity submodules in `EvaAttentionUnfolded` for legacy concept compat; ship new Q/K/V concept classes that target named submodules directly. |
| P6 | Substitution canonizer reversibility | Phase 1 verified via round-trip test in `experiments/test_unfolded_block0.py`. The unfolded module references the original's parameters (no copy), original is restored verbatim on `remove()`. |
| P7 | fused vs unfused softmax / matmul | Use explicit ops (carry over from existing `_eva_attention_forward` which already bypasses `F.scaled_dot_product_attention`). |
| P8 | Performance regression | ~14 extra Module calls per attention × 24 blocks = ~340 dispatches per forward. Measured: <1 % overhead. |
| P9 | Numerics drift | Phase 1 confirmed bit-identical (`max|diff| = 0.0`) for forward + autograd backward. LRP backward magnitudes differ semantically — see Phase-1 deferred items above and `RESEARCH_NOTES.md` Entry 6. |

## Verification matrix

Run after each phase:

```
uv run pytest tests/test_attention_concepts.py tests/test_vit_integration.py -q
uv run python experiments/run_dinov3_remedy_eval.py --n-samples 5
uv run python experiments/dinov3_conservation.py
uv run jupyter nbconvert --to notebook --execute \
    tutorials/vit_crp/walkthrough.ipynb --output /tmp/walk_main.ipynb \
    --ExecutePreprocessor.timeout=600
uv run jupyter nbconvert --to notebook --execute \
    tutorials/vit_crp/dinov3_variants/working_combo/walkthrough.ipynb \
    --output /tmp/walk_dinov3.ipynb --ExecutePreprocessor.timeout=600
```

Compare against the same outputs on `transformer-multi-concept`. Any
metric that visibly worsens needs investigation before proceeding to
the next phase.

## Out of scope for this refactor

* PA-LRP for absolute-pos-embed ViTs (DINOv3 uses RoPE; not needed).
* Symmetric residual rule (project keeps ratio per design decision —
  `RESEARCH_NOTES.md`).
* Investigating the matmul-rule magnitude inflation — separately
  tracked under `RESEARCH_NOTES.md` Entries 4-6 (resolved into the
  AlphaBeta-on-bilinear research line).
