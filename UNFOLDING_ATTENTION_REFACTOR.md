# Unfolding the attention module into atomic LRP layers

**Status:** plan, not yet implemented. Implementation lives on a separate
branch (`attention-unfolding`); this document is on `transformer-multi-concept`
so it's discoverable from the main working tree.

## Why

The current AttnLRP implementation in `crp/transformer_patches.py`
*replaces the `Attention.forward` method* with a single big function
that strings together autograd `Function`s for the LRP rule kernels
(`_MatmulFactor2Fn`, `_IdentityRuleFn`, `_DivideGradientFn`,
`_ResidualRatioFn`). The graph-level structure of attention is
encoded inside that one function. This has worked (see
`tutorials/vit_crp/dinov3_variants/working_combo/` for a working
DINOv3 recipe), but it has several persistent costs that are blocking
the next round of work:

1. **Conservation checking is hard.** `register_full_backward_hook` on
   the wrapping `Attention` module reports relevance at the *outer*
   boundary only. Internal nodes (after Q-projection, after softmax,
   after attn·V) are reachable only by attaching tensor-level autograd
   hooks at the right point inside the swapped forward, which is
   error-prone and noisy. We hit a concrete instance of this:
   `experiments/dinov3_conservation.py` reports a chain inconsistency
   between `model.norm.R_in` (=0, per LayerNorm-with-stop-gradient(std)
   sum-zero derivation) and `blocks.23.R_out` (=initial logit) that
   shouldn't exist if the chain were fully transparent to my probe.
   Step 1 of post-plan work is to debug that anomaly on the current
   architecture; the refactor would make such diagnostics trivial
   going forward.
2. **Concept conditioning is restricted to two named taps**
   (`qkv_tap`, `attn_out_tap`) installed by `AttentionTapsCanonizer`.
   Targeting Q-only, post-softmax-attention-weights, or
   per-head-after-rotation requires either adding more taps (more
   identity modules to install + more shape conventions to maintain)
   or doing post-hoc slicing inside the concept class
   (`KQVHeadConcept.mask` does this for the 3-cat'd qkv tensor; easy
   to get index-off-by-one wrong).
3. **Visualisation tools** (`torchviz`, `hiddenlayer`, the standard
   PyTorch `print(model)` repr) walk the module tree, not the autograd
   graph. The current single-forward approach makes the attention
   appear as a black box.
4. **Rule swapping per layer / per attention head** is currently
   parameter-driven inside the swapped forward, requiring conditional
   branches. Swapping `BilinearMatmul.forward = ...` would be a
   one-liner per instance.

The refactor exchanges a self-contained-but-opaque forward function
for a subgraph of named `nn.Module`s, each owning one LRP rule. The
zennit hook framework attaches to each module by type or name without
any tap injection.

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

### Phase 1 — prototype (verify the design works on one block)

1. Create new file `crp/attention_unfolded.py` — all new module
   classes (`BilinearMatmul`, `SoftmaxAlongLastDim`, `RotaryEmbedding`,
   etc.) + `EvaAttentionUnfolded`. Migrate the autograd Function
   kernels from `crp/transformer_patches.py` (those stay; the new
   modules just wrap them).
2. Verify zennit's `BasicHook` handles 2-input modules. `BilinearMatmul`
   takes two tensors; need to confirm `input_modifiers`,
   `gradient_mapper`, etc. work. If not, write a small custom Hook
   that does. Likely just a matter of `inputs[0], inputs[1]` instead of
   `inputs[0]`.
3. New canonizer `EvaAttentionSubstitutionCanonizer` that *replaces*
   `EvaAttention` instances with `EvaAttentionUnfolded` instances at
   `apply()` time, restoring the original on `remove()`. Ports the
   weights from the original Linear modules.
4. Build a one-off test: pretrained DINOv3 + the unfolded attention
   on block 0 only (rest of model untouched). Verify forward output
   matches the original's forward output to within fp32 noise. Verify
   backward gradient (no LRP rule) matches autograd's natural backward.
5. Add LRP rules to the unfolded modules; verify backward under the
   working composite gives the same relevance as the current
   single-forward implementation, on the same image.

**Exit criterion**: forward + backward bit-identical (or within
~1e-5) to current implementation on `block 0 unfolded`.

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

### Problem 1 — `BasicHook` with 2-input modules

zennit's `BasicHook.backward` expects `grad_input` of length 1 for
single-input modules. Bilinear modules have 2 inputs. Two paths:

* **Use `BasicHook` directly** with `input_modifiers=[lambda inp: inp]`
  applied to each input; `reducer` returning a tuple
  `(R_a, R_b)`. Need to verify the framework propagates a tuple
  return correctly (`return tuple(r if shape_matches else None ...)`
  is what the LRP-ε hook does).
* **Custom Hook**: subclass `BasicHook`, override `backward` to handle
  2 inputs explicitly. ~30 lines, mirrors the LRP-ε hook with `inputs`
  unpacked.

Mitigation: try option 1 first; fall back to option 2 if zennit's
machinery proves rigid. Either works.

### Problem 2 — DAG forward (Q, K, V split)

`ChunkAlongLastDim(3)` returns 3 tensors. Subsequent modules consume
them in different ways. There's no `nn.Sequential` for this; the
container `EvaAttentionUnfolded.forward` writes the wiring explicitly.

That's already standard PyTorch — no real problem. Just to set
expectations: the container Module's `forward` method is the wiring,
not zennit's job.

### Problem 3 — `ChunkAlongLastDim` LRP rule

The QKV split is just a slice: `qkv[:, :, :D]`, `qkv[:, :, D:2D]`,
`qkv[:, :, 2D:3D]`. Backward is concatenation. This is identity in
LRP terms — relevance from each slice goes back to its original
position in the cat'd tensor. No rule needed; autograd's natural
backward does the right thing.

### Problem 4 — RoPE as a Module

`RotaryEmbedding.forward(self, q, rope)` takes two inputs but only one
needs LRP attribution (`q`). The `rope` argument is a positional
constant. With the usual `Pass` rule the natural backward works; with
the `rope_detach` flag (currently a parameterised composite) we just
detach `rope` inside the module's forward. No new design.

### Problem 5 — Pre-existing concept code

`KQVHeadConcept.mask` is hard-coded to read `qkv_tap` and slice
`(B, N, 3·D)` into K/Q/V. Refactoring this is a breaking change for
users on this branch. Mitigation: ship a shim
`KQVHeadConcept._legacy_mask` for one release, and emit a
`DeprecationWarning` pointing at the new per-K/Q/V concepts. See
"Concept conditioning" above.

### Problem 6 — `EvaAttentionSubstitutionCanonizer` reversibility

`Canonizer.remove()` must restore the original `EvaAttention` instance
verbatim. The substitution swaps `EvaAttention` → `EvaAttentionUnfolded`
on `apply()`, must swap back on `remove()`. The risk: weights got
*shared* between the two (we'd port references, not copies). On
`remove()`, the original module's parameters are still attached;
nothing is lost. But we MUST verify this with a round-trip test —
apply, run forward, remove, re-apply with a different composite, run
forward, compare.

### Problem 7 — fused vs unfused softmax / matmul

timm's `EvaAttention.forward` uses `F.scaled_dot_product_attention` if
`fused_attn=True`. Our existing replacement bypasses this for the
explicit ops. The unfolded version should also use explicit ops
(otherwise the bilinear and softmax aren't separately attribute-able).
Already handled in current code; carry over.

### Problem 8 — Performance regression

Module dispatch is faster than people fear, but ~14 extra Modules per
attention block × 24 blocks = ~340 extra Module calls per forward.
Probably <1 % overhead. Will verify with `time.perf_counter()` on the
walkthrough.

### Problem 9 — Numerics drift

Even with mathematically equivalent rules, fp32 reordering can shift
results by O(1e-5). If the working_combo notebook produces visibly
different heatmaps on the new branch, verify each phase-1 step
independently to localise the drift. The autograd graph reorganisation
is the most likely source.

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

* Investigating *why* the `2y+ε` matmul rule produces such large
  sign-cancelling magnitudes. Worth a separate paper if we can pin it
  down — see "max|R| 200 cancelled by -200" finding in
  `tutorials/vit_crp/dinov3_variants/working_combo/FINDINGS.md`.
* PA-LRP for the patch-embed Conv2d. Currently DINOv3 uses RoPE so PA-LRP
  is only relevant for older (absolute-pos-embed) ViTs.
* Symmetric residual rule (we keep ratio per project decision).

## Branch organisation

* `transformer-multi-concept` — main working branch. Always has a
  working `tutorials/vit_crp/walkthrough.ipynb` (currently
  vit_base_patch16_224 + ε-LRP) and a working
  `tutorials/vit_crp/dinov3_variants/working_combo/walkthrough.ipynb`
  (DINOv3 ViT-L + matmul + layerscale + ratio).
* `attention-unfolding` — refactor branch. Phase 1 lives here as a
  prototype; phase 2 expands; phase 3 prepares the merge.
* This document (`UNFOLDING_ATTENTION_REFACTOR.md`) is committed to
  both branches so it's findable from either working tree.
