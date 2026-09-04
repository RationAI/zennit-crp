# Plan: prefix-activation caching for concept_detector_optimal.py

Goal: cut the heuristic's long runtimes by reusing the forward pass up to the
occluded layer. Every `tau()` currently runs a **full** `model(xn.expand(R,…))`
with `ZeroChannelsHook` zeroing channels at `layer_name(site,b)`. The occlusion is
applied *at* that layer, so **input→layer is identical** across all occlusion
rows and all greedy steps — only **layer→output** varies. Cache the prefix once,
run only the suffix per trial. Must be **bit-exact** (same DAPC as the committed
npz).

## Unified block-boundary split (all 4 sites)

Split the ViT only at **block boundaries** (the residual stream). The input to
block `b` is the same regardless of which site inside block `b` we occlude
(residual = block-b output; proj_drop / q_lrp_probe / v_lrp_probe = internal to
block b). So:

1. Clean forward **once per image** → cache the input tensor to each block.
2. For occluding at `(site, b)`: resume from `A_{b}` (= cached input to block b),
   run `blocks[b:] + norm + head` with the `ZeroChannelsHook` registered as now
   (it fires inside block b for internal sites, or on block-b output for
   residual). One split point, works for every site, bit-exact.

## Implementation

### 1. Capture block inputs (once per image, reused across all sites+blocks)
```python
@torch.no_grad()
def capture_block_inputs(model, xn):
    bb = model.backbone
    cache, hs = {}, []
    for i, blk in enumerate(bb.blocks):
        hs.append(blk.register_forward_pre_hook(
            lambda m, args, i=i: cache.__setitem__(i, args[0].detach())))
    model(xn)                 # hook.keep is None here → no occlusion (clean)
    for h in hs: h.remove()
    return cache              # {b: (1, N, D)}
```
`cache[b]` = residual stream entering block b; serves residual/proj_drop/q/v at
block b. ~7 MB/image for ViT-B (12×197×768×4) — keep on device or CPU.

### 2. Suffix runner
```python
def run_blocks_from(model, b, x):        # x: (R, N, D)
    bb = model.backbone
    for i in range(b, len(bb.blocks)):
        x = bb.blocks[i](x)
    return x                              # last-block output, pre-final-norm

def make_head_tail(model):
    """final-block tokens (R,N,D) -> logits (R,C). Branch on head provenance."""
    bb = model.backbone
    if getattr(model, "head_name", "") == "timm_builtin":   # ImagenetViTBase
        return lambda x: bb.forward_head(bb.norm(x))
    return lambda x: model.head(bb.norm(x)[:, 0])           # Probe / DINOv3 canvit
```
Note: timm `forward_features` applies `norm` then `forward_head` does pool+head;
Probe/DINOv3 do `head(features[:,0])`. Verify per model type in the zoo
(FinetunedProbe, ImagenetViTBase, ImagenetDinoV3Base). DINOv3 has register tokens
but cls is still index 0 (`num_prefix_tokens`≥1) — confirm `[:,0]` is the cls.

### 3. Thread a `forward_fn` through tau (replaces `model(xn.expand(...))`)
`tau` should not rebuild the full forward. Give it a closure that runs the suffix
on the cached prefix for a batch of R rows:
```python
def tau(forward_fn, pred, keep_rows, hook, D):
    out = torch.empty(keep_rows.shape[0])
    with torch.no_grad():
        for s in range(0, keep_rows.shape[0], BATCH):
            kb = keep_rows[s:s+BATCH].to(DEVICE)
            hook.keep = kb
            logits = forward_fn(kb.shape[0])          # suffix on cached A_b
            out[s:s+kb.shape[0]] = logits.softmax(-1)[:, pred].cpu()
    hook.keep = None
    return out.numpy()
```
In `run_combo`, build once per (site, b):
```python
A = cache[b]                                          # (1, N, D)
head_tail = make_head_tail(model)
forward_fn = lambda R: head_tail(run_blocks_from(model, b, A.expand(R, -1, -1)))
```
`greedy_order` / `dual_order` / `pair_order` / `prob_curve` / `marginal_deltas`
call `tau(forward_fn, pred, rows, hook, D)` instead of `tau(model, xn, pred, …)`.
(Mechanical signature swap: drop `model, xn`, pass `forward_fn`.)

`.expand(R,-1,-1)` is a view; the residual-site hook's `out*keep` materializes it,
but for internal sites block b runs on the view first — add `.contiguous()` if any
op complains.

### 4. Where the clean forward happens
Capture `cache = capture_block_inputs(model, xn)` **once per image**, before the
site/block loops for that image (in `run_combo`, or hoisted to `action_run`'s
per-image scope so it's shared across sites+blocks of the same image). Sharing
across the whole (site×block) sweep of an image is the big win — one clean
forward, reused by all 48 combos of that image.

## Speedup
Per (image, block b) every tau runs `blocks[b:]+tail` instead of all 12 blocks.
b=11 → ~1 block+tail (~10×); b=0 → ~no gain; plus one clean forward/image. Net
~2–4× over a full sweep, largest on the deep, most-decisive blocks.

## Bit-equality verification (required before commit)
The prefix is captured with `hook.keep=None`, so `A_b` equals the value the full
hooked forward would produce before the occlusion multiply → suffix identical →
DAPC identical (deterministic, eval, no_grad). Prove it:
- Rerun 3–4 stored combos with `--action probe` (one residual + one proj_drop +
  one q + one v, on M1 and M2) and assert the recomputed `dapc`/`order`/`morf`
  match the committed `cdet_dapc_*__optimal.npz` **bit-for-bit** (or ≤1e-6).
- Run the existing quick pytest suite.
Only commit once bit-equal is shown; otherwise the two npz columns would mix two
algorithms.

## Risks / notes
- Canonizer (`VanillaViTAttentionSubstitutionCanonizer`) is applied in `load()`;
  `run_blocks_from` runs the canonized blocks so q/v_lrp_probe hooks fire. Good.
- `make_head_tail` is the only model-specific piece — verify the 3 head paths.
- Keep `cache` tensors' dtype/device consistent with the model (amp? the current
  code runs fp32 eval — match it).
- Non-goal: don't change greedy/dual/pair/tau *semantics*, only where the forward
  starts. `p_current` micro-redundancy (recomputed each greedy step though it
  equals the prior step's winner τ) is a separate, optional tweak — leave it.
