# Plan: HEAD-concept experiments (occlude/perturb whole attention heads)

Replicate the recent embed-dim concept-detector work for the **head** concept:
DAPC + MoRF/LeRF curves, heuristic order search vs LRP rankings, and CRP-gallery
representatives — across both models/datasets, cp_lrp + chefer composites, with
maximum code reuse. Zero-occlusion only (no mean/sign-flip; concept_flipping.py
out of scope this round).

## Locked decisions (confirmed)
- **Sites:** `value` (v_lrp_probe) + `qk` (q_lrp_probe) **FIRST** — the clean per-head
  sites (pre-`_to_heads`, a head_dim slice is exactly that head). cp_lrp excludes `qk`
  (StopGradient on Q/K), so cp_lrp runs on `value` only.
  `residual` + `proj_drop` **DEFERRED to last** — user is rethinking how to approach them.
  *Caveat driving the deferral:* those two sit after cross-head mixing (block output /
  post-projection), so a head_dim slice is the head's nominal channel group, not an
  isolated head — head numbers there need a different interpretation.
- **Blocks:** 6–11 (both measurement and gallery).
- **Gallery ref modes:** composites (cp_lrp, chefer) → all 4 (relsum, relmax, actsum, actmax);
  heuristic-derived reps → actsum + actmax only.
- **fv_class:** original only (aligned is code-blocked — no g-convention adapter).
- **Concept:** `head`. D = num_heads: **12** (ViT-B/16) / **6** (ViT-S/16); head_dim = 64 both.

## Models / datasets (2)
- **M2** — `vit_base` + `imagenet` (12 blocks, embed 768, 12 heads). tag `m2_vitb_in`.
- **M1** — `vit_small` + `funny_birds` (12 blocks, embed 384, 6 heads). tag `m1_vits_fb`.

---

## Phase 0 — Code (reuse-first)

Existing `HeadConcept` (crp/concepts.py:147), gallery `--concept head`, and all 4 ref
modes already exist. Minimal wiring:

1. **`concept_detector_bench.py` + `concept_detector_optimal.py`:** add `--concept {embed_dim,head}`.
   - `D = num_heads` (search width) when `head`; keep occlusion via `ZeroChannelsHook`,
     widened: `keep (R, num_heads) → repeat_interleave(head_dim, -1) → (R, embed_dim)` before
     the `out * keep` multiply (matches HeadConcept's contiguous-slice mask).
   - ranking `rank_psi`: `HeadConcept(num_heads)` instead of `EmbeddingDimConcept`.
   - npz naming: `cdet_dapc_<key>__head.npz` + `cdet_dapc_<key>__head_optimal.npz`.
   - `occlusion_check`: assert last-dim == embed_dim; search dim == num_heads.
2. **Gallery:** register `chefer_lrp` as a gallery config (currently never used as one);
   extend `gallery_optimal_actmax.py` to read `__head_optimal.npz` and drive head detectors.
3. **Manifest/ref fixes** (shared with Phase 3): `render_entry` must stamp `meta["ref"]`
   so aggregate reps are filed under the correct bucket (not defaulted to relsum);
   `render_local_entry` must not silently drop max-refs; `record_job` dedup key must
   include `blocks` for replayability.

---

## Phase 1 — HEAD measurement (DAPC + MoRF/LeRF + heuristic orders)

Output: `cdet_dapc_<key>__head.npz` (+ `__head_optimal.npz`), 2 models. Web: head DAPC
bench page per model (per-layer DAPC table, curves, combined scores).

Methods × sites (blocks 6–11, both models). **value + qk first; residual + proj_drop deferred.**

| method | value | qk | ~~residual~~ | ~~proj_drop~~ |
|---|---|---|---|---|
| cp_lrp (LRP)        | ✓ | — | later | later |
| chefer (LRP)        | ✓ | ✓ | later | later |
| optimal (greedy)    | ✓ | ✓ | later | later |
| optimal_dual        | ✓ | ✓ | later | later |
| random (K=5)        | ✓ | ✓ | later | later |

Runs: 2 bench + 2 optimal (greedy+dual in one). Heuristic-vs-LRP compared via DAPC
(baseline-subtracted mean over comparable sites), as in the embed-dim pipeline.

---

## Phase 2 — HEAD gallery representatives

`crp_gallery --concept head`, fv_class original, blocks 6–11, both models.

| config | sites | refs | per (model×site×block) |
|---|---|---|---|
| cp_lrp_baseline | residual, proj_drop, value (no qk) | relsum, relmax, actsum, actmax | k detectors |
| chefer_lrp (NEW gallery config) | residual, proj_drop, value, qk | relsum, relmax, actsum, actmax | k detectors |
| cp_lrp_baseline_optimal (heuristic) | residual, proj_drop, value, qk | actsum, actmax | top-k from npz |

Instance fan-out (aggregate ref-views, 2 models × 6 blocks):
- cp_lrp: 2×3 sites×6×4 refs = **144**
- chefer: 2×4×6×4 = **192**
- heuristic: 2×4×6×2 = **96**
- **≈ 432 head ref-views total** (+ optional local `_img` views).

---

## Phase 3 — DEFERRED: embed-dim web backfill (LAST, after heads reviewed)

1. **funny_birds heuristic:** expose **actmax** aggregate bucket + regenerate local
   `_img` actmax/relmax/relsum (today only actsum; blocked by `fv_max=None` drop +
   stale `samples:false` jobs). Fixed by the Phase-0 manifest/ref changes.
2. **imagenet heuristic:** add sites proj_drop/value/qk + blocks 1–11 + local views
   (today only residual b0).
3. **Optional:** chefer_lrp + `::negincl` flavour for heuristic/funny_birds.
4. **Blocked:** aligned fv_class (needs g-convention adapter) — separate task.

---

## Order of execution
Phase 0 → Phase 1 on **value + qk** (so head *behavior* is visible first) → review →
Phase 2 gallery on value + qk → then residual + proj_drop (after user's rethink) → Phase 3.
