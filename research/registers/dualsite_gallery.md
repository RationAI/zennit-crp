# Dual-site gallery extension (reviewer: site mismatch)

Reviewer (Adam) found a **site mismatch** in the CRP gallery: concept
visualisations existed only for site `proj_drop` (attention output projection,
BEFORE the residual add), while the register-outlier activation maps were
recorded at the `blocks[i]` output (residual stream, AFTER the adds). Relevance
heatmaps at proj_drop do not show the register artifacts the residual-stream
activations show. Resolution ordered: present BOTH sites properly for the
baseline profile (`cp_lrp_baseline`, concept `embed_dim`).

## 1. Gallery entries at site `residual`

Chunked runners (GPU-lock protocol, per-chunk acquire/release; chunk 0 of each
= FV-index build for all 12 residual layers in one dataset pass, then 2–3-block
render chunks with the standard spec `--n 5 --n-ref 6 --plot heat_rf`,
class-conditional ranking):

```
bash data/logs/run_gallery_residual_funny_birds.sh   # log: data/logs/gallery_residual_funny_birds.log
bash data/logs/run_gallery_residual_imagenet.sh      # log: data/logs/gallery_residual_imagenet.log
```

which per chunk invoke the existing CLI, e.g.

```
python -m experiments.crp_gallery compute --base vit_small --dataset funny_birds \
  --config cp_lrp_baseline --site residual --blocks .. --concept embed_dim \
  --n 5 --n-ref 6 --plot heat_rf --device cuda
python -m experiments.crp_gallery compute --base vit_base --dataset imagenet \
  ... --site residual ... --classes 0..9   # same ranking classes as the proj_drop job
```

* FV index for `residual` layer names (`backbone.blocks.{b}`) built fresh on
  scratch and mirrored to the persistent root (`experiments/crp_gallery.py`
  cache discipline); funny_birds = 29330 train-clean images, imagenet = 10k val
  subset (n_per_class=10) — the long build.
* `record_job` runs per chunk; the residual jobs.jsonl line is canonicalized to
  `blocks 0..11, n=5` at the end (the imagenet runner does it itself; the
  funny_birds line was fixed by hand).

Status: see final section.

## 2. Dual-site activation maps (norm maps)

New generator: `experiments/scripts/registers_actmaps_dualsite.py`
(companion to `registers_position_freq.py`, whose `actmap_*` figures were
single-site residual only).

```
python -m experiments.scripts.registers_actmaps_dualsite --dataset both --device cpu
```

* One forward per model over the 6 canonical gallery samples
  (FunnyBirds `c0_0, c1_603, c2_1222, c3_1810, c4_2402, c5_2988`; ImageNet
  `lizard 563, cheeseburger 471, goldfish 3232, sports_car 598, daisy 132,
  golden_retriever 358` — `crp_gallery.pick_samples`, ds_indices verified
  against `data/results/registers/gallery_samples_vit_base_imagenet.npz`).
* Records token L2 norms at BOTH sites per block: `residual` = `blocks[i]`
  output, `proj_drop` = `blocks[i].attn.proj_drop` output.
* Outlier flags **per sample** at each site/block: norm > mean + 4·sd over that
  sample's own 196 patch tokens (reviewer's per-sample criterion; single-block
  flags, CLS excluded). Magenta borders per site.
* Sanity crosscheck: residual-site norms match the previously stored ones
  (`step1b_position_freq_funny_birds.npz::gallery_norms`,
  `gallery_samples_vit_base_imagenet.npz::norms`) — **OK for both models**.

Outputs:

* `figures/registers/actmaps_dualsite/actmap2_<key>.{png,pdf}` (12 samples)
* raw arrays: `data/results/registers/actmaps_dualsite_{funny_birds,imagenet}.npz`
* deployed as the gallery sample norm-maps (manifest `normmap` field):
  `webapp/crp_gallery/figures/vit_small_funny_birds/_normmaps/<key>.png`
  (OVERWRITTEN with the dual-site version) and
  `webapp/crp_gallery/figures/vit_base_imagenet/_normmaps/<key>.png` (new).

Observation (e.g. `actmap2_c0_0`): the residual site shows the persistent
top-row/corner register outliers from block ~3 onward; at proj_drop the same
blocks show object-shaped norm structure and (almost) no flagged registers —
exactly the mismatch the reviewer pointed at.

## 3. Verification

* `python -m experiments.crp_gallery manifest` rebuilt at the end; checked that
  both models list instances with `site: residual` layers and non-null
  `normmap` fields for all 6 samples each.
* Web endpoint auth check: `curl -sk -o /dev/null -w '%{http_code}'
  https://claude-bajger.dyn.cloud.e-infra.cz/zennit-crp-gallery/` → expect 401.

## Status

(filled at hand-off — see run logs above for live state)

* funny_birds residual gallery: RUNNING/DONE per log.
* imagenet residual gallery: FV build is the long pole (≤45-min lock holds
  allowed for it); render chunks after.
* dual-site norm maps: DONE (both models), deployed.
