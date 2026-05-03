# DINOv3 LRP remedy diagnostic — `vit_large_patch16_dinov3`

Sweep across 13 remedy composites × 5 correctly-classified Imagenette images. Each cell of the diagnostic records relevance health at every `blocks.{i}.attn.attn_out_tap`. See `crp/transformer_patches.py` for the per-remedy composite classes.

## Summary

| Remedy | finite (heatmap) | median max\|R\| | median focus@10% | median register-leak | first divergence |
|--------|------------------|-----------------:|-----------------:|---------------------:|------------------|
| `baseline_epsilon` | 0/5 | NaN | NaN | 0.000 | blocks.0.attn.attn_out_tap (NaN) |
| `baseline_gamma` | 5/5 | 0.098 | 0.497 | 0.000 | — |
| `matmul_factor_2` | 0/5 | NaN | NaN | 0.000 | blocks.0.attn.attn_out_tap (NaN) |
| `signed_epsilon` | 0/5 | NaN | NaN | 0.000 | blocks.0.attn.attn_out_tap (NaN) |
| `rope_detach` | 0/5 | NaN | NaN | 0.000 | blocks.0.attn.attn_out_tap (NaN) |
| `layerscale_uniform` | 5/5 | 3.26e+14 | 0.775 | 0.000 | blocks.0.attn.attn_out_tap (5.21e+13) |
| `linear_gamma_005` | 5/5 | 4.85e+05 | 0.597 | 0.000 | — |
| `combined_all` | 5/5 | 123.409 | 0.818 | 0.000 | — |
| `layerscale+signed` | 5/5 | 3.26e+14 | 0.775 | 0.000 | blocks.0.attn.attn_out_tap (5.21e+13) |
| `layerscale+rope_detach` | 5/5 | 3.26e+14 | 0.775 | 0.000 | blocks.0.attn.attn_out_tap (5.21e+13) |
| `layerscale+matmul` | 5/5 | 217.979 | 0.809 | 0.000 | — |
| `layerscale+linear_gamma_005` | 5/5 | 9.73e-06 | 0.511 | 0.000 | — |
| `layerscale+ratio_residual_only` | 5/5 | 3.26e+14 | 0.775 | 0.000 | blocks.0.attn.attn_out_tap (5.21e+13) |

**Interpretation.** `finite=N/N` means the heatmap has no NaN/Inf after channel sum. `max|R|` near 0 (≪1) suggests under-flow; `max|R|` ≫ 1e10 is an over-flow regime. `focus@10%` is the share of total |R| concentrated in the top-10% of pixels — uniform = 0.10, well-localised = 0.5–0.9. `register-leak` is the fraction of total |R| living on the 5 cls/register tokens at the deepest recorded tap (Darcet et al. 2023, arXiv:2309.16588 — register tokens absorb high-norm artifacts and attach to no input pixel).

## Per-remedy notes

### `baseline_epsilon`
- 0/5 samples produce a finite heatmap; median max\|R\| = NaN, focus = NaN, register-leak = 0.000.
- First per-layer divergence (max\|R\| > 1e10) at **blocks.0.attn.attn_out_tap (NaN)**.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 8.68e+23 | 6.35e+20 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 1.29e+17 | 1.69e+14 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 4.03e+10 | 3.89e+07 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 1818.701 | 5.674 | 0 |

### `baseline_gamma`
- 5/5 samples produce a finite heatmap; median max\|R\| = 0.098, focus = 0.497, register-leak = 0.000.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 2.487 | 1.60e-04 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 0.961 | 1.13e-04 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 0.151 | 1.10e-04 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 0.199 | 5.67e-05 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 0.189 | 0.001 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 0.054 | 5.05e-04 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 0.016 | 3.11e-04 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.004 | 1.44e-04 | 0 |

### `matmul_factor_2`
- 0/5 samples produce a finite heatmap; median max\|R\| = NaN, focus = NaN, register-leak = 0.000.
- First per-layer divergence (max\|R\| > 1e10) at **blocks.0.attn.attn_out_tap (NaN)**.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 1.35e+20 | 8.48e+16 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 1.47e+14 | 2.93e+11 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 8.65e+08 | 1.39e+06 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 1818.701 | 5.674 | 0 |

### `signed_epsilon`
- 0/5 samples produce a finite heatmap; median max\|R\| = NaN, focus = NaN, register-leak = 0.000.
- First per-layer divergence (max\|R\| > 1e10) at **blocks.0.attn.attn_out_tap (NaN)**.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 8.68e+23 | 6.35e+20 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 1.29e+17 | 1.69e+14 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 4.03e+10 | 3.89e+07 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 1818.701 | 5.674 | 0 |

### `rope_detach`
- 0/5 samples produce a finite heatmap; median max\|R\| = NaN, focus = NaN, register-leak = 0.000.
- First per-layer divergence (max\|R\| > 1e10) at **blocks.0.attn.attn_out_tap (NaN)**.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | NaN | NaN | 267264 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 8.68e+23 | 6.35e+20 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 1.29e+17 | 1.69e+14 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 4.03e+10 | 3.89e+07 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 1818.701 | 5.674 | 0 |

### `layerscale_uniform`
- 5/5 samples produce a finite heatmap; median max\|R\| = 3.26e+14, focus = 0.775, register-leak = 0.000.
- First per-layer divergence (max\|R\| > 1e10) at **blocks.0.attn.attn_out_tap (5.21e+13)**.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 5.21e+13 | 6.39e+09 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 2.40e+13 | 3.37e+09 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 5.00e+12 | 1.59e+09 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 1.24e+12 | 5.13e+08 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 6.00e+04 | 97.271 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 6715.039 | 2.982 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 16.135 | 0.054 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.031 | 3.65e-04 | 0 |

### `linear_gamma_005`
- 5/5 samples produce a finite heatmap; median max\|R\| = 4.85e+05, focus = 0.597, register-leak = 0.000.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 3.37e+05 | 188.936 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 1.29e+05 | 94.587 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 8.33e+04 | 86.576 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 1.42e+05 | 64.471 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 290.352 | 1.651 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 11.114 | 0.176 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 1.687 | 0.019 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.026 | 0.001 | 0 |

### `combined_all`
- 5/5 samples produce a finite heatmap; median max\|R\| = 123.409, focus = 0.818, register-leak = 0.000.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 43.688 | 0.004 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 40.292 | 0.004 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 2.041 | 9.76e-04 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 1.919 | 3.88e-04 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 11.541 | 0.014 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 1.153 | 0.004 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 0.394 | 0.002 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.031 | 3.65e-04 | 0 |

### `layerscale+signed`
- 5/5 samples produce a finite heatmap; median max\|R\| = 3.26e+14, focus = 0.775, register-leak = 0.000.
- First per-layer divergence (max\|R\| > 1e10) at **blocks.0.attn.attn_out_tap (5.21e+13)**.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 5.21e+13 | 6.39e+09 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 2.40e+13 | 3.37e+09 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 5.00e+12 | 1.59e+09 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 1.24e+12 | 5.13e+08 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 6.00e+04 | 97.271 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 6715.039 | 2.982 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 16.135 | 0.054 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.031 | 3.65e-04 | 0 |

### `layerscale+rope_detach`
- 5/5 samples produce a finite heatmap; median max\|R\| = 3.26e+14, focus = 0.775, register-leak = 0.000.
- First per-layer divergence (max\|R\| > 1e10) at **blocks.0.attn.attn_out_tap (5.21e+13)**.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 5.21e+13 | 6.39e+09 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 2.40e+13 | 3.37e+09 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 5.00e+12 | 1.59e+09 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 1.24e+12 | 5.13e+08 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 6.00e+04 | 97.271 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 6715.039 | 2.982 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 16.135 | 0.054 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.031 | 3.65e-04 | 0 |

### `layerscale+matmul`
- 5/5 samples produce a finite heatmap; median max\|R\| = 217.979, focus = 0.809, register-leak = 0.000.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 43.642 | 0.004 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 46.733 | 0.005 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 2.521 | 0.001 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 1.903 | 3.97e-04 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 11.874 | 0.016 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 2.028 | 0.004 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 0.828 | 0.002 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.031 | 3.65e-04 | 0 |

### `layerscale+linear_gamma_005`
- 5/5 samples produce a finite heatmap; median max\|R\| = 9.73e-06, focus = 0.511, register-leak = 0.000.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 0.003 | 1.05e-07 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 1.33e-04 | 2.98e-08 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 1.46e-05 | 7.36e-09 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 8.67e-07 | 4.99e-10 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 0.003 | 1.93e-05 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 0.002 | 2.17e-05 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 0.004 | 5.05e-05 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.002 | 7.05e-05 | 0 |

### `layerscale+ratio_residual_only`
- 5/5 samples produce a finite heatmap; median max\|R\| = 3.26e+14, focus = 0.775, register-leak = 0.000.
- First per-layer divergence (max\|R\| > 1e10) at **blocks.0.attn.attn_out_tap (5.21e+13)**.

Per-layer max\|R\| trajectory (sample 0, in backward order — first row is the deepest tap = first to receive relevance):

| layer | shape | max\|R\| | mean\|R\| | nan |
|-------|-------|---------:|----------:|----:|
| `blocks.0.attn.attn_out_tap` | (1, 261, 1024) | 5.21e+13 | 6.39e+09 | 0 |
| `blocks.1.attn.attn_out_tap` | (1, 261, 1024) | 2.40e+13 | 3.37e+09 | 0 |
| `blocks.2.attn.attn_out_tap` | (1, 261, 1024) | 5.00e+12 | 1.59e+09 | 0 |
| `blocks.3.attn.attn_out_tap` | (1, 261, 1024) | 1.24e+12 | 5.13e+08 | 0 |
| `blocks.20.attn.attn_out_tap` | (1, 261, 1024) | 6.00e+04 | 97.271 | 0 |
| `blocks.21.attn.attn_out_tap` | (1, 261, 1024) | 6715.039 | 2.982 | 0 |
| `blocks.22.attn.attn_out_tap` | (1, 261, 1024) | 16.135 | 0.054 | 0 |
| `blocks.23.attn.attn_out_tap` | (1, 261, 1024) | 0.031 | 3.65e-04 | 0 |

---

Raw per-layer dumps in `diagnostic_raw.json`. Each remedy has its own subfolder under this directory; subfolders for non-working remedies contain a `FINDINGS.md` only, working ones additionally contain a `walkthrough.ipynb` demo.
