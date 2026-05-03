# AlphaBeta-on-bilinear evaluation — `vit_large_patch16_dinov3`

Sweep across 4 variants of the bilinear matmul rule × 5 class-distinct Imagenette samples. See `RESEARCH_NOTES.md` Entry 6 for derivation, motivation, evaluation rationale, and acceptance criteria.

Substitution: all 24 `EvaAttention` modules replaced with `EvaAttentionUnfolded` configured for the variant's `matmul_rule`. Other LRP rules unchanged (`layerscale_uniform=True`, `residual_lrp='ratio'`, `Epsilon` on Linears, `Pass` on LayerNorm).

## Summary

| variant | finite | median max\|R\| | median focus@10% | median |sum(R)| |
|---|---|---:|---:|---:|
| `baseline_2y_eps` | 5/5 | 2.12e+23 | 0.918 | (see per_layer in raw.json) |
| `alphabeta_1_0` | 5/5 | 895.060 | 0.885 | (see per_layer in raw.json) |
| `alphabeta_2_-1` | 5/5 | 3.09e+08 | 0.853 | (see per_layer in raw.json) |
| `alphabeta_05_05` | 5/5 | 43.159 | 0.738 | (see per_layer in raw.json) |

## Per-block max|R| trajectory (sample 0)

Block-by-block magnitude trajectory shows whether the AlphaBeta variants control the per-layer amplification documented in `RESEARCH_NOTES.md` Entry 4.

| block | `baseline_2y_eps` | `alphabeta_1_0` | `alphabeta_2_-1` | `alphabeta_05_05` |
|---|---:|---:|---:|---:|

Raw per-sample / per-layer dump in `raw.json`. Conclusions appended to `RESEARCH_NOTES.md` Entry 6.
