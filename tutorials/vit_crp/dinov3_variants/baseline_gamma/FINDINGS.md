# `baseline_gamma` — `AttnLRPGammaComposite(gamma=0.25)`

**Status:** ❌ all-NaN heatmaps on DINOv3 (5/5 classes), same failure
mode as `baseline_epsilon`.

Same root cause: missing bilinear matmul rule + missing residual rule.
γ-LRP on Linears alone cannot fix transformer-specific issues. AttnLRP
§3.2.1's γ recommendation was validated on standard timm ViT (no
LayerScale); on DINOv3's EvaBlock stack it inherits the same gap.

**No notebook in this folder.**
