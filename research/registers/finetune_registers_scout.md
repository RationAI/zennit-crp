# Scout: does finetuning reintroduce register outliers? Literature + recipe

*Sonnet scout, 2026-07-25. Full report below is verbatim-condensed; decisions
distilled first.*

## Distilled decisions

- **Nobody has tested our exact question** (register-equipped backbone,
  full end-to-end finetune, do patch outliers reappear?). Publishable gap.
- Registers paper: outliers = large models + after ~1/3 of training; fix is
  purely architectural (no regularizer); 4 registers = adopted default;
  finetuning not studied.
- Closest real-world analogue: Feng et al. (arXiv:2510.17201) full-finetune
  DINOv2+registers at **backbone lr 5e-6** (precaution, not ablation).
- Causal hyperparameter evidence (arXiv:2405.19279): LOWER LR, larger Adam
  eps, non-diagonal optimizers suppress outlier-feature formation
  (pretraining evidence).
- Post-hoc registers exist: PH-Reg (2505.21501, self-distilled, backbone
  mostly frozen); test-time register-neuron shifting (2506.08010).
- DINOv3 report: patch norms stable through pretraining WITH registers, but
  long optimization degrades patch locality via CLS-patch cosine creep
  (Gram anchoring fix). Diagnostics to monitor in OUR finetunes: patch-norm
  max/mean ratio AND CLS-patch cosine.
- **Our experiment design consequence**: run TWO finetune arms —
  (A) standard M1 protocol (backbone lr 5e-4) = the "naive finetune risk"
  arm; (B) conservative literature recipe (backbone lr ~5e-6, LLRD, early
  stopping) = the "prevention" arm. E3 compares pretrained vs A vs B.

## Full scout report

(see git history of this file for the verbatim agent report; key citations:
Darcet et al. 2309.16588 Fig.4/Fig.8; Jiang et al. 2506.08010; PH-Reg
2505.21501; cross-arch reassessment 2603.25803; outlier-features training
dynamics 2405.19279; DINOv3 2508.10104; Feng et al. 2510.17201.)
