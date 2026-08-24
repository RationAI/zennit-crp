"""Chefer et al. (CVPR 2021) LRP rules — code-exact (released implementation).

Matches the authors' *released code* bit-for-bit, not the paper's Eq. 9:
https://github.com/hila-chefer/Transformer-Explainability (commit c3e578f),
``modules/layers_ours.py`` + ``baselines/ViT/ViT_LRP.py``.

Only the two rules the reference's ``transformer_attribution`` path actually
exercises live here; both are ``safe_divide`` variants of an existing z-rule, so
they subclass the AttnLRP kernels and override **only** the backward:

* :class:`CheferMatmul` — plain z-rule + ÷2 on both branches (their ``einsum``
  ``RelPropSimple`` followed by ``cam /= 2``; ``ViT_LRP.py:160-173``). Same
  bilinear backward as :class:`~zennit_extensions.rules.attnlrp.MatmulAttnLRP`,
  but with Chefer's :func:`safe_divide` stabilizer instead of zennit's
  ``stabilize`` (they diverge near-zero denominators, so the swap is not free).
* :class:`CheferAdd` — z-rule split + absolute-mass renorm over **GLOBAL** sums
  (incl. batch; ``layers_ours.py:97-120``). Extends
  :class:`~zennit_extensions.rules.attnlrp.EpsilonAdd`'s split with the renorm.

The first-conv z-box (``Conv2d.relprop`` ``X.shape[1]==3`` branch) is **not**
reproduced: ``transformer_attribution`` reads the attention relevance ``R_A`` at
each softmax (above every block) and never propagates below it, so no reference
ground truth exercises the conv. Pixel-space Chefer (``method="full"``) is out of
scope.

Sourced from 'Transformer Interpretability Beyond Attention Visualization',
https://doi.org/10.1109/CVPR46437.2021.00084
"""
from zennit_extensions.rules.attnlrp import EpsilonAdd, MatmulAttnLRP


def safe_divide(a, b):
    """Chefer's stabilizer, verbatim from ``layers_ours.py``.

    ``den = clamp(b, min=1e-9) + clamp(b, max=1e-9)`` ≈ ``b + 1e-9`` for
    ``|b| > 1e-9``; ``+1e-9`` where ``den == 0``; multiply by ``(b ≠ 0)``.
    """
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).to(den.dtype) * 1e-9
    return a / den * b.ne(0).to(b.dtype)


class CheferMatmul(MatmulAttnLRP):
    """Chefer et al. (CVPR 2021) matmul rule — **code-exact**.

    Reuses :class:`~zennit_extensions.rules.attnlrp.MatmulAttnLRP`'s tensor
    capture (identical ``forward``); overrides only the backward to use Chefer's
    :func:`safe_divide` and their external ``cam /= 2`` (``ViT_LRP.py:160-173``).
    ``r_a = ½·a·(s @ bᵀ)``, ``r_b = ½·b·(aᵀ @ s)`` with ``s = safe_divide(R, O)``.

    Attach to :class:`~zennit_extensions.attention_unfolded.BilinearMatmul`.
    """

    def backward(self, module, grad_input, grad_output):
        a = self.stored_tensors["a"]
        b = self.stored_tensors["b"]
        s = safe_divide(grad_output[0], self.stored_tensors["output"])
        r_a = 0.5 * a * (s @ b.transpose(-1, -2))
        r_b = 0.5 * b * (a.transpose(-1, -2) @ s)
        return (r_a, r_b)

    def copy(self):
        return CheferMatmul()


class CheferAdd(EpsilonAdd):
    """Chefer et al. (CVPR 2021) residual-add rule — **code-exact**.

    Reuses :class:`~zennit_extensions.rules.attnlrp.EpsilonAdd`'s tensor capture;
    overrides the backward with Chefer's z-rule split (:func:`safe_divide`) plus
    absolute-mass renormalization over **GLOBAL** sums (incl. the batch
    dimension), matching their ``Add.relprop`` (``layers_ours.py:97-120``):
    ``a_fact = safe_divide(|Σa|, |Σa|+|Σb|) · ΣR``. Equal to a per-sample
    formulation at ``B=1``; conserves only the global sum for ``B>1``.

    Attach to :class:`~zennit_extensions.attention_unfolded.ResidualAdd`
    and :class:`~zennit_extensions.attention_unfolded.PosEmbedAdd`.
    """

    def backward(self, module, grad_input, grad_output):
        x = self.stored_tensors["x"]
        branch = self.stored_tensors["branch"]
        rel = grad_output[0]
        s = safe_divide(rel, self.stored_tensors["output"])
        a = x * s
        b = branch * s
        denom = a.sum().abs() + b.sum().abs()
        a_fact = safe_divide(a.sum().abs(), denom) * rel.sum()
        b_fact = safe_divide(b.sum().abs(), denom) * rel.sum()
        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())
        return (a, b)

    def copy(self):
        return CheferAdd()
