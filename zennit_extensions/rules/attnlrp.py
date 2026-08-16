import torch
from zennit.core import BasicHook, Hook, Stabilizer, stabilize
from zennit.rules import NoMod

from zennit_extensions.rules.bajger_contrib import MultiInputBasicHook

#: Bias-handling modes named in AttnLRP Appendix A.2.1: keep the Taylor/affine
#: bias term absorbing its relevance share (the paper's recommended, stable
#: choice), omit it from the decomposition (strict conservation), or
#: distribute its share uniformly across the input variables (Voita et al.
#: 2021 style; also strict conservation).
BIAS_MODES = ("absorb", "omit", "distribute")


def _check_bias_mode(bias_mode: str) -> str:
    if bias_mode not in BIAS_MODES:
        raise ValueError(f"bias_mode must be one of {BIAS_MODES}, got {bias_mode!r}")
    return bias_mode


class SoftmaxAttnLRP(Hook):
    r"""AttnLRP softmax rule (Proposition 3.1) for ``y = softmax(x)`` along the
    last dim. The softmax-SPECIFIC rule, NOT the generic elementwise identity
    (:class:`zennit.rules.Pass` / :class:`Identity`): it keeps the cross-term
    coupling all positions and the input scaling. Map
    :class:`~zennit_extensions.attention_unfolded.SoftmaxAlongLastDim` to this hook.
    Only fires when the attention weights are differentiable (full AttnLRP); under
    CP-LRP (StopGradient on Q/K) the softmax is a graph constant.

    The Taylor decomposition at x carries a bias term
    ``b̃_j = s_j·(1 − x_j + Σ_i s_i x_i)``; ``bias_mode`` selects among the
    handling options of Appendix A.2.1:

    * ``'absorb'`` (default, the paper's recommended stable choice, = Prop 3.1):
      the bias keeps its relevance share, ``R_i = x_i·(R_i − s_i·Σ_j R_j)``.
    * ``'omit'``: bias excluded from the denominator — the mapped relevance is
      normalised by ``(Jx)_j = s_j(x_j − Σ_i s_i x_i)`` instead of ``s_j``;
      strictly conserving, but the paper reports this family of choices can be
      numerically unstable for a standalone softmax (A.2.1).
    * ``'distribute'``: the bias share ``b̃_j·R_j/s_j`` is spread uniformly over
      the N inputs (Voita et al. 2021 style); strictly conserving, same
      stability caveat.

    Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
    Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
    """

    def __init__(self, bias_mode: str = "absorb", epsilon: float = 1e-6):
        super().__init__()
        self.bias_mode = _check_bias_mode(bias_mode)
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["input"] = args[0]
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        x = self.stored_tensors["input"]
        s = self.stored_tensors["output"]
        rel = grad_output[0]
        if self.bias_mode == "omit":
            jx = s * (x - (s * x).sum(dim=-1, keepdim=True))
            v = rel / stabilize(jx, self.epsilon)
            return (x * s * (v - (s * v).sum(dim=-1, keepdim=True)),)
        relevance = x * (rel - s * rel.sum(dim=-1, keepdim=True))
        if self.bias_mode == "distribute":
            c = (s * x).sum(dim=-1, keepdim=True)
            # bias share per output j is b̃_j·R_j/s_j = (1 + c − x_j)·R_j,
            # spread uniformly over the N softmax inputs
            share = ((1.0 + c - x) * rel).sum(dim=-1, keepdim=True) / x.shape[-1]
            relevance = relevance + share
        return (relevance,)

    def copy(self):
        return SoftmaxAttnLRP(self.bias_mode, self.epsilon)


class MatmulAttnLRP(Hook):  # reviewed by AVB
    """AttnLRP bilinear rule for a 2-input matmul ``y = a @ b`` (Eq. 15): both
    operands share the ``2·output + ε`` stabiliser, splitting conservation in
    half between the two factors. Attach to
    :class:`~zennit_extensions.attention_unfolded.BilinearMatmul` (both the ``q@kᵀ`` and
    ``attn@v`` products). Numerically matches the LXT reference matmul rule.

    Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
    Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["a"] = args[0]
        self.stored_tensors["b"] = args[1]
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        A, B, O = self.stored_tensors["a"], self.stored_tensors["b"], self.stored_tensors["output"]
        R_out = grad_output[0]
        s = R_out / stabilize(2.0 * O, self.epsilon)
        grad_a = A * (s @ B.transpose(-1, -2))  # \sum_{j} R_{ij} * B_{jk} = (R @ B^T)_{ik}
        grad_b = B * (A.transpose(-1, -2) @ s)  # \sum_{i} A_{ij} * R_{ik} = (A^T @ R)_{jk}
        return (grad_a, grad_b)

    def copy(self):
        return MatmulAttnLRP(self.epsilon)


class EpsilonAdd(Hook):
    """AttnLRP standard ε add rule for a 2-input residual add ``y = x + branch``:
    signed proportional split ``R_x = R_y·x/(y+ε)``, ``R_branch = R_y·branch/(y+ε)``
    (conserves the signed sum). AttnLRP's skip-connection handling; mirrors LXT's
    ``add2`` rule. Attach to
    :class:`~zennit_extensions.attention_unfolded.ResidualAdd`.

    Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
    Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["x"] = args[0]
        self.stored_tensors["branch"] = args[1]
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        x, branch = self.stored_tensors["x"], self.stored_tensors["branch"]
        s = grad_output[0] / stabilize(self.stored_tensors["output"], self.epsilon)
        return (x * s, branch * s)

    def copy(self):
        return EpsilonAdd(self.epsilon)


class LayerNormEpsilon(Hook):
    """ε-rule for :class:`~zennit_extensions.attention_unfolded.LayerNormDetachedStd`
    — the LXT ``layer_norm_grad_fn`` treatment: with σ detached the module is
    linear in x (identity rule on ``x/σ`` per Prop. 3.4, ε-attribution of the
    remaining affine map per §3.3.3), so the relevance is
    ``R_x = x ⊙ ∇_x⟨y, R/(y+ε)⟩``.

    ``bias_mode`` selects the β handling (Appendix A.2.1 options):

    * ``'absorb'`` (default, = LXT and the paper's recommendation): β keeps its
      share ``R·β/(y+ε)``; conservation up to that share.
    * ``'omit'``: β removed from forward and denominator; strict conservation.
    * ``'distribute'``: β's share is spread uniformly over the N normalized
      inputs; strict conservation.

    Modes coincide when the module has no β (``elementwise_affine=False`` or
    RMS-style γ-only).
    """

    def __init__(self, epsilon: float = 1e-6, bias_mode: str = "absorb"):
        super().__init__()
        self.epsilon = epsilon
        self.bias_mode = _check_bias_mode(bias_mode)

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["input"] = args[0]

    def backward(self, module, grad_input, grad_output):
        rel = grad_output[0]
        x = self.stored_tensors["input"].clone().requires_grad_()
        with torch.autograd.enable_grad():
            y = module.forward(x)
            if self.bias_mode == "omit" and module.bias is not None:
                y = y - module.bias
        v = rel / stabilize(y.detach(), self.epsilon)
        (gradient,) = torch.autograd.grad(y, x, v)
        relevance = x.detach() * gradient
        if self.bias_mode == "distribute" and module.bias is not None:
            dims = tuple(range(-len(module.normalized_shape), 0))
            numel = 1
            for size in module.normalized_shape:
                numel *= size
            relevance = relevance + (module.bias * v).sum(dim=dims, keepdim=True) / numel
        return (relevance,)

    def copy(self):
        return LayerNormEpsilon(self.epsilon, self.bias_mode)


# ─── BasicHook-based reformulations ─────────────────────────────────────────
# Each rule above restated as a parameterisation of (MultiInput)BasicHook, in
# the same idiom as zennit.rules.Epsilon. Kept side by side with the raw Hook
# originals for numerical comparison (see tests/test_basichook_variants.py);
# one of the two sets is to be removed during cleanup.


class SoftmaxAttnLRPBasicHook(BasicHook):
    """:class:`SoftmaxAttnLRP` as a ``BasicHook`` parameterisation: the
    default input×gradient schema with ``R/softmax(x)`` as the incoming
    gradient. Equivalence: with ``v = R/s``, ``(Jᵀv)_j = Σ_i (R_i/s_i)·
    s_i(δ_ij − s_j) = R_j − s_j·Σ_i R_i``, so ``x ⊙ Jᵀ(R/s) = x·(R − s·ΣR)``
    — exactly Proposition 3.1. ``epsilon=0`` (default) keeps the division
    exact, matching the division-free original; softmax output is strictly
    positive so no stabilisation is needed.

    Mirrors :class:`SoftmaxAttnLRP` with ``bias_mode='absorb'`` only: the
    ``'distribute'`` mode needs an additive bias share (outside the
    input×gradient reducer schema) and ``'omit'`` needs an input-dependent
    denominator ``Jx`` (the gradient mapper only sees outputs) — neither is
    expressible as a BasicHook parameterisation.
    """

    def __init__(self, epsilon: float = 0.0):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod()],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class MatmulAttnLRPBasicHook(MultiInputBasicHook):
    """:class:`MatmulAttnLRP` as a ``MultiInputBasicHook`` parameterisation:
    the ε-rule with the shared ``2·output`` stabilised denominator (Eq. 15).
    With ``s = R/(2y+ε)``, autograd of ``y = a @ b`` yields per-operand
    gradients ``s @ bᵀ`` and ``aᵀ @ s``, and the input×gradient reducer
    reproduces the raw hook's ``grad_a``/``grad_b`` exactly.
    """

    def __init__(self, epsilon: float = 1e-6):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod()],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(2.0 * outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class EpsilonAddBasicHook(MultiInputBasicHook):
    """:class:`EpsilonAdd` as a ``MultiInputBasicHook`` parameterisation: the
    plain ε-rule. Since ``∂(x+branch)/∂x = I``, the mapped gradient
    ``R/(y+ε)`` passes through unchanged to both operands and the
    input×gradient reducer gives the signed proportional split.
    """

    def __init__(self, epsilon: float = 1e-6):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod()],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )