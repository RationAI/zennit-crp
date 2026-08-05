from zennit.core import Hook, stabilize


class Uniform(Hook):
    """Uniform allocation rule (Eq. 14): divide the incoming relevance equally,
    ``grad_input / factor``. ``factor=2`` is the per-bilinear default (e.g. the
    LayerScale γ multiply or the ``x + pos_embed`` add).

    Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
    Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def backward(self, module, grad_input, grad_output):
        return tuple(
            (g / self.factor if g is not None else None) for g in grad_input
        )

    def copy(self):
        return Uniform(self.factor)


class SoftmaxAttnLRP(Hook):
    r"""AttnLRP softmax rule (Proposition 3.1) for ``y = softmax(x)`` along the
    last dim. The softmax-SPECIFIC rule, NOT the generic elementwise identity
    (:class:`zennit.rules.Pass` / :class:`Identity`): it keeps the cross-term
    coupling all positions and the input scaling. Map
    :class:`~zennit_extensions.attention_unfolded.SoftmaxAlongLastDim` to this hook.
    Only fires when the attention weights are differentiable (full AttnLRP); under
    CP-LRP (StopGradient on Q/K) the softmax is a graph constant.

    Sourced from 'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for
    Transformers', https://proceedings.mlr.press/v235/achtibat24a.html
    """

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["input"] = args[0]
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        x = self.stored_tensors["input"]
        s = self.stored_tensors["output"]
        rel = grad_output[0]
        return (x * (rel - s * rel.sum(dim=-1, keepdim=True)),)

    def copy(self):
        return SoftmaxAttnLRP()


class MatmulAttnLRP(Hook):
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
        a, b = self.stored_tensors["a"], self.stored_tensors["b"]
        s = grad_output[0] / stabilize(2.0 * self.stored_tensors["output"], self.epsilon)
        grad_a = a * (s @ b.transpose(-1, -2))
        grad_b = b * (a.transpose(-1, -2) @ s)
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