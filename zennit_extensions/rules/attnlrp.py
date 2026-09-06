"""AttnLRP rules (Achtibat et al. 2024,
https://proceedings.mlr.press/v235/achtibat24a.html) and their
grad×input-convention variants."""
import torch
from zennit.canonizers import AttributeCanonizer
from zennit.core import BasicHook, Hook, ParamMod, Stabilizer, stabilize
from zennit.rules import Gamma, NoMod

#: Bias handling (Appendix A.2.1): 'absorb' keeps the bias its relevance
#: share, 'omit' drops it from the decomposition, 'distribute' spreads its
#: share uniformly over the inputs.
BIAS_MODES = ("absorb", "omit", "distribute")


def _check_bias_mode(bias_mode: str) -> str:
    if bias_mode not in BIAS_MODES:
        raise ValueError(f"bias_mode must be one of {BIAS_MODES}, got {bias_mode!r}")
    return bias_mode


class SoftmaxAttnLRP(Hook):
    r"""Softmax rule (Prop. 3.1) for ``y = softmax(x)`` along the last dim,
    with ``s = softmax(x)``: ``R_i = x_i·(R_i − s_i·Σ_j R_j)`` — the single
    variant the paper specifies (ε-free). Attach to
    :class:`~zennit_extensions.attention_unfolded.SoftmaxAlongLastDim`.
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
    """Bilinear rule (Eq. 15) for ``y = a @ b``: ``s = R/(2y+ε)``,
    ``R_a = a ⊙ (s @ bᵀ)``, ``R_b = b ⊙ (aᵀ @ s)``. Attach to
    :class:`~zennit_extensions.attention_unfolded.BilinearMatmul`."""

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
        grad_a = A * (s @ B.transpose(-1, -2))  # \sum_{j} R_{ij} B_{jk} = (R @ B^T)_{ik}
        grad_b = B * (A.transpose(-1, -2) @ s)  # \sum_{i} A_{ji} R_{ik} = (A^T @ R)_{jk}
        return (grad_a, grad_b)

    def copy(self):
        return MatmulAttnLRP(self.epsilon)


class EpsilonAdd(Hook):
    """ε rule for ``y = x + branch``: ``R_x = R·x/(y+ε)``,
    ``R_branch = R·branch/(y+ε)``. Attach to
    :class:`~zennit_extensions.attention_unfolded.ResidualAdd`."""

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
    """ε rule for
    :class:`~zennit_extensions.attention_unfolded.LayerNormDetachedStd`:
    ``R_x = x ⊙ ∇_x⟨y, R/(y+ε)⟩``. ``bias_mode`` for β: ``'absorb'``
    (default) leaves β its share ``R·β/(y+ε)``; ``'omit'`` removes β from
    forward and denominator; ``'distribute'`` spreads β's share uniformly
    over the normalized dims. Modes coincide without β."""

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


# ── grad×input convention ────────────────────────────────────────────────────
# Backward stream carries g with R = g·x at every tensor; read the heatmap as
# x.grad * x.


class GradTimesInputMultiInputBasicHook(BasicHook):
    """:class:`zennit.core.BasicHook` generalised to modules with multiple
    tensor inputs, running in the grad×input convention: entry converts the
    incoming g to relevance (``g·y``); every tensor input requiring grad is
    differentiated, each ``input_modifier`` applied to every such input, the
    ``reducer`` run per input slot with the ``BasicHook``
    ``(inputs, gradients)`` signature; exit converts each slot back
    (``R_i / stabilize(x_i)``). ``None`` for inputs without grad. Single
    output only (asserted). Subclass and pass a BasicHook parameterisation
    to ``__init__``."""

    def forward(self, module, args, kwargs, output):
        super().forward(module, args, kwargs, output)
        self.stored_tensors['output'] = output

    def backward(self, module, grad_input, grad_output):
        assert len(grad_output) == 1, 'single output only'
        rel_out = grad_output[0] * self.stored_tensors['output']
        rel_out.requires_grad = True
        original_args = self.stored_tensors['input']
        original_kwargs = self.stored_tensors['kwargs']
        diff_mask = [
            isinstance(arg, torch.Tensor) and arg.requires_grad for arg in original_args
        ]
        num_diff = sum(diff_mask)
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(
            self.input_modifiers, self.param_modifiers, self.output_modifiers
        ):
            args = [
                in_mod(arg.clone()).requires_grad_() if diff else arg
                for arg, diff in zip(original_args, diff_mask)
            ]
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = out_mod(modified.forward(*args, **original_kwargs))
            inputs.append([arg for arg, diff in zip(args, diff_mask) if diff])
            outputs.append(output)
        grad_outputs = self.gradient_mapper(rel_out, outputs)
        gradients = torch.autograd.grad(
            outputs,
            [arg for mod_inputs in inputs for arg in mod_inputs],
            grad_outputs=grad_outputs,
            create_graph=rel_out.requires_grad,
        )
        relevances = iter(
            self.reducer(
                [mod_inputs[slot] for mod_inputs in inputs],
                [gradients[mod * num_diff + slot] for mod in range(len(inputs))],
            )
            for slot in range(num_diff)
        )
        return tuple(
            next(relevances) / stabilize(orig, epsilon=1e-10) if diff else None
            for orig, diff in zip(original_args, diff_mask))


class GammaGradInput(GradTimesInputMultiInputBasicHook):
    """:class:`zennit.rules.Gamma` parameterisation in the grad×input
    convention."""

    def __init__(self, gamma: float = 0.25, stabilizer=1e-6, zero_params=None):
        proto = Gamma(gamma, stabilizer, zero_params)
        super().__init__(
            input_modifiers=proto.input_modifiers,
            param_modifiers=proto.param_modifiers,
            output_modifiers=proto.output_modifiers,
            gradient_mapper=proto.gradient_mapper,
            reducer=proto.reducer,
        )


class EpsilonAddGradTimesInput(GradTimesInputMultiInputBasicHook):
    """ε add rule (:class:`EpsilonAdd`) in the grad×input convention."""

    def __init__(self, epsilon: float = 1e-6):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod()],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class MatmulAttnLRPGradTimesInput(GradTimesInputMultiInputBasicHook):
    """Bilinear matmul rule (:class:`MatmulAttnLRP`, Eq. 15) in the
    grad×input convention."""

    def __init__(self, epsilon: float = 1e-6):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod()],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(2.0 * outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class IdentityGradTimesInput(Hook):
    """Elementwise identity in the grad×input convention:
    ``grad_in = grad_out · y/(x+ε)``. Attach to ``nn.GELU``; not valid in
    the stock relevance convention."""

    def __init__(self, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["input"] = args[0]
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        x = self.stored_tensors["input"]
        y = self.stored_tensors["output"]
        return (grad_output[0] * (y / (x + self.epsilon)),)

    def copy(self):
        return IdentityGradTimesInput(self.epsilon)


class _TorchvisionMHAAdapter(torch.nn.Module):
    """Present a torchvision fused ``nn.MultiheadAttention`` in timm
    ``Attention`` shape — a packed ``qkv`` Linear sharing ``in_proj_weight`` /
    ``in_proj_bias`` by reference, ``proj`` = the original ``out_proj`` —
    so :class:`~zennit_extensions.attention_unfolded.TimmAttentionUnfolded`
    can unfold it. Checkpoint loading through the original MHA still flows
    (parameters are shared, not copied)."""

    def __init__(self, mha: torch.nn.Module) -> None:
        super().__init__()
        embed_dim = mha.embed_dim
        self.num_heads = int(mha.num_heads)
        self.head_dim = embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=mha.in_proj_bias is not None)
        self.qkv.weight = mha.in_proj_weight          # shared, not copied
        if mha.in_proj_bias is not None:
            self.qkv.bias = mha.in_proj_bias
        self.proj = mha.out_proj
        self.attn_drop = torch.nn.Dropout(mha.dropout)
        self.proj_drop = torch.nn.Dropout(0.0)


class TorchvisionEncoderBlockCanonizer(AttributeCanonizer):
    """Canonize torchvision ``EncoderBlock`` to the SAME layout the timm
    canonizers produce: the fused MHA is unfolded via
    :class:`~zennit_extensions.attention_unfolded.TimmAttentionUnfolded`
    (through :class:`_TorchvisionMHAAdapter`) — exposing ``qkv``/``proj``
    Linears, ``BilinearMatmul``/``SoftmaxAlongLastDim`` atomics and the
    ``Q/K/VInspectionLayer`` probes — and both residual additions route
    through :class:`~zennit_extensions.attention_unfolded.ResidualAdd`.
    One layer_map then applies identically to torchvision and timm models.
    torchvision imported lazily."""

    def __init__(self):
        super().__init__(self._attribute_map)

    def _attribute_map(self, _name, module):
        from torchvision.models.vision_transformer import EncoderBlock
        if not isinstance(module, EncoderBlock):
            return None
        from zennit_extensions.attention_unfolded import ResidualAdd, TimmAttentionUnfolded
        from zennit_extensions.canonisation.canonizers import _bind_forward

        def fwd(self, input):
            torch._assert(input.dim() == 3,
                          f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
            x = self.ln_1(input)
            x = self._attn_unfolded(x)                # unfolded attention (timm layout)
            x = self.dropout(x)
            x = self._lrp_res1(input, x)              # rule site: residual rule
            y = self.ln_2(x)
            y = self.mlp(y)
            return self._lrp_res2(x, y)               # rule site: residual rule

        attrs = _bind_forward(module, fwd)
        attrs["_attn_unfolded"] = TimmAttentionUnfolded(
            _TorchvisionMHAAdapter(module.self_attention), num_prefix_tokens=1)
        attrs["_lrp_res1"] = ResidualAdd()
        attrs["_lrp_res2"] = ResidualAdd()
        return attrs

    def copy(self):
        return type(self)()
