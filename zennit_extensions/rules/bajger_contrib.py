import torch
from zennit.core import BasicHook, Hook, ParamMod


class MultiInputBasicHook(BasicHook):
    """Own contribution — :class:`zennit.core.BasicHook` generalised to modules
    with multiple tensor inputs (e.g.
    :class:`~zennit_extensions.attention_unfolded.BilinearMatmul`,
    :class:`~zennit_extensions.attention_unfolded.ResidualAdd`).

    Stock ``BasicHook`` computes gradients only w.r.t. the first input and
    hands the resulting relevance to every ``grad_input`` slot whose shape
    matches — for two-operand modules this cuts relevance flow, or silently
    mis-routes it when operand shapes coincide (square matmul). This subclass
    overrides only ``backward``: every tensor input that required a gradient
    is differentiated, each ``input_modifier`` is applied to every such input,
    and the ``reducer`` is applied per input slot with the same
    ``(inputs, gradients)`` signature as in ``BasicHook``. Relevance is
    returned per slot, with ``None`` for non-differentiable inputs — no
    shape-matching heuristic.

    ``__init__``, ``forward``, ``copy`` and the default modifiers, gradient
    mapper and reducer are inherited unchanged, so rules parameterise this
    class exactly like the stock zennit rules parameterise ``BasicHook``.
    """

    def backward(self, module, grad_input, grad_output):
        original_args = self.stored_tensors["input"]
        original_kwargs = self.stored_tensors["kwargs"]
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
        grad_outputs = self.gradient_mapper(grad_output[0], outputs)
        gradients = torch.autograd.grad(
            outputs,
            [arg for mod_inputs in inputs for arg in mod_inputs],
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevances = iter(
            self.reducer(
                [mod_inputs[slot] for mod_inputs in inputs],
                [gradients[mod * num_diff + slot] for mod in range(len(inputs))],
            )
            for slot in range(num_diff)
        )
        return tuple(next(relevances) if diff else None for diff in diff_mask)


class ResidualL1(Hook):
    """Own contribution — sign-preserving, L1-conserving residual split for
    ``y = x + branch``: ``R_x = R_y·x/S``, ``R_branch = R_y·branch/S`` with
    ``S = |x|+|branch|+ε``. Keeps each operand's sign, bounded (``|R_x|≤|R_y|``),
    no cancellation pole. Conserves L1 mass, not the signed sum. Attach to
    :class:`~zennit_extensions.attention_unfolded.ResidualAdd`.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["x"] = args[0]
        self.stored_tensors["branch"] = args[1]

    def backward(self, module, grad_input, grad_output):
        x, branch = self.stored_tensors["x"], self.stored_tensors["branch"]
        rel = grad_output[0]
        denom = x.abs() + branch.abs() + self.epsilon
        return (rel * x / denom, rel * branch / denom)

    def copy(self):
        return ResidualL1(self.epsilon)


class AlphaBetaMatmul(Hook):
    """Own contribution — AlphaBeta LRP rule generalised to a 2-input bilinear
    matmul ``y = a @ b`` (separate positive/negative pre-activation paths, ``α+β=1``
    ⇒ exact conservation modulo ε). Attach to
    :class:`~zennit_extensions.attention_unfolded.BilinearMatmul` via a composite
    ``layer_map``.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, epsilon: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["a"] = args[0]
        self.stored_tensors["b"] = args[1]

    def backward(self, module, grad_input, grad_output):
        a, b = self.stored_tensors["a"], self.stored_tensors["b"]
        rel = grad_output[0]
        a_pos, a_neg = a.clamp(min=0), a.clamp(max=0)
        b_pos, b_neg = b.clamp(min=0), b.clamp(max=0)
        y_pos = a_pos @ b_pos + a_neg @ b_neg
        y_neg = a_pos @ b_neg + a_neg @ b_pos
        f = rel / (y_pos + self.epsilon)
        g = rel / (y_neg - self.epsilon)
        bpT, bnT = b_pos.transpose(-1, -2), b_neg.transpose(-1, -2)
        grad_a = 0.5 * (
            a_pos * (self.alpha * (f @ bpT) + self.beta * (g @ bnT))
            + a_neg * (self.alpha * (f @ bnT) + self.beta * (g @ bpT))
        )
        apT, anT = a_pos.transpose(-1, -2), a_neg.transpose(-1, -2)
        grad_b = 0.5 * (
            b_pos * (self.alpha * (apT @ f) + self.beta * (anT @ g))
            + b_neg * (self.alpha * (anT @ f) + self.beta * (apT @ g))
        )
        return (grad_a, grad_b)

    def copy(self):
        return AlphaBetaMatmul(self.alpha, self.beta, self.epsilon)