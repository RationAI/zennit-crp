from zennit.core import Hook


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