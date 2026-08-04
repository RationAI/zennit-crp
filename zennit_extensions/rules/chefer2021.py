from zennit.core import Hook, stabilize


def _chefer_normalize(r_u, r_v, rel, epsilon):
    """Chefer et al. (CVPR 2021) binary-op conservation normalisation (Eq. 9):
    rescale the two operand relevances so they split the incoming relevance
    ``rel`` by their absolute mass and conserve the signed sum. Literal paper
    form ``R̄^u = R^u · |S_u|/(|S_u|+|S_v|) · (Σ rel)/S_u``; every division goes
    through zennit's :func:`~zennit.core.stabilize`. Per-sample scalars (sum over
    all but the batch dim).
    """
    dims = tuple(range(1, rel.dim()))
    s_u = r_u.sum(dims, keepdim=True)
    s_v = r_v.sum(dims, keepdim=True)
    r_tot = rel.sum(dims, keepdim=True)
    denom = stabilize(s_u.abs() + s_v.abs(), epsilon)
    r_u = r_u * (s_u.abs() / denom) * (r_tot / stabilize(s_u, epsilon))
    r_v = r_v * (s_v.abs() / denom) * (r_tot / stabilize(s_v, epsilon))
    return (r_u, r_v)


class CheferMatmul(Hook):
    """Chefer et al. (CVPR 2021) relevance rule for a 2-input matmul ``y = a @ b``:
    the z-rule (gradient×input) decomposition onto each operand followed by the
    Eq. 9 conservation normalisation. The paper applies this to BOTH attention
    matmuls and skip-connection adds (matrix multiplication otherwise violates
    conservation, Lemma 1). Attach to
    :class:`~zennit_ext.attention_unfolded.BilinearMatmul`.

    NB. The authors' released code normalises only the ``Add`` layer and leaves
    ``einsum`` (the matmul) un-normalised — a paper/code mismatch; this follows
    the paper.

    Sourced from 'Transformer Interpretability Beyond Attention Visualization',
    https://doi.org/10.1109/CVPR46437.2021.00084
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
        rel = grad_output[0]
        s = rel / stabilize(self.stored_tensors["output"], self.epsilon)
        r_a = a * (s @ b.transpose(-1, -2))
        r_b = b * (a.transpose(-1, -2) @ s)
        return _chefer_normalize(r_a, r_b, rel, self.epsilon)

    def copy(self):
        return CheferMatmul(self.epsilon)


class CheferAdd(Hook):
    """Chefer et al. (CVPR 2021) relevance rule for a 2-input add ``y = x + b``:
    the z-rule split followed by the same Eq. 9 conservation normalisation as
    :class:`CheferMatmul`. Mirrors their ``Add`` layer. Attach to
    :class:`~zennit_ext.attention_unfolded.ResidualAdd`.

    Sourced from 'Transformer Interpretability Beyond Attention Visualization',
    https://doi.org/10.1109/CVPR46437.2021.00084
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["x"] = args[0]
        self.stored_tensors["branch"] = args[1]
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        rel = grad_output[0]
        s = rel / stabilize(self.stored_tensors["output"], self.epsilon)
        r_x = self.stored_tensors["x"] * s
        r_b = self.stored_tensors["branch"] * s
        return _chefer_normalize(r_x, r_b, rel, self.epsilon)

    def copy(self):
        return CheferAdd(self.epsilon)