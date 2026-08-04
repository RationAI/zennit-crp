from zennit.core import Hook


class ResidualRatio(Hook):
    """Otsuki ratio-split residual rule for a 2-input add ``y = x + branch``:
    distribute ``R_y`` ∝ ``|x|`` vs ``|branch|``. Attach to
    :class:`~zennit_ext.attention_unfolded.ResidualAdd`.

    Sourced from 'Layer-Wise Relevance Propagation with Conservation Property for
    ResNet', https://doi.org/10.1007/978-3-031-72775-7_20
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
        return (rel * x.abs() / denom, rel * branch.abs() / denom)

    def copy(self):
        return ResidualRatio(self.epsilon)