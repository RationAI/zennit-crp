import torch
from zennit.core import Hook


class StopGradient(Hook):
    """LRP rule that blocks relevance flow through the module.

    Sourced from 'XAI for Transformers: Better Explanations through Conservative
    Propagation', https://proceedings.mlr.press/v162/ali22a.html

    Backward returns a zero tensor in place of every non-``None`` element
    of ``grad_input``. Forward is untouched.
    """

    def backward(self, module, grad_input, grad_output):
        return tuple(
            torch.zeros_like(gi) if gi is not None else None
            for gi in grad_input
        )

    def copy(self) -> "StopGradient":
        return type(self)()