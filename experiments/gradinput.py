"""Grad×input-convention adapters for the CRP machinery.

Under a grad×input (g-convention) composite the backward stream carries g with
R = g·x at every tensor: masking/conditioning works unchanged (same zero
support), but every READOUT must materialise relevance by multiplying with the
stored activation. These subclasses do exactly that — nothing else:

* :class:`GradTimesInputAttribution` — ``heatmap = data.grad * data``;
  recorded layer relevances = captured grad × captured activation.
* :class:`GradTimesInputFeatureVisualization` — the RelMax/RelStats analysis
  receives grad × activation instead of the raw grad.
"""
from __future__ import annotations

import torch

from crp.attribution import CondAttribution
from crp.hooks import FeatVisHook
from crp.visualization import FeatureVisualization


class GradTimesInputAttribution(CondAttribution):
    """CondAttribution with g-convention read-outs."""

    def heatmap_modifier(self, data, on_device=None):
        heatmap = (data.grad * data).detach()
        heatmap = heatmap.to(on_device) if on_device else heatmap
        return torch.sum(heatmap, dim=1)

    def _collect_hook_activation_relevance(self, layer_out, on_device=None, length=None):
        activations, relevances = super()._collect_hook_activation_relevance(
            layer_out, on_device, length)
        relevances = {name: rel * activations[name] for name, rel in relevances.items()}
        return activations, relevances


class _GradTimesInputFeatVisHook(FeatVisHook):
    """FeatVisHook whose relevance analysis receives grad × activation.

    ``self.FV.negative_clamp`` decides whether negative relevance reaches the
    index: ``True`` (default) clamps it away, ``False`` analyzes the signed
    tensor."""

    def post_forward(self, module, input, output):
        out0 = output[0] if isinstance(output, tuple) else output
        self._activation = out0.detach()
        return super().post_forward(module, input, output)

    def backward(self, module, grad):
        relevance = grad.detach() * self._activation
        if getattr(self.FV, "negative_clamp", True):
            relevance = relevance.clamp(min=0)
        relevance = relevance.to(self.on_device) if self.on_device else relevance
        s_indices, targets = self.dict_inputs["sample_indices"], self.dict_inputs["targets"]
        self.FV.analyze_relevance(relevance, self.layer_name, self.concept, s_indices, targets)
        return grad


class GradTimesInputFeatureVisualization(FeatureVisualization):
    """FeatureVisualization whose index analysis runs on g × activation.

    ``run_distributed`` is inherited; only the hook class differs — injected by
    shadowing the name the parent looks up at construction time inside
    :meth:`run_distributed` (the hook is instantiated per layer there), so the
    override wraps that one call.

    ``negative_clamp`` (default ``True``) selects the relevance-sign flavour of
    the index: clamped (negative relevance discarded) or fully signed.
    """

    def __init__(self, *args, negative_clamp: bool = True, **kwargs):
        self.negative_clamp = negative_clamp
        super().__init__(*args, **kwargs)

    def run_distributed(self, composite, data_start, data_end, batch_size=16,
                        checkpoint=500, on_device=None):
        import crp.visualization as _vis
        original = _vis.FeatVisHook
        _vis.FeatVisHook = _GradTimesInputFeatVisHook
        try:
            return super().run_distributed(composite, data_start, data_end,
                                           batch_size, checkpoint, on_device)
        finally:
            _vis.FeatVisHook = original
