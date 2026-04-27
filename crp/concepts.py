import torch
import numpy as np
from typing import List, Dict


class Concept:
    """
    Abstract class that imlplements the core functionality for the attribution computation of concepts.
    """

    def mask(self, batch_id, concept_ids, layer_name):

        raise NotImplementedError("'Concept'class must be implemented!")

    def mask_rf(self, neuron_ids, layer_name):

        raise NotImplementedError("'Concept'class must be implemented!")

    def reference_sampling(self, relevance, layer_name: str = None, max_target: str = "sum", abs_norm=True):

        raise NotImplementedError("'Concept'class must be implemented!")

    def get_rf_indices(self, output_shape, layer_name):

        raise NotImplementedError("'Concept'class must be implemented!")

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        raise NotImplementedError("'Concept'class must be implemented!")


class ChannelConcept(Concept):
    """
    Concept Class for torch.nn.Conv2D and torch.nn.Linear layers
    """

    @staticmethod
    def mask(batch_id: int, concept_ids: List, layer_name=None):
        """
        Wrapper that generates a function that modifies the gradient (replaced by zennit by attributions).

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        concept_ids: list of integer values
            integer lists corresponding to channel indices.

        Returns:
        --------
        callable function that modifies the gradient
        """

        def mask_fct(grad):

            mask = torch.zeros_like(grad[batch_id])
            mask[concept_ids] = 1
            grad[batch_id] = grad[batch_id] * mask

            return grad

        return mask_fct

    @staticmethod
    def mask_rf(batch_id: int, c_n_map: Dict[int, List], layer_name=None):
        """
        Wrapper that generates a function that modifies the gradient (replaced by zennit by attributions) for a single neuron.

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        c_n_map: dist with int keys and list values
            Keys correspond to channel indices and values correspond to neuron indices.
            Neuron Indices are counted as if the 2D Channel has 1D dimension i.e. channel dimension [3, 20, 20] -> [3, 400],
            so that neuron indices range between 0 and 399.

        Returns:
        --------
        callable function that modifies the gradient
        """

        def mask_fct(grad):

            grad_shape = grad.shape
            grad = grad.view(*grad_shape[:2], -1)

            mask = torch.zeros_like(grad[batch_id])

            for channel in c_n_map:
            
                mask[channel, c_n_map[channel]] = 1

            grad[batch_id] = grad[batch_id] * mask
            return grad.view(grad_shape)

        return mask_fct

    def get_rf_indices(self, output_shape, layer_name=None):

        if len(output_shape) == 1:
            return [0]
        else:
            end = np.prod(output_shape[1:])
            return np.arange(0, end)

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l

    def reference_sampling(self, relevance, layer_name: str = None, max_target: str = "sum", abs_norm=True):
        """
        Parameters:
            max_target: str. Either 'sum' or 'max'.
        """

        # position of receptive field neuron
        rel_l = relevance.view(*relevance.shape[:2], -1)
        rf_neuron = torch.argmax(rel_l, dim=-1)

        # channel maximization target
        if max_target == "sum":
            rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        elif max_target == "max":
            rel_l = torch.gather(rel_l, -1, rf_neuron.unsqueeze(-1)).squeeze(-1)

        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)
        
        d_ch_sorted = torch.argsort(rel_l, dim=0, descending=True)
        rel_ch_sorted = torch.gather(rel_l, 0, d_ch_sorted)
        rf_ch_sorted = torch.gather(rf_neuron, 0, d_ch_sorted)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted


class AttentionHeadConcept(ChannelConcept):
    """
    Concept Class for Vision Transformer attention heads.
    Each concept corresponds to a single attention head.

    Architecture-agnostic: infers head_dim from gradient tensor shape and number of heads.
    """

    def __init__(self):
        self._num_heads_by_layer = {}

    def register_num_heads(self, layer_name: str, num_heads: int):
        """Register number of attention heads for a specific layer name."""

        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("<num_heads> must be a positive integer.")
        self._num_heads_by_layer[layer_name] = num_heads

    def register_from_model(self, model):
        """
        Extract and register attention head counts from all named modules in a model.
        Supports common attribute names used by ViT implementations.
        """

        for layer_name, module in model.named_modules():
            for attr_name in ("num_heads", "n_heads", "num_attention_heads"):
                if hasattr(module, attr_name):
                    n_heads = getattr(module, attr_name)
                    if isinstance(n_heads, int) and n_heads > 0:
                        self._num_heads_by_layer[layer_name] = n_heads
                        break

    def _resolve_num_heads(self, layer_name: str):
        """
        Resolve number of heads for layer_name. If not found directly,
        progressively tries parent module names.
        """

        if not layer_name:
            return None

        if layer_name in self._num_heads_by_layer:
            return self._num_heads_by_layer[layer_name]

        parts = layer_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_name = ".".join(parts[:i])
            if parent_name in self._num_heads_by_layer:
                return self._num_heads_by_layer[parent_name]

        return None

    def mask(self, batch_id: int, concept_ids: List, layer_name=None):
        """
        Wrapper that generates a function which masks attention head dimensions
        in the gradient tensor.

        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        concept_ids: list of integer values
            Attention head indices (0..num_heads-1) to be masked.
        layer_name: str (optional)
            Name of the layer (used for architecture lookup if needed).

        Returns:
        --------
        callable function that modifies the gradient
        """

        def mask_fct(grad):

            # grad shape (typically): [batch, seq_len, hidden_dim]
            hidden_dim = grad[batch_id].shape[-1]
            num_heads = self._resolve_num_heads(layer_name)

            if num_heads is None:
                raise ValueError(
                    f"AttentionHeadConcept: could not resolve num_heads for layer '{layer_name}'. "
                    "Call register_from_model(model) or register_num_heads(layer_name, num_heads) "
                    "before running attribution."
                )

            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"AttentionHeadConcept: hidden_dim ({hidden_dim}) not divisible by "
                    f"num_heads ({num_heads}) for layer '{layer_name}'. "
                    "This likely means a gradient with an unexpected layout reached the mask "
                    "(e.g. attention weights instead of the projected output). "
                    "Verify registration via register_from_model() or register_num_heads()."
                )

            head_dim = hidden_dim // num_heads

            mask = torch.zeros_like(grad[batch_id])

            for head_id in concept_ids:
                if head_id < 0 or head_id >= num_heads:
                    raise IndexError(
                        f"Attention head index {head_id} out of range for layer '{layer_name}' "
                        f"with {num_heads} heads."
                    )
                start = head_id * head_dim
                end = (head_id + 1) * head_dim
                mask[..., start:end] = 1

            grad[batch_id] = grad[batch_id] * mask

            return grad

        return mask_fct

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):

        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        num_heads = self._resolve_num_heads(layer_name) if layer_name else None

        if num_heads is not None and relevance.dim() == 3:
            batch, seq_len, hidden_dim = relevance.shape
            head_dim = hidden_dim // num_heads
            rel_l = relevance.view(batch, seq_len, num_heads, head_dim).sum(dim=(1, 3))
        else:
            rel_l = torch.sum(relevance.view(*relevance.shape[:2], -1), dim=-1)

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l
