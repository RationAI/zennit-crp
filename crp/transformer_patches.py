"""
Transformer LRP patches for zennit-crp.

Ported from LRP-eXplains-Transformers (LXT) by Achtibat et al.
Paper: AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers. ICML 2024.
https://arxiv.org/abs/2402.05602
"""
from functools import partial
from warnings import warn

import torch
from torch.autograd import Function


# ─── LRP Rules ───────────────────────────────────────────────────────────────


class _IdentityRuleFn(Function):
    """Identity rule (Eq. 9 in AttnLRP paper) via gradient*input framework.
    Used on element-wise non-linear activations.
    """

    @staticmethod
    def forward(ctx, fn, input, epsilon=1e-10):
        output = fn(input)
        if input.requires_grad:
            ctx.save_for_backward(output / (input + epsilon))
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        gradient = ctx.saved_tensors[0] * out_relevance[0]
        return None, gradient, None


class _DivideGradientFn(Function):
    """Uniform rule (Eq. 7 in AttnLRP paper) via gradient*input framework.
    Used after matmul or element-wise multiplication.
    """

    @staticmethod
    def forward(ctx, input, factor=2):
        ctx.factor = factor
        return input

    @staticmethod
    def backward(ctx, *out_relevance):
        return out_relevance[0] / ctx.factor, None


def identity_rule_implicit(fn, input):
    """Apply fn(input) with identity LRP rule in the backward pass."""
    return _IdentityRuleFn.apply(fn, input)


def divide_gradient(input, factor=2):
    """Identity in forward; divide relevance by factor in backward."""
    return _DivideGradientFn.apply(input, factor)


def stop_gradient(input):
    """Stop gradient flow (used in CP-LRP)."""
    return input.detach()


# ─── Patch Utilities ──────────────────────────────────────────────────────────


def _check_already_patched(target_fn, new_fn):
    if target_fn.__module__ == new_fn.__module__:
        name = getattr(target_fn, "__name__", str(target_fn))
        warn(f"{name} already patched.")
        return True
    return False


def patch_method(fn, module, method_name="forward", keep_original=False):
    """Replace a method on a class with fn.

    Parameters
    ----------
    fn : callable
    module : class or module
    method_name : str
    keep_original : bool
        If True, saves original as ``original_<method_name>``.

    Returns
    -------
    bool
        True if patched successfully.
    """
    if _check_already_patched(getattr(module, method_name), fn):
        return False
    if keep_original:
        setattr(module, f"original_{method_name}", getattr(module, method_name))
    setattr(module, method_name, fn)
    return True


def replace_module(patched_module, original_module):
    """Copy all non-dunder attributes from patched_module onto original_module."""
    if original_module == patched_module:
        return False
    for attr in dir(patched_module):
        if not attr.startswith("__"):
            setattr(original_module, attr, getattr(patched_module, attr))
    return True


# ─── AttnLRP Forward Patches ─────────────────────────────────────────────────


def rms_norm_forward(self, hidden_states):
    """Identity rule for RMSNorm — stop gradient through variance."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * stop_gradient(
        torch.rsqrt(variance + self.variance_epsilon)
    )
    return self.weight * hidden_states.to(input_dtype)


def layer_norm_forward(self, x):
    """Identity rule for LayerNorm — stop gradient through std."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    y = (x - mean) / stop_gradient(std)
    if self.weight is not None:
        y *= self.weight
    if self.bias is not None:
        y += self.bias
    return y


def gated_mlp_forward(self, x):
    """Identity rule on activation + uniform rule on element-wise multiplication."""
    gate_out = self.gate_proj(x)
    gate_out = identity_rule_implicit(self.act_fn, gate_out)
    weighted = gate_out * self.up_proj(x)
    weighted = divide_gradient(weighted, 2)
    return self.down_proj(weighted)


def mlp_forward(self, x):
    """Identity rule on activation for standard (non-gated) MLP."""
    up_out = self.up_proj(x)
    up_out = identity_rule_implicit(self.act_fn, up_out)
    return self.down_proj(up_out)


def non_linear_forward(self, x):
    """Identity rule for standalone non-linear modules (e.g. GELU, ReLU)."""
    return identity_rule_implicit(self.original_forward, x)


def dropout_forward(self, x):
    """Disable dropout during LRP (required when model is in train mode)."""
    return x


def patch_attention(module):
    """Patch all attention functions in module with uniform rule on Q, K, V."""
    new_fwd = wrap_attention_forward(module.eager_attention_forward)
    if _check_already_patched(module.eager_attention_forward, new_fwd):
        return False
    module.eager_attention_forward = new_fwd

    new_map = {}
    for key, value in module.ALL_ATTENTION_FUNCTIONS.items():
        wrapped = wrap_attention_forward(value)
        if _check_already_patched(value, wrapped):
            return False
        new_map[key] = wrapped
    module.ALL_ATTENTION_FUNCTIONS = new_map
    return True


def wrap_attention_forward(forward_fn):
    def attention_forward(module, query, key, value, *args, **kwargs):
        query = divide_gradient(query, 4)
        key = divide_gradient(key, 4)
        value = divide_gradient(value, 2)
        if "dropout" in kwargs:
            kwargs["dropout"] = 0.0
        return forward_fn(module, query, key, value, *args, **kwargs)
    return attention_forward


# ─── CP-LRP Patches ───────────────────────────────────────────────────────────


def cp_gated_mlp_forward(self, x):
    """CP-LRP: stop gradient through gating before activation."""
    gate_out = stop_gradient(self.gate_proj(x))
    gate_out = self.act_fn(gate_out)
    weighted = gate_out * self.up_proj(x)
    return self.down_proj(weighted)


def patch_cp_attention(module):
    """Patch attention functions in module with CP-LRP (stop gradient on Q, K)."""
    new_fwd = cp_wrap_attention_forward(module.eager_attention_forward)
    if _check_already_patched(module.eager_attention_forward, new_fwd):
        return False
    module.eager_attention_forward = new_fwd

    for key, value in module.ALL_ATTENTION_FUNCTIONS.items():
        wrapped = cp_wrap_attention_forward(value)
        if _check_already_patched(value, wrapped):
            return False
        module.ALL_ATTENTION_FUNCTIONS[key] = wrapped
    return True


def cp_wrap_attention_forward(forward_fn):
    def cp_attention_forward(module, query, key, value, *args, **kwargs):
        query = stop_gradient(query)
        key = stop_gradient(key)
        if "dropout" in kwargs:
            kwargs["dropout"] = 0.0
        return forward_fn(module, query, key, value, *args, **kwargs)
    return cp_attention_forward


def cp_multi_head_attention_forward(self, query, key, value, *args, **kwargs):
    """CP-LRP patch for torch.nn.MultiheadAttention."""
    query = stop_gradient(query)
    key = stop_gradient(key)
    return self.original_forward(query, key, value, *args, **kwargs)


# ─── Gemma3-specific patch ────────────────────────────────────────────────────


def gemma3_norm_forward(self, x):
    """Identity rule for Gemma3RMSNorm._norm."""
    return x * stop_gradient(torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


# ─── GPT-2 MLP patch (inline, identity rule on activation) ───────────────────


def gpt2_mlp_forward(self, hidden_states):
    hidden_states = self.c_fc(hidden_states)
    hidden_states = identity_rule_implicit(self.act, hidden_states)
    hidden_states = self.c_proj(hidden_states)
    return hidden_states


# ─── Zennit BasicHook Patch ───────────────────────────────────────────────────


def _patched_basic_hook_forward(self, module, input, output):
    """Forward hook: save both input and output for backward use."""
    self.stored_tensors["input"] = input
    self.stored_tensors["output"] = output


def _patched_basic_hook_backward(self, module, grad_input, grad_output):
    """Backward hook: implement gradient*input LRP framework for transformers."""
    from zennit.core import ParamMod, stabilize

    assert len(grad_output) == 1, "Only single output supported for now!"
    grad_output = (grad_output[0] * self.stored_tensors["output"],)
    grad_output[0].requires_grad = True

    original_input = self.stored_tensors["input"][0].clone()
    inputs = []
    outputs = []
    for in_mod, param_mod, out_mod in zip(
        self.input_modifiers, self.param_modifiers, self.output_modifiers
    ):
        inp = in_mod(original_input).requires_grad_()
        with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
            out = modified.forward(inp)
            out = out_mod(out)
        inputs.append(inp)
        outputs.append(out)

    grad_outputs = self.gradient_mapper(grad_output[0], outputs)
    gradients = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        create_graph=grad_output[0].requires_grad,
    )
    relevance = self.reducer(inputs, gradients)
    relevance = relevance / stabilize(original_input, epsilon=1e-10)

    return tuple(
        relevance if original.shape == relevance.shape else None
        for original in grad_input
    )


def monkey_patch_zennit(verbose=False):
    """Patch zennit's BasicHook to support gradient*input LRP for transformers.

    1. Forward hook saves module output.
    2. Backward hook multiplies grad_output by saved output, then divides
       relevance by input to implement the gradient*input framework.
    """
    try:
        from zennit.core import BasicHook
    except ImportError:
        warn("'zennit' not available. Install it to use transformer LRP.")
        return

    for method, name in [
        (_patched_basic_hook_forward, "forward"),
        (_patched_basic_hook_backward, "backward"),
    ]:
        success = patch_method(method, BasicHook, method_name=name)
        if not success:
            warn(f"Failed to patch Zennit BasicHook's {name}. Skipping...")
        elif verbose:
            print(f"Patched Zennit BasicHook's {name}")


# ─── Default Patch Maps ───────────────────────────────────────────────────────


def _build_default_map():
    """Build a mapping from Python module → patch_map.
    Each entry is loaded lazily so missing optional deps don't break imports.
    """
    default_map = {}

    # ViT (torchvision) — always available if torchvision is installed
    try:
        import torch.nn as nn
        from torchvision.models import vision_transformer

        default_map[vision_transformer] = {
            nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
            nn.LayerNorm: partial(patch_method, layer_norm_forward),
            nn.MultiheadAttention: partial(
                patch_method, cp_multi_head_attention_forward, keep_original=True
            ),
        }
    except ImportError:
        pass

    # LLaMA
    try:
        from torch.nn import Dropout
        from transformers.models.llama import modeling_llama
        from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

        default_map[modeling_llama] = {
            LlamaMLP: partial(patch_method, gated_mlp_forward),
            LlamaRMSNorm: partial(patch_method, rms_norm_forward),
            Dropout: partial(patch_method, dropout_forward),
            modeling_llama: patch_attention,
        }
    except ImportError:
        pass

    # Qwen2
    try:
        from torch.nn import Dropout
        from transformers.models.qwen2 import modeling_qwen2
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm

        default_map[modeling_qwen2] = {
            Qwen2MLP: partial(patch_method, gated_mlp_forward),
            Qwen2RMSNorm: partial(patch_method, rms_norm_forward),
            Dropout: partial(patch_method, dropout_forward),
            modeling_qwen2: patch_attention,
        }
    except ImportError:
        pass

    # GPT-2
    try:
        from torch.nn import Dropout
        from transformers.models.gpt2 import modeling_gpt2
        from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

        default_map[modeling_gpt2] = {
            GPT2MLP: partial(patch_method, gpt2_mlp_forward),
            modeling_gpt2.nn.LayerNorm: partial(patch_method, layer_norm_forward),
            Dropout: partial(patch_method, dropout_forward),
            modeling_gpt2: patch_cp_attention,
        }
    except ImportError:
        pass

    # Gemma3
    try:
        from torch.nn import Dropout
        from transformers.models.gemma3 import modeling_gemma3
        from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3RMSNorm

        default_map[modeling_gemma3] = {
            Gemma3MLP: partial(patch_method, gated_mlp_forward),
            Gemma3RMSNorm: partial(
                patch_method, gemma3_norm_forward, method_name="_norm"
            ),
            Dropout: partial(patch_method, dropout_forward),
            modeling_gemma3: patch_attention,
        }
    except ImportError:
        pass

    return default_map


def get_default_map(module):
    """Return the default patch map for the given Python module.

    Raises ValueError for unsupported modules.
    """
    default_map = _build_default_map()
    if module in default_map:
        return default_map[module]
    supported = ", ".join(m.__name__ for m in default_map)
    raise ValueError(
        f"{module.__name__} not supported. Supported: {supported}. "
        "Provide a custom patch_map argument."
    )


# ─── Public API ───────────────────────────────────────────────────────────────


def monkey_patch(module, patch_map=None, verbose=False):
    """Patch a module's classes to implement LRP-aware forward passes.

    Parameters
    ----------
    module : Python module
        The module to patch (e.g. ``torchvision.models.vision_transformer``).
    patch_map : dict, optional
        Maps target classes to patch functions. Uses built-in defaults if None.
    verbose : bool
        Print each patched class name.
    """
    if patch_map is None:
        patch_map = get_default_map(module)

    for target, patch in patch_map.items():
        success = patch(target)
        if not success:
            name = getattr(target, "__name__", str(target))
            warn(f"Failed to patch {name}. Skipping...")
        elif verbose:
            name = getattr(target, "__name__", str(target))
            print(f"Patched {name}")
