"""PA-LRP positional-sink rules (Bakish, Zimerman, Chefer & Wolf,
"Revisiting LRP: Positional Attribution as the Missing Ingredient for
Transformer Explainability", arXiv:2506.02138).

Equation <-> rule map:

    Eq. 5   -> :class:`PosEmbedSink`    (input-level PE: learnable add)
    Eq. 10  -> :class:`RotaryRopeSink`  (attention-level PE: RoPE)
    Eq. 8/9 -> the stash is the per-site flattened positional sink
    Eq. 11  -> downstream aggregation over the stashes (consumer code sums
              only positive entries over the feature dims and layers)

Both rules are pure AttnLRP-style allocation rules plus a positional sink:
the forward computation is untouched, relevance is distributed per the
equation, and the positional share is stashed (``.detach()``-ed) on the
module as ``module._palrp_sink`` (per-sample, batch dim preserved;
overwrite-per-backward semantics -- read right after the attribution, while
the composite context is active). Default recipes map NO rule here: PA-LRP
is an opt-in ``layer_map`` entry.
"""

import torch
from timm.layers import apply_rot_embed_cat

from zennit.core import Hook, stabilize


class PosEmbedSink(Hook):
    """Eq. 5. Input-level PE, ``z = embedded + positional`` (E_i + P'_i):

        R(P'_i) = P'_i * R(z_i) / (P'_i + E_i + eps)
        R(E_i)  = E_i  * R(z_i) / (P'_i + E_i + eps)

    the standard LRP-epsilon rule for addition (kernel identical to
    ``EpsilonAdd``, spelled out for direct verification). Attach to
    :class:`~zennit_extensions.attention_unfolded.PosEmbedAdd`.

    Stash: per-sample ``rel_P`` (B, N, D) as ``module._palrp_sink``.
    Returned: ``rel_E`` (B, N, D) for the token stream; for the broadcast
    ``pos_embed`` input the batch-summed ``rel_P.sum(0, keepdim=True)``
    (autograd broadcast contract -- inputs are shared over the batch), with
    the per-sample detail kept in the stash.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["embedded"] = args[0]      # E
        self.stored_tensors["positional"] = args[1]    # P'
        self.stored_tensors["z"] = output              # z = E + P'

    def backward(self, module, grad_input, grad_output):
        rel_z = grad_output[0]
        denom = self.stored_tensors["z"] + self.epsilon    # z + ε (Eq. 5, plain)
        rel_E = self.stored_tensors["embedded"] * rel_z / denom
        rel_P = self.stored_tensors["positional"] * rel_z / denom   # Eq. 5
        module._palrp_sink = rel_P.detach()
        return (rel_E, rel_P.sum(0, keepdim=True))

    def copy(self):
        return PosEmbedSink(self.epsilon)


class RotaryRopeSink(Hook):
    """Eq. 7-10. RoPE: ``Q_tilde_i = R_{i,k} Q_i`` (same for ``K_tilde``).
    The rotation matrix ``R_{i,k}`` (Eq. 7) mixes position into queries/keys;
    per Lemma 3.2 the relevance at each such mixing node splits uniformly,
    so the rotation receives one half from the Q_tilde node and one from the
    K_tilde node:

        R(R_{i,k}) = 1/2 R(Q_tilde_i) + 1/2 R(K_tilde_i)        (Eq. 10)

    Stash: one half of the OUTPUT-side per-sample relevance
    (``grad_output[0]``) -- literal Eq. 10, with batch dim preserved. The
    ``rope_q`` and ``rope_k`` stashes of one block sum to Eq. 10 per
    sample. (Eq. 8/9: the cat-form ``rope`` input holds the nonzero
    (cos, sin) entries of the block-diagonal R; the stash is that flattened
    positional sink, Eq. 11-aggregated downstream.)

    The NON-sink half continues down the Q (resp. K) path with the
    ε-contribution rule, matching the reference implementation
    (github.com/YardenBakish/PE-AWARE-LRP: LXT ``add2`` ε-split of
    ``q·cos + rot_half(q)·sin`` followed by the uniform ``mul2`` halving):

        R(Q) = Q ⊙ Jᵀ( R(Q_tilde) / (2 · (Q_tilde + ε)) )

    with J the rotation Jacobian. Summed per token this equals
    ``R(Q_tilde)/2`` as ε→0, so ``stash + R(Q) = R(Q_tilde)`` per token --
    the conservation property of Lemma 3.2. (A plain halved vanilla
    gradient ``Jᵀ R(Q_tilde) / 2`` would backward-rotate the relevance and
    break per-token conservation.)

    Prefix rows (cls+register tokens, the first
    ``module.num_prefix_tokens`` of token axis -2) are NOT rotated by RoPE:
    their gradients pass through unmodified and their stash rows are zero --
    otherwise the rule would shave half the relevance off tokens that carry
    no positional mixing. Identity path (``rope=None``): unmodified
    pass-through, stash ``None`` -- e.g. when the graph was deliberately cut
    via ``detach_rope=True``; never combine that with PA-LRP.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, module, args, kwargs, output):
        self.stored_tensors["q"] = args[0]
        self.stored_tensors["rope"] = args[1]
        self.stored_tensors["output"] = output

    def backward(self, module, grad_input, grad_output):
        rope = self.stored_tensors["rope"]
        if rope is None:
            module._palrp_sink = None
            # rope=None ⇒ only q is a tensor input; pass its grad through unchanged.
            return (grad_input[0],) if grad_input else None
        rot = slice(module.num_prefix_tokens, None)   # rotated rows, token axis -2
        grad_q, grad_rope = grad_input
        rel_out = grad_output[0]

        sink = torch.zeros_like(rel_out)
        sink[..., rot, :] = rel_out[..., rot, :] / 2    # 1/2 R(Q_tilde) / 1/2 R(K_tilde), Eq. 10
        module._palrp_sink = sink.detach()

        # ε-contribution propagation of the non-sink half: recompute the
        # rotation on detached inputs and pull s = R/(2·(out+ε)) back through
        # its Jacobian.
        q_rot = self.stored_tensors["q"][..., rot, :].detach().requires_grad_(True)
        rope_in = rope.detach().requires_grad_(grad_rope is not None)
        with torch.enable_grad():
            y_rot = apply_rot_embed_cat(q_rot, rope_in, half=module.rotate_half)
        out_rot = self.stored_tensors["output"][..., rot, :].detach()
        s = (rel_out[..., rot, :] / stabilize(2.0 * out_rot, self.epsilon)).to(y_rot.dtype)
        inputs = (q_rot, rope_in) if grad_rope is not None else (q_rot,)
        vjps = torch.autograd.grad(y_rot, inputs, grad_outputs=s)

        grad_q_out = None
        if grad_q is not None:
            grad_q_out = grad_q.clone()
            grad_q_out[..., rot, :] = q_rot.detach() * vjps[0]
        grad_rope_out = None if grad_rope is None else rope_in.detach() * vjps[1]
        return (grad_q_out, grad_rope_out)

    def copy(self):
        return RotaryRopeSink(self.epsilon)
