# The AH-rule (CP-LRP attention-head rule)

Ali et al., *XAI for Transformers: Better Explanations through Conservative
Propagation*, ICML 2022 — [arXiv:2202.07304](https://arxiv.org/html/2202.07304),
[PMLR v162](https://proceedings.mlr.press/v162/ali22a.html).

## Setup — what an attention head computes

The paper formalizes the head (§3.1 *Propagation in Attention Heads*, Eq. 6) as,
per output token $j$:

$$
y_j \;=\; \sum_i x_i\, p_{ij}
$$

with $x_i$ the value tokens and $p_{ij} = \operatorname{softmax}_i(q_j
k_i^\top/\sqrt{d})$ the attention weights. The head is **bilinear**: $p_{ij}$
depends on the input too (through Q and K), while LRP's standard rules assume a
(locally) linear layer.

## The problem — conservation breaks

- **Prop. 3.1**: propagating relevance through the attention head with standard
  gradient-based LRP breaks conservation (a covariance term between the softmax
  input and the queries appears).
- **Prop. 3.2**: LayerNorm likewise never satisfies conservation (relevance
  collapse through the normalization).

## The rule

§4 *Better LRP Rules for Transformers*:

> "Specifically, we propose at explanation time to make a locally linear
> expansion of the attention head by viewing the gating terms $p_{ij}$ as
> constants."

With $p$ frozen, the head is an ordinary linear layer with weights $p_{ij}$, and
the ε-rule applied to it is the **AH-rule**:

$$
\mathcal{R}(x_i) \;=\; \sum_j
\frac{x_i\, p_{ij}}{\sum_{i'} x_{i'}\, p_{i'j}}\;
\mathcal{R}(y_j)
\tag{AH-rule}
$$

Each output token's relevance is redistributed to the value tokens in
proportion to their contribution $x_i p_{ij}$ to $y_j$ (the denominator *is*
$y_j$, so the shares sum to one → conservation, modulo ε).

**Query/key receive nothing** — by construction:

> "With such a reformulation, we also note that the query sequence $x'$ appears
> disconnected, and consequently, we have implicitly $\mathcal{R}(x'_j) = 0$."

All relevance flows through the **value path** — hence *conservative
propagation*. The rule answers "which tokens carried the content", deliberately
not "why was attention paid there".

The companion **LN-rule** (same section) applies the same freeze to LayerNorm's
normalization:

$$
\mathcal{R}(x_i) \;=\; \sum_j
\frac{x_i\,(\delta_{ij} - \tfrac{1}{N})}{\sum_{i'} x_{i'}\,(\delta_{i'j} - \tfrac{1}{N})}\;
\mathcal{R}(y_j)
\tag{LN-rule}
$$

## Implementation — one detach, no explicit formula

§4, implementation trick:

> "In practice, these rules do not need to be implemented explicitly … rewrite
> [the attention head with $p_{ij}$ detached, and every LayerNorm with the
> normalization detached]. Then extract the LRP explanation by calling
> Gradient×Input on the resulting function $f$."

I.e. detach Q/K (and LayerNorm's $\sigma$), then **Gradient×Input realizes the
AH-rule exactly**. In the g-convention ($R = g \odot x$ at every tensor), g is
shared through the frozen matmul, so:

$$
\mathcal{R}(x_i) \;=\; x_i \sum_j p_{ij}\, g_j
\;=\; x_i \sum_j p_{ij}\, \frac{\mathcal{R}(y_j)}{y_j}
\;=\; \sum_j \frac{x_i\, p_{ij}}{\sum_{i'} x_{i'}\, p_{i'j}}\, \mathcal{R}(y_j)
$$

— the AH-rule term for term, ε-free. Note the paper itself prescribes the
Gradient×Input read-out: the grad×input convention is not an LXT invention but
the paper's own extraction recipe.

## Why plain autograd performs the ε-rule (the part that looks like magic)

Autograd does compute gradients — nothing else. The trick is **what quantity
the backward stream is declared to carry**, plus a **rewritten forward graph**.

**The identity.** Take one linear op $y = Wx$ (or $y=Wx+b$). The ε-rule is

$$
R_i \;=\; \sum_j \frac{x_i\, w_{ij}}{y_j + \varepsilon}\, R_j .
$$

Define $g_j := R_j / (y_j + \varepsilon)$ and pull $x_i$ out:

$$
R_i \;=\; x_i \sum_j w_{ij}\, g_j \;=\; x_i \cdot [W^\top g]_i .
$$

$W^\top g$ is **exactly the autograd VJP** of the linear op applied to $g$. So:
if the backward stream carries $g$ instead of $R$, then *for a linear op* the
ε-rule is literally

$$
\text{ε-rule} \;=\; \text{plain backward pass} \;\times\; \text{input}.
$$

No rule hook needed. And the invariant propagates: the VJP output is again a
valid $g$ for the layer below ($R^{\text{below}} = g^{\text{below}} \odot
x^{\text{below}}$), by the same algebra. Seed the output with $g=1$
(`logit.backward()`), read any layer as $g \odot x$ — ε-LRP everywhere, by
induction. Bias handled free: the VJP ignores $b$ (constant), the denominator
$y_j$ contains it → the bias share is silently absorbed, exactly the ε-rule's
bias-absorb behaviour.

**The catch: only for linear ops.** A real transformer is not linear —
softmax, LayerNorm's $1/\sigma$, GELU. This is where the *forward rewrites*
come in: freeze every nonlinearity into a constant coefficient
(`p.detach()` in attention, $\sigma$`.detach()` in LN, GELU as
$x\cdot(y/x)$`.detach()`). The rewritten graph **is piecewise linear with
frozen coefficients**, so the identity above applies to the entire network.
That is precisely the paper's recipe: rewrite, then "extract the LRP
explanation by calling Gradient×Input".

### Where is the g-substitution in the code? Nowhere — and that's the point

Nothing modifies the linear layers. No hook, no rewrite, no gradient surgery.
The "substitution" $g = R/(y+\varepsilon)$ exists only **on paper**, in the
proof that connects the two conventions. In code, three facts conspire:

1. **The backward stream is whatever you seed.** `logit.backward()` seeds the
   output gradient with $1$. Interpreted in the R-convention that seed is
   $g_{\text{out}} = R_{\text{out}}/y_{\text{out}} = y/y = 1$ — consistent by
   construction. From there autograd only ever *composes VJPs*; the tensor
   flowing backward simply **is** $g$ at every layer. Nobody divides anything.
2. **Hidden-layer $R$ is never computed.** Only where you read out do you form
   $R = g \odot x$ (i.e. `x.grad * x`). One multiplication, at the end.
3. **The linear VJP is already the ε-rule's redistribution matrix.** Given a
   correct incoming $g$, $W^\top g$ *is* the bracket of the ε-rule — see the
   identity above. So the correct move on Linear/Conv is to do *nothing*.

### Where did the ε go?

The ε in the classic rule exists to stabilise the **division** $R_j/(y_j +
\varepsilon)$. The g-framework never performs that division on linear ops —
$g$ arrives by VJP composition, not by dividing $R$ by $y$. So on the linear
chain the g-form equals the ε-rule **in the exact limit $\varepsilon \to 0$**,
division-free and therefore stable. The classic ε>0 rule differs from it by
$O(\varepsilon)$ — measurably: an explicit ε-add rule (ε=1e-6) vs rule-free
g autograd differed by ~1e-5 on ViT-B/16, exactly the stabiliser (measured in
an earlier revision of `lxt_reference.ipynb`; the notebook now runs rule-free
adds, the paper-faithful ε→0 form).

ε survives only where real divisions remain: the nonlinearity rewrites (GELU's
gain $y/(x+\varepsilon)$) and inside boundary-converted non-ε rules (Gamma's
stock machinery, the adapter's $R/\mathrm{stabilize}(x)$ exit).

### The full recipe (what you actually do)

```python
# 1. REWRITE the nonlinear ops — the one thing that "happens to the model":
#    p.detach() in attention, sigma.detach() in LayerNorm,
#    GELU as x * (f(x)/x).detach()          ← canonizers do exactly this
# 2. Leave Linear / Conv completely untouched.
x.requires_grad_(True)
out = model(x)
out[0, target].backward()      # 3. seed g_out = 1
R = x.grad * x                 # 4. read out relevance ONCE, at the input
```

That is ε-LRP (bias-absorb, ε→0) for the entire rewritten network. Step 1 is
why the model's gradient "changes" — not on the linear layers, but by freezing
every nonlinear coefficient so the graph the VJPs traverse is linear.

### Materializing R, and how R-space rules embed in the g-stream

The stream is g end-to-end; **R exists only where you read it**, by one
elementwise multiply against that tensor's stored activation:

* input heatmap: `R = x.grad * x`;
* intermediate site (CRP probe): keep the activation (forward hook /
  `retain_grad()`), then `R_layer = grad_at_tensor * activation`. A readout,
  side-effect-free — the stream continues as g underneath.

A **relevance-space rule** (γ, αβ, ratio, …) consumes R and produces R. It
embeds via a three-step boundary conversion — literally what
`GradTimesInputMultiInputBasicHook` (and LXT's `monkey_patch_zennit`) does:

```
        g_out                        ← autograd hands the hook grad_output
          │
entry:  R_out = g_out ⊙ y            ← y stored at forward
          │
rule:   R_in  = Rule(R_out)          ← stock zennit machinery, unmodified
          │
exit:   g_in  = R_in / stabilize(x)  ← x stored at forward
          │
        g_in returned as grad_input  → autograd continues, oblivious
```

Both conversions are exact by the convention's definition ($R = g \odot x$ at
every tensor); the only cost is the exit division's `stabilize(x, 1e-10)` —
the one place ε re-enters.

The paradigm, layered:

1. **ε-behaviour ops** (linears under ε, residual adds, frozen
   nonlinearities): no hook — the VJP *is* the rule.
2. **non-ε rules**: boundary-converted hook
   (`GradTimesInputMultiInputBasicHook`).
3. **readouts** (input heatmap, CRP probes): g ⊙ activation, anywhere, any
   number of times.

**So what is zennit still for?** The ε-rule is the *one* rule that coincides
with gradient×input on a linear op. Anything else — γ, αβ, ZBox, flat — needs
a *different* denominator or modified weights, which no forward rewrite plus
plain gradient produces. Those still need hooks; in the g-convention they must
convert at the boundary ($g\!\to\!R$, apply rule, $R\!\to\!g$) — that is
exactly what `GradTimesInputMultiInputBasicHook` / LXT's `monkey_patch_zennit` do around
zennit's `Gamma`. Zennit is not made redundant; it is only *unnecessary for
the ε-parts* once the graph is rewritten.

### CRP conditional machinery under the g-convention

CRP's masking does **not** require an LRP rule Hook with stored relevance:
`crp.hooks.MaskHook` registers a *tensor-level* backward hook on the module
output and multiplies whatever flows backward by the concept mask — any named
module, rule or no rule (the gallery conditions on `proj_drop` under ``Pass``
and on bare block outputs today). Because $R = g \odot x$ elementwise, masking
the g-stream has identical support and effect to masking R — conditional flow
$\theta = \{y{:}\,\text{class},\, l{:}\,\text{concept}\}$ carries over
unchanged.

What does NOT carry over: the **readouts**. Under g,
``res.relevances[l_name]`` holds g (ranking by ``concept.attribute`` would sum
g — meaningless) and ``res.heatmap`` is ``data.grad`` alone. Both need the one
materialization multiply: $R = g \odot \text{activation}$ at the probe,
``data.grad * data`` at the input. Porting CRP onto a g-convention composite is
therefore a thin readout adapter, not a rewrite.

## Three equivalent realizations in this repo

| where | mechanism |
|---|---|
| `zennit_extensions.cp_lrp.StopGradient` on `Q/KInspectionLayer` (timm, unfolded attention; `CPLRPComposite`) | rule zeroes the Q/K backward → $p$ constant |
| `TorchvisionEncoderBlockCanonizer` probes + `StopGradient` (`tutorials/vit_crp/lxt_reference.ipynb`, Path A) | same, on torchvision's fused MHA |
| LXT `cp_multi_head_attention_forward` (Path B) | `query.detach(), key.detach()` before the fused forward |

All three: "make $p$ a constant, let autograd do the ε-rule." Bit-level
equivalence of the last two is verified in the notebook
($|R_A - R_B| = 0$).

## Contrast with AttnLRP

AttnLRP (Achtibat et al., 2024) *does* attribute the attention-formation path —
its matmul rule (Eq. 15) and softmax rule (Prop. 3.1) conduct relevance into
Q/K. CP-LRP trades that away for exact conservation and cleaner value-path
heatmaps. That trade-off is precisely the `cp_lrp_baseline` vs
`attnlrp_baseline` comparison served by the CRP gallery.

## References

- Ali, Schnake, Eberle, Montavon, Müller, Wolf. *XAI for Transformers: Better
  Explanations through Conservative Propagation.* ICML 2022.
  [arXiv:2202.07304](https://arxiv.org/html/2202.07304) — head formalization
  §3.1 Eq. (6); conservation failures Props. 3.1–3.2; AH-rule + LN-rule and the
  Gradient×Input implementation trick §4.
- Achtibat et al. *AttnLRP: Attention-Aware Layer-Wise Relevance Propagation
  for Transformers.* ICML 2024.
  [PMLR v235](https://proceedings.mlr.press/v235/achtibat24a.html) — the
  contrasting attention-formation attribution (Eq. 15, Prop. 3.1).
