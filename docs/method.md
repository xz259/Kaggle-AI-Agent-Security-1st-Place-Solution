# Method

## Objective

Let `x` be the mutable prompt tokens. Bootstrap records the greedy hop-1 output

```math
y^{(1)}=(y_1,\ldots,y_T)
```

from llama.cpp. The search then tries to make a chosen EOG token `e` win at the
first hop-2 position:

```math
m_2(x)=z_e(x)-\max_{v\ne e}z_v(x).
```

The primary objective is to increase `m_2`. A solution requires `m_2 > 0` and a
free greedy replay whose first sampled hop-2 token is `e`.

## Hop-1 preservation

For each hop-1 position, the GGUF scorer teacher-forces the observed prefix and
checks

```math
z_{y_t}(x)>\max_{v\ne y_t}z_v(x).
```

Teacher forcing is followed by a free greedy decode. A proposal is rejected if
either check differs from the bootstrapped token sequence. Hop 1 is therefore a
hard constraint rather than a soft term that can be traded against hop 2.

The hop-2 template must represent the context produced by the real framework,
not merely a concatenation of raw generations. Tool runtimes often parse hop 1
and re-render a canonical structured call before appending the tool result. The
template can therefore embed fixed canonical history and omit `{{HOP1}}`; the
only required placeholder is the exact mutable `{{PROMPT}}` span.

## Gradient proposals

The GGUF does not expose gradients. A matching upstream Transformers checkpoint
computes the differentiable proxy loss

```math
\ell(x)=\operatorname{softplus}
\left(z_{v^*}(x)-z_e(x)\right),
\qquad
v^*=\arg\max_{v\ne e}z_v(x).
```

For mutable position `i`, HotFlip ranks replacement token `w` by the first-order
change

```math
\nabla_{E(x_i)}\ell\;\cdot\;\left(E(w)-E(x_i)\right).
```

The proxy ranks proposals only. It never accepts a prompt or supplies reported
final margins.

## Candidate sampling

At each coordinate, candidates are sorted best to worst according to the proxy.
For rank `r=1,...,K`, the sampler uses

```math
P_\tau(r)=
\frac{(K+1-r)^{1/\tau}}
{\sum_{j=1}^{K}(K+1-j)^{1/\tau}}.
```

Coordinates are sampled uniformly after an initial coverage pass. Duplicate
prompt tuples are excluded within and across resumed panels.

## Optional GGUF-calibrated ridge reranking

The BF16 ordering can transfer imperfectly to a quantized model. With
`search.ridge.enabled = true`, the search uses an initial allocation of the
same candidate budget to collect local GGUF labels for ordinary radius-1
proposals. Each observation records the coordinate, gradient rank, HotFlip
delta, GGUF target-NLL change, and GGUF hop-2 margin gain relative to the
incumbent.

Two ridge regressions are fit using the standardized HotFlip delta, normalized
rank, their interaction, a squared-delta term, and coordinate-specific
intercepts and slopes. One predicts NLL change and the other predicts margin
gain. The search interleaves both predicted orderings while reserving a
configurable fraction for the Faster-GCG rank sampler.

This is a shortlist policy, not a surrogate acceptance rule. Every proposed
candidate is still scored by the GGUF backend, and promotion still requires
the clean full-context score plus exact teacher-forced and greedy hop-1 gates.
The fit is deliberately local: observations persist across plateau panels but
are cleared immediately after an accepted token replacement.

## Two GGUF scoring paths

All candidates are first scored by llama.cpp while reusing the byte-identical
prefix before `{{PROMPT}}`. This reduces repeated long-prefix evaluation, but a
different forward split can slightly alter floating-point results.

The best cached-prefix candidates are therefore rescored from a clean, complete
context. Only this full score can improve the incumbent. The hop-1 teacher-forced
and greedy gates are also run from complete contexts.

## Deliberate omissions

The presentation implementation omits multi-token moves, evolutionary prompt
generation, distributed llama.cpp workers, and competition parsers. Keeping one
authoritative radius-1 path makes the proposal/validation boundary easier to
audit.
