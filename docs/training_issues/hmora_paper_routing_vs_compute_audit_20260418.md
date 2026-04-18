# HMoRA Paper Audit: Routing Sparsity vs Compute Sparsity

Date: 2026-04-18

Status: background audit note.
Open first: [moe_lora_runtime_research_synthesis_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/moe_lora_runtime_research_synthesis_20260419.md)

## Question

Is HMoRA's current "routing-sparse but compute-dense" behavior documented in the authors' paper, and how does that compare to the checked-in implementations of `HMoRA`, `AdaMoLE`, and `MoELoRA` in this repo?

## Short Answer

Partially, but not cleanly.

The HMoRA paper clearly describes:

- soft routing as activating all experts
- top-k routing as a sparse-activation mechanism

However, the paper text does not clearly separate:

- logical routing sparsity
- actual kernel-level compute sparsity

In the checked-in code, both the standalone `HMoRA/*` implementation and the integrated `LLaMA-Factory` HMoRA tuner are routing-sparse but compute-dense. They zero out non-top-k gate weights, but they still execute the packed expert projections through dense matrix multiplies.

By contrast:

- `AdaMoLE` is token-level compute-sparse in the implementation, using selected-expert dispatch with per-expert gather/scatter
- `MoELoRA` is batch/task-routed sparse in the implementation, using per-expert loops over active batch items

## What The Paper Says

The relevant statements are in:

- the main HMoRA architecture discussion in Section 3
- Appendix A on routing methods

The paper defines the mixture-of-LoRA-experts block as:

- `Y = sum_i g_i E_i`
- `Z = XW + Y`

where expert outputs are weighted by router gates and added to the frozen dense layer output.

In Appendix A, the paper distinguishes:

- soft routing: `g = softmax(h(X))`
- top-k routing: `g = softmax(KeepTopK(softmax(h(X)), k))`

The paper explicitly states that:

- soft routing "requires the activation of all experts"
- top-k routing keeps only the top-k experts and sets the rest to zero
- "Sparse activation is achieved by activating only experts with non-zero gate values"

That wording strongly suggests sparse execution, not just sparse weights.

At the same time, the paper's core MoRA equations remain written as a weighted sum over expert outputs. The paper does not give a low-level execution description showing whether non-selected experts are skipped in actual computation or only zero-weighted after a packed projection.

So the paper supports this interpretation:

- intended semantics: sparse activation under top-k routing
- implementation-level compute story: unspecified

## What The Standalone HMoRA Code Does

In [HMoRA/hmora/model.py](/local/joeldag/repos/moe-study/HMoRA/hmora/model.py), the standalone implementation follows the same packed-dense execution pattern as the integrated tuner.

Router behavior:

- `TokenRouter.forward(...)` computes softmax gate values
- optional top-k routing replaces non-top-k logits with `-inf`, then re-softmaxes

MoRA execution:

- `MoRa.forward(...)` first projects through `mora_a`
- reshapes or expands into `(num_experts, rank)`
- multiplies by `gate.unsqueeze(-1)`
- flattens back into a packed expert dimension
- applies one dense `F.linear(..., self.mora_b)`

This means:

- top-k affects which expert dimensions get non-zero weights
- but the heavy A/B projections are still executed as dense packed linear algebra
- non-selected experts are not skipped as separate compute units

So the standalone HMoRA code is:

- logically sparse in routing
- not physically sparse in expert execution

## What The Integrated HMoRA Tuner Does

The integrated tuner in [LLaMA-Factory/src/peft/tuners/hmora/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/hmora/layer.py) matches the same pattern.

Routing:

- `TokenRouter.forward(...)` computes router weights
- optional top-k routing masks non-top-k experts and re-normalizes

Adapter forward:

- `LinearHMoRaLayer.forward(...)` computes `hidden_states = F.linear(dropped, lora_A.weight)`
- reshapes or expands to `(num_experts, rank)`
- multiplies by `gate.unsqueeze(-1)`
- flattens
- applies one dense `F.linear(..., lora_B.weight)`

So the integrated HMoRA implementation is also:

- routing-sparse
- compute-dense

This matches the throughput behavior observed earlier in the repo study:

- changing `top_k` changes routing semantics
- throughput stays relatively stable because the main compute remains dense

## Comparison To AdaMoLE

`AdaMoLE` in [LLaMA-Factory/src/peft/tuners/adamole/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/adamole/layer.py) is materially different.

It performs:

- token-level gate softmax
- token-level thresholding
- expert selection mask construction
- per-expert token indexing
- per-expert LoRA execution on selected token rows
- scatter-style accumulation back into the output tensor

That means `AdaMoLE` is compute-sparse in the implementation, but the sparsity comes with irregular execution:

- gather/scatter overhead
- small expert-specific sub-batches
- poorer GPU kernel utilization

This is why `AdaMoLE` can be slower than dense-packed HMoRA even when it activates fewer experts logically.

## Comparison To MoELoRA

`MoELoRA` in [LLaMA-Factory/src/peft/tuners/moelora/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/moelora/layer.py) is also different from HMoRA.

It uses:

- batch/task-level routing weights
- optional top-k gating on expert logits
- per-expert loops
- conditional execution on active batch items

Unlike HMoRA, it does skip inactive experts in the loop. But unlike AdaMoLE, the routing is not token-level in the integrated implementation. It is batch-level, then broadcast across sequence positions inside the adapted layer.

So its execution pattern is:

- logically sparse
- compute-sparse
- batch-routed rather than token-routed

## Practical Summary

The checked-in repo implementations currently fall into these buckets:

| Method | Routing granularity | Logical sparsity | Compute sparsity | Main execution pattern |
|---|---|---:|---:|---|
| `LoRA` | none | no | no | one dense low-rank A/B path |
| `HMoRA` | token-level or hybrid routing | yes | no | packed dense A/B with gate masking |
| `AdaMoLE` | token-level | yes | yes | per-expert token dispatch and scatter-add |
| `MoELoRA` | batch/task-level | yes | yes | per-expert batch dispatch |

## Main Conclusion

The HMoRA paper supports sparse-routing semantics, but it does not cleanly document the implementation distinction between:

- sparse expert selection
- sparse expert computation

The standalone HMoRA code and the integrated HMoRA tuner both resolve that ambiguity in the same direction:

- HMoRA is routing-sparse
- HMoRA is not compute-sparse

So the current repo behavior is not a contradiction of the paper's high-level routing story, but it is more specific than the paper text. The code implements top-k as sparse weighting inside a packed dense expert path, not as true expert skipping.

That makes HMoRA qualitatively different from both `AdaMoLE` and `MoELoRA`, which do perform sparse expert execution in the checked-in implementations.
