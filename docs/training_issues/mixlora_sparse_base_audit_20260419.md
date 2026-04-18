# MixLoRA As A Sparse MoE+LoRA Base

Date: 2026-04-19

Status: background design note for the sparse MixLoRA prototype work.
Open first: [moe_lora_runtime_research_synthesis_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/moe_lora_runtime_research_synthesis_20260419.md)

## Question

Is `MixLoRA` a better base than `AdaMoLE` or `MoELoRA` for studying whether an optimized sparse MoE+LoRA execution path can approach or beat `HMoRA` on H100?

## Short Answer

Yes.

`MixLoRA` is the strongest sparse-execution base in this repo for a follow-on prototype because it already combines:

- token-level top-k routing
- auxiliary router load-balancing loss
- FFN-only expert scope
- explicit compute sharing across experts for the frozen FFN path

That makes it much closer to a plausible high-performance sparse design than `AdaMoLE`, which dispatches every expert path independently at token granularity, and more structurally aligned with dense-packed `HMoRA` than `MoELoRA`, which uses a different routing granularity and module scope.

## What The MixLoRA Paper Claims

The MixLoRA paper presents the method as:

- a sparse MoE model based on LoRA
- using a top-k router
- with auxiliary load-balancing loss
- plus a "high-throughput framework" that reduces compute and memory cost

In the paper text and figures:

- experts live inside the FFN block
- tokens are routed to top-k experts
- experts share the frozen FFN weights
- LoRA adapters carry the expert-specific updates

The paper explicitly frames the optimization as reducing the computational complexity of expertized FFN execution by reusing shared FFN computation instead of recomputing the whole FFN per expert.

## What The Integrated Repo Implementation Does

The integrated tuner in [mixlora/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/mixlora/layer.py) matches that high-level story quite well.

Important implementation properties:

- `MixLoraMoeLayer` replaces whole Llama-style MLP blocks, not isolated linear layers
- the router is token-level and uses top-k expert selection
- router auxiliary loss is exposed through `get_aux_loss()` in [mixlora/model.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/mixlora/model.py)
- the frozen `gate_proj` and `up_proj` are computed once for all tokens
- only the expert-specific LoRA deltas are dispatched per selected expert

Current compute pattern in `_compute_expert_outputs(...)`:

1. compute shared frozen `gate_proj` once
2. compute shared frozen `up_proj` once
3. loop over experts
4. gather only tokens assigned to that expert
5. add expert-specific LoRA deltas for `gate_proj` and `up_proj`
6. apply activation and shared frozen `down_proj`
7. optionally add expert-specific `down_proj` LoRA delta
8. weight by router score and `index_add_` back to the output

This is already a meaningful optimization relative to a naive sparse MoE+LoRA implementation.

## Why It Is A Better Sparse Base

Compared with the other routed methods already analyzed in this repo:

`HMoRA`
- routing-sparse, compute-dense
- very GPU-friendly
- excellent control baseline for "how good can dense-packed execution be"

`AdaMoLE`
- token-level sparse dispatch
- gathers and scatters full expert paths directly
- expensive and irregular execution pattern

`MoELoRA`
- sparse execution too
- but routing is batch/task-level in the integrated implementation
- not as close to the Mixtral-style token-routed FFN MoE story

`MixLoRA`
- token-level sparse dispatch
- FFN-only expert scope
- shared frozen FFN compute
- top-k router + aux loss already in place

So `MixLoRA` is the cleanest starting point for testing whether sparse expert execution can be made materially more efficient without giving up true token-level expert selection.

## Main Limitation Of Current MixLoRA

Even though the current MixLoRA implementation already shares frozen FFN work, it still performs expert dispatch through:

- expert-level loops
- repeated token gathers
- repeated per-expert small-batch LoRA delta application
- `index_add_` accumulation back into the flat token output

That means it is still vulnerable to the same core sparse-execution problem seen in other methods:

- logical sparsity does not automatically translate into high GPU efficiency

This is exactly the gap the next prototype should target.

## Why This Deepens MoE+LoRA Understanding

Using MixLoRA as the next research base is useful because it isolates the right question:

- not "can sparse routing help"
- but "can true sparse execution be made efficient enough to compete with dense-packed HMoRA-style execution"

That is a much sharper systems question than continuing only with `AdaMoLE`, because MixLoRA starts from a stronger sparse baseline and already contains one real optimization from the paper.

## Practical Conclusion

If the next step is a research prototype rather than immediate pipeline integration, `MixLoRA` is the right place to start.

The most informative comparison is:

- `lora_ffn_dense`
- `hmora_ffn_dense_packed`
- `mixlora_current`
- `mixlora_sparse_grouped_v2`

This keeps the experiment focused on FFN expert execution, where MixLoRA is already strongest and where a sparse optimization has the best chance of closing the gap to `HMoRA`.
