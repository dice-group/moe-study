# AdaMoLE vs LoRA Batch Size Investigation

Date: 2026-04-18

Status: supporting investigation note for the broader MoE+LoRA runtime study bundle.
Open first: [moe_lora_runtime_research_synthesis_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/moe_lora_runtime_research_synthesis_20260419.md)

## Question

Why does `adamole` require a much smaller batch size than `lora` in the MoE study profile, even in the 10-language setup where AdaMoLE uses only 2 experts and the trainable parameter count is in the same order of magnitude?

## Short Answer

The batch size gap is explained much more by execution pattern and activation memory than by trainable parameter count.

`lora` adds one dense low-rank update per adapted module. `adamole` adds:

- multiple LoRA experts per adapted module
- a learned gate per token
- a learned threshold per token
- token-to-expert selection masks
- per-expert gathers/scatters in the forward pass
- retained routing state for the auxiliary loss

In the current AdaMoLE implementation, the 2-expert setup is also close to dense routing at initialization, so it behaves more like "run both experts for almost every token" than like sparse top-k MoE.

## Config Evidence

Shared 10-language training config:

- common LoRA targets are `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- see [configs/pipeline/profiles/moe_study_profile.json](/local/joeldag/repos/moe-study/configs/pipeline/profiles/moe_study_profile.json:55)

10-language run settings:

- `lora_r8`: `rank=8`, `alpha=16`, `batch_size=32`, `grad_accum=2`
- `adamole`: `rank=2`, `alpha=4`, `adamole_num_experts=2`, `adamole_max_threshold=0.25`, `batch_size=7`, `grad_accum=2`
- see [configs/pipeline/profiles/moe_study_profile.json](/local/joeldag/repos/moe-study/configs/pipeline/profiles/moe_study_profile.json:513)
- see [configs/pipeline/profiles/moe_study_profile.json](/local/joeldag/repos/moe-study/configs/pipeline/profiles/moe_study_profile.json:592)

So the observed discrepancy is real in the checked-in profile, not just a stale runtime artifact.

## Implementation Evidence

Plain LoRA uses a single dense low-rank path:

- one `A` projection and one `B` projection per adapted module
- one additive result path on top of the base layer output
- see [LLaMA-Factory/src/peft/tuners/lora/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/lora/layer.py:744)

AdaMoLE builds a more expensive path per adapted module:

- one `A/B` pair per expert
- one gate linear layer
- one threshold linear layer
- see [LLaMA-Factory/src/peft/tuners/adamole/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/adamole/layer.py:188)

AdaMoLE forward does the following for every token:

1. flatten tokens
2. compute gate softmax
3. compute thresholds
4. build `selected_experts`
5. renormalize selected weights
6. loop over experts
7. gather selected token rows for each expert
8. run expert LoRA on those rows
9. scatter-add expert outputs back into the result tensor

See [LLaMA-Factory/src/peft/tuners/adamole/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/adamole/layer.py:67).

This is much less efficient than the LoRA path because it replaces one regular dense adapter computation with token routing, indexing, masking, and per-expert sub-batches.

AdaMoLE also stores per-layer routing loss state for the trainer's auxiliary loss pass:

- layer loss is written in the AdaMoLE forward path
- see [LLaMA-Factory/src/peft/tuners/adamole/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/adamole/layer.py:88)
- losses are aggregated by `get_aux_loss`
- see [LLaMA-Factory/src/peft/tuners/adamole/model.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/adamole/model.py:105)
- trainer adds that term on every step when `adamole_aux_loss_coef > 0`
- see [LLaMA-Factory/src/llamafactory/train/sft/trainer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/llamafactory/train/sft/trainer.py:315)

## Why Trainable Params Are Misleading Here

For one adapted linear layer with input size `d_in`, output size `d_out`, LoRA rank `r`, and AdaMoLE experts `E`:

- LoRA params: `r * (d_in + d_out)`
- AdaMoLE params: `E * r * (d_in + d_out) + E * d_in + d_in + 1`

For the 10-language comparison values:

- LoRA uses `r=8`
- AdaMoLE uses `r=2`, `E=2`

That means AdaMoLE has roughly:

- expert weights equal to `4 * (d_in + d_out)`
- plus gate and threshold overhead of about `d_in * 3`

So AdaMoLE can have a similar trainable parameter count while still doing more runtime work, because:

- parameters only tell you model size
- they do not capture routing masks, gathers/scatters, irregular memory access, or expert fan-out

## Synthetic Benchmark

To isolate the adapter cost, a small synthetic benchmark compared one LoRA-style projection against one AdaMoLE-style projection using Llama-like attention dimensions:

- input size: `4096`
- output size: `4096`
- tokens per step: `8 * 512 = 4096`
- LoRA config: `r=8`, `alpha=16`
- AdaMoLE config: `r=2`, `alpha=4`, `num_experts=2`, `max_threshold=0.25`

Observed result:

- LoRA trainable params: `65,536`
- AdaMoLE trainable params: `45,057`
- LoRA mean step time: `0.0316s`
- AdaMoLE mean step time: `0.1525s`
- AdaMoLE slowdown: `4.82x`
- AdaMoLE average experts selected per token: `1.98 / 2`

Interpretation:

- even with fewer trainable params than LoRA, AdaMoLE was about 4.8x slower in this controlled adapter-only test
- the 2-expert / `max_threshold=0.25` setup routed almost every token to both experts
- in practice this is close to dense expert execution, not sparse routing

This synthetic benchmark is not a pipeline-faithful throughput number, but it strongly supports the implementation-level explanation.

## Main Conclusion

The current AdaMoLE batch size is not surprising.

The biggest reasons are:

- AdaMoLE runs routing logic per token for every adapted module
- the current implementation uses expert loops with token indexing and scatter-adds
- the 2-expert setup is almost dense at initialization, so both experts usually run
- routing tensors and aux-loss state increase activation/backward cost
- common trainable-parameter summaries do not reflect those runtime costs

## Practical Implication

If AdaMoLE throughput or batch size needs improvement, the first things to examine are not parameter count but:

- routing sparsity in actual runs, especially `train/adamole/experts_per_token`
- `adamole_max_threshold`
- whether the implementation can avoid the per-expert gather/scatter pattern
- whether the adapted module set should be narrower for AdaMoLE than for plain LoRA

As configured today, `adamole` is closer to a dense multi-expert adapter than to a cheap sparse extension of LoRA, so a much smaller batch size than `lora` is expected.
