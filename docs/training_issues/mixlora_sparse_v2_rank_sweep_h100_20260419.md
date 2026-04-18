# MixLoRA Sparse-v2 High-Rank H100 Sweep

Date: 2026-04-19

Status: intermediate benchmark result.
Superseded by: [mixlora_sparse_v2_extended_scalability_h100_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/mixlora_sparse_v2_extended_scalability_h100_20260419.md)
Open first: [moe_lora_runtime_research_synthesis_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/moe_lora_runtime_research_synthesis_20260419.md)

## Question

Does sparse MixLoRA begin to close the runtime gap to dense-packed HMoRA when LoRA rank is increased, especially in the regime where sparse execution should have its best chance?

## Setup

Benchmark harness:

- [scripts/local/mixlora_sparse_v2_study.py](/local/joeldag/repos/moe-study/scripts/local/mixlora_sparse_v2_study.py)

Variants:

- `lora_ffn_dense`
- `hmora_ffn_dense_packed`
- `mixlora_current`
- `mixlora_sparse_grouped_v2`

Hardware:

- local `NVIDIA H100-80GB`

Main sweep:

- `device=cuda`
- `dtype=bfloat16`
- `pass_mode=forward_backward`
- `hidden_size=4096`
- `ffn_multiplier=4`
- `num_experts=32`
- `rank in {8, 16, 32, 64, 128}`
- `batch_size=8`
- `seq_len in {256, 1024}`
- controlled routing with active experts `{1, 2}`
- native routing with `top_k in {1, 2}`

Artifacts:

- `/tmp/mixlora_sparse_v2_rank_sweep_h100_20260419/results.jsonl`
- `/tmp/mixlora_sparse_v2_rank_sweep_h100_20260419/results.csv`

## Short Answer

Sparse MixLoRA improves with rank, but it does not beat HMoRA in the tested regime.

The most favorable result for sparse v2 was:

- `seq_len=1024`
- controlled routing
- `top_k=1`
- `rank=128`
- `mixlora_sparse_grouped_v2 / hmora_ffn_dense_packed = 0.914x`

So sparse v2 gets close in the best large-shape, high-rank, low-top-k case, but it still does not cross over.

## Main Findings

### 1. HMoRA remains the practical winner

For the large `seq_len=1024` shape:

- controlled `top_k=1`
  - `rank=8`: sparse v2 is `0.393x` of HMoRA
  - `rank=16`: `0.403x`
  - `rank=32`: `0.462x`
  - `rank=64`: `0.626x`
  - `rank=128`: `0.914x`

- controlled `top_k=2`
  - `rank=8`: `0.354x`
  - `rank=16`: `0.394x`
  - `rank=32`: `0.444x`
  - `rank=64`: `0.547x`
  - `rank=128`: `0.719x`

This is the clearest signal from the sweep:

- the gap narrows as rank grows
- the best-case sparse regime is low `top_k`
- but HMoRA still wins in all tested cases

### 2. Sparse v2 is a real improvement over current MixLoRA

For the large `seq_len=1024` shape:

- controlled `top_k=1`
  - sparse v2 is `1.075x` to `1.139x` faster than current MixLoRA across ranks
- native `top_k=1`
  - sparse v2 is `1.033x` to `1.191x` faster than current MixLoRA

So the grouped sparse execution idea is not dead. It is directionally correct, just not yet strong enough to overtake HMoRA.

### 3. The smaller-shape case is not favorable to sparse execution

For `seq_len=256`, sparse v2 stays far below HMoRA at every rank:

- best controlled result is only about `0.288x` of HMoRA

So any sparse advantage here is not broad. It only begins to appear in the larger-workload, higher-rank regime.

### 4. Memory is still a problem for sparse v2

At `seq_len=1024`, sparse v2 usually uses more memory than current MixLoRA and clearly more than HMoRA.

Example:

- controlled `top_k=1`, `rank=128`
  - HMoRA: `4.181 GB`
  - MixLoRA current: `4.114 GB`
  - sparse v2: `4.559 GB`

- controlled `top_k=2`, `rank=128`
  - HMoRA: `4.181 GB`
  - MixLoRA current: `5.262 GB`
  - sparse v2: `6.214 GB`

So the current sparse v2 path is trading some throughput improvement for worse memory behavior.

## Interpretation

These results support a narrower conclusion than "dense is always better":

- low-rank sparse execution is clearly not competitive with HMoRA here
- higher-rank sparse execution does move in the right direction
- the crossover regime may exist in principle, but it was not reached in this sweep

What this means in practice:

- `HMoRA` remains the best practical FFN MoE+LoRA systems design in this repo today
- `mixlora_sparse_grouped_v2` is promising enough to justify one more optimization pass only if we explicitly target:
  - reducing packing overhead
  - reducing activation memory
  - lowering the cost of expert-specific launches

Without that, the evidence is not strong enough to justify a full tuner integration.

## Recommendation

Recommended next step:

- accept `HMoRA` as the practical winner for current repo usage
- do not integrate `mixlora_sparse_grouped_v2` as a real tuner yet
- only continue sparse MixLoRA research if the goal is kernel-level or dispatch-level optimization work

In other words:

- for practical training and study comparisons now: use `HMoRA`
- for systems research on true sparse MoE+LoRA: sparse MixLoRA remains interesting, but only as a deeper optimization project, not as a near-ready tuner
