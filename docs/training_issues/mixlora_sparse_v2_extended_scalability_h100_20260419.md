## MixLoRA Sparse v2 Extended Scalability Study on H100

Status: current detailed result for the sparse MixLoRA vs HMoRA scalability question.
Open first: [moe_lora_runtime_research_synthesis_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/moe_lora_runtime_research_synthesis_20260419.md)

### Goal

This study extends the earlier `mixlora_sparse_v2` rank sweep to answer the remaining systems question:

> Can an optimized true-sparse MoE+LoRA execution path beat HMoRA's dense-packed route-sparse design once rank, expert count, and `top_k` are pushed high enough?

The comparison here is intentionally narrow:

- `hmora_ffn_dense_packed`
- `mixlora_sparse_grouped_v2`

The objective is systems scalability on the local H100, not final task quality.

### Setup

Harness:

- [scripts/local/mixlora_sparse_v2_study.py](/local/joeldag/repos/moe-study/scripts/local/mixlora_sparse_v2_study.py)

Runs:

- Controlled routing:
  - output: `/tmp/mixlora_sparse_v2_ext_controlled_h100_20260419`
  - `300` rows
- Native routing:
  - output: `/tmp/mixlora_sparse_v2_ext_native_h100_20260419`
  - `72` rows

Shared benchmark settings:

- device: local `H100`
- dtype: `bfloat16`
- pass mode: `forward_backward`
- hidden size: `4096`
- FFN multiplier: `4`
- batch size: `8`
- sequence lengths: `256`, `1024`

Controlled-routing sweep:

- rank: `128, 192, 256, 384, 512`
- experts: `32, 64, 128`
- active experts: `1, 2, 4, 8, 16`

Native-routing slice:

- rank: `128, 256, 512`
- experts: `32, 128`
- `top_k`: `1, 2, 4`

All rows passed finite forward/backward checks.

### Main Finding

The answer is no longer simply "HMoRA wins."

The more precise answer is:

- `HMoRA` is still the better practical default.
- `mixlora_sparse_grouped_v2` becomes the more scalable execution design in the high-rank, high-expert, low-`top_k`, long-sequence regime.

That crossover did not appear in the earlier smaller sweep because the dense-packed HMoRA path had not yet been pushed into the regime where its packed expert cost dominates.

### Aggregate Results

Controlled routing mean over the full matrix:

- `hmora_ffn_dense_packed`: `74.9k tok/s`, `7.858 GB`
- `mixlora_sparse_grouped_v2`: `47.8k tok/s`, `11.267 GB`
- sparse/HMoRA mean throughput ratio: `0.735x`
- sparse/HMoRA mean memory ratio: `1.785x`

Native routing mean over the smaller diagnostic slice:

- `hmora_ffn_dense_packed`: `79.9k tok/s`, `8.578 GB`
- `mixlora_sparse_grouped_v2`: `59.6k tok/s`, `7.893 GB`
- sparse/HMoRA mean throughput ratio: `0.889x`
- sparse/HMoRA mean memory ratio: `1.066x`

So the global average still favors HMoRA. The important change is that the average now hides a real crossover region.

### Where Sparse v2 Actually Wins

The decisive regime is:

- `seq_len = 1024`
- low active experts / low `top_k`
- high ranks
- more experts

Best controlled-routing case:

- `rank=512`, `experts=128`, `seq_len=1024`, active experts `=1`
- `HMoRA`: `20.5k tok/s`, `22.082 GB`
- `MixLoRA sparse v2`: `50.3k tok/s`, `16.887 GB`
- sparse/HMoRA throughput ratio: `2.455x`
- sparse/HMoRA memory ratio: `0.765x`

Another strong controlled case:

- `rank=384`, `experts=64`, `seq_len=1024`, active experts `=2`
- `HMoRA`: `51.1k tok/s`, `8.659 GB`
- `MixLoRA sparse v2`: `91.6k tok/s`, `7.775 GB`
- sparse/HMoRA throughput ratio: `1.792x`

At `seq_len=1024`, the earliest measured controlled-routing crossovers were:

- `experts=32`
  - active `1`: crossover at `rank=192`
  - active `2`: crossover at `rank=256`
  - active `4`: crossover at `rank=384`
  - active `8` or `16`: no crossover up to `rank=512`
- `experts=64`
  - active `1`: crossover at `rank=192`
  - active `2`: crossover at `rank=192`
  - active `4`: crossover at `rank=256`
  - active `8`: crossover at `rank=512`
  - active `16`: no crossover up to `rank=512`
- `experts=128`
  - active `1`: crossover at `rank=192`
  - active `2`: crossover at `rank=192`
  - active `4`: crossover at `rank=256`
  - active `8`: crossover at `rank=384`
  - active `16`: no crossover up to `rank=512`

Native routing shows the same direction:

- `seq_len=1024`, `experts=32`
  - `top_k=1`: crossover at `rank=256`
  - `top_k=2`: crossover at `rank=256`
  - `top_k=4`: crossover at `rank=512`
- `seq_len=1024`, `experts=128`
  - `top_k=1`: crossover at `rank=256`
  - `top_k=2`: crossover at `rank=256`
  - `top_k=4`: crossover at `rank=256`

### Where HMoRA Still Wins

For shorter sequence length, HMoRA remains clearly better.

At `seq_len=256`, there was no measured crossover in any controlled or native case up to `rank=512`.

Representative example:

- `rank=512`, `experts=128`, `seq_len=256`, active experts `=1`
- `HMoRA`: `16.8k tok/s`, `21.980 GB`
- `MixLoRA sparse v2`: `12.9k tok/s`, `16.653 GB`
- sparse/HMoRA throughput ratio: `0.771x`

So the sparse advantage is not universal. It appears only when the workload is large enough that HMoRA's dense-packed expert dimension becomes the dominant cost.

### Interpretation

This makes the scaling difference between the two designs much clearer:

- `HMoRA` is route-sparse but compute-dense.
  - Runtime scales with the packed expert path even when only a few experts matter logically.
  - This is excellent at lower ranks and smaller workloads because GPUs like the dense GEMMs.
  - It becomes less attractive as `rank * num_experts` grows.

- `mixlora_sparse_grouped_v2` is route-sparse and compute-sparse.
  - It pays dispatch and packing overhead.
  - That overhead dominates at smaller workloads.
  - It starts to win once active sparse work is much smaller than HMoRA's packed dense work.

The cleanest evidence is the controlled-routing means by active experts:

- active `1`: sparse/HMoRA mean throughput ratio `0.998x`
- active `2`: `0.944x`
- active `4`: `0.781x`
- active `8`: `0.577x`
- active `16`: `0.375x`

And by rank:

- `r=128`: `0.395x`
- `r=192`: `0.538x`
- `r=256`: `0.682x`
- `r=384`: `0.920x`
- `r=512`: `1.140x`

That is the crossover story in one line:

> Sparse v2 is not the better average implementation, but it is the better asymptotic implementation in the low-`top_k`, high-rank, high-expert, long-sequence regime.

### Implications For Multi-Task Fine-Tuning

If the goal is "the most scalable MoE+LoRA approach" for multi-task adaptation, the answer depends on the target regime.

Use `HMoRA` if:

- rank is moderate
- sequence length is not extremely long
- `top_k` is not very low
- you want the safest high-throughput default on current hardware

Pursue sparse MixLoRA-style execution if:

- you expect high LoRA ranks
- you want many experts
- you intend to keep `top_k` small
- long-context training matters
- scaling capacity without paying dense-packed expert cost is a priority

So the practical recommendation is:

- short term default for the repo: `HMoRA`
- most promising path for true long-run scalability: optimized sparse MixLoRA-style execution

### Recommendation

This is enough evidence to continue sparse-MixLoRA work.

But the next step should still be a systems optimization pass, not immediate tuner integration. The right follow-up is a `v3` benchmark prototype focused on:

- reducing dispatch overhead further
- reducing accumulation memory blow-up at larger `top_k`
- preserving the strong `top_k=1/2` gains at high rank

The current results justify that work because the crossover is now real, measured, and not just extrapolated.
