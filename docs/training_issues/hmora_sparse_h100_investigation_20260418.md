# HMoRA Dense vs True-Sparse H100 Investigation

Date: 2026-04-18

## Question

Should this repo grow a true-sparse `hmora_sparse` execution path, or is the current dense-packed HMoRA design the better choice for GPU training?

## Short Answer

The current dense-packed HMoRA design is the right default for this codebase.

On the local H100, a true-sparse Python-dispatch prototype was slower than dense-packed HMoRA in every tested case, even at `top_k=1`. The current HMoRA implementation is routing-sparse but compute-dense, and that matches GPU hardware much better than per-expert token dispatch.

## Setup

Benchmark harness:

- [scripts/local/hmora_sparse_study.py](/local/joeldag/repos/moe-study/scripts/local/hmora_sparse_study.py)
- [docs/training_issues/hmora_sparse_study_analysis.ipynb](/local/joeldag/repos/moe-study/docs/training_issues/hmora_sparse_study_analysis.ipynb)

Variants compared:

- `dense_packed`: current HMoRA-style execution, where expert dimensions stay packed into large dense matmuls
- `true_sparse`: reference implementation that dispatches tokens to selected experts and only executes active experts

Hardware:

- local `NVIDIA H100-80GB`

Main sweep:

- `device=cuda`
- `dtype=bfloat16`
- `pass_mode=forward_backward`
- `tier in {adapter, block}`
- `hidden_size in {2048, 4096}`
- `seq_len in {256, 1024}`
- `num_experts in {8, 16, 32}`
- `top_k` swept up to `num_experts`
- `batch_size=1`
- `rank=8`
- `ffn_multiplier=4`
- `use_hydra_lora=true`

Saved artifacts:

- `/tmp/hmora_sparse_gpu_smoke/results.jsonl`
- `/tmp/hmora_sparse_gpu_main/results.jsonl`

## Correctness Check

All paired dense-vs-sparse runs passed the equivalence checks in the benchmark harness:

- paired cases: `120`
- equivalence failures: `0`

So the throughput comparison is not being driven by a numerical mismatch between the two implementations.

## Results

Aggregate `true_sparse / dense_packed` throughput ratio on the H100:

- adapter tier mean speedup: `0.0693x`
- adapter tier median speedup: `0.0582x`
- block tier mean speedup: `0.0436x`
- block tier median speedup: `0.0382x`
- sparse wins: `0 / 120`

In practical terms:

- adapter tier sparse was about `14x` slower on average
- block tier sparse was about `23x` slower on average

Best sparse case in the entire sweep:

- tier: `adapter`
- `hidden_size=2048`
- `seq_len=256`
- `num_experts=8`
- `top_k=1`
- dense-packed: `179,829 tok/s`
- true-sparse: `40,388 tok/s`
- sparse vs dense: `0.2246x`

Even the best sparse case was still about `4.5x` slower than dense-packed.

Representative larger cases:

- `adapter`, `hidden=4096`, `seq=1024`, `E=8`, `top_k=1`
  - dense-packed: `1,301,041 tok/s`
  - true-sparse: `137,537 tok/s`
  - sparse vs dense: `0.1057x`
- `adapter`, `hidden=4096`, `seq=1024`, `E=32`, `top_k=1`
  - dense-packed: `811,507 tok/s`
  - true-sparse: `36,505 tok/s`
  - sparse vs dense: `0.0450x`
- `block`, `hidden=4096`, `seq=1024`, `E=32`, `top_k=32`
  - dense-packed: `239,127 tok/s`
  - true-sparse: `5,101 tok/s`
  - sparse vs dense: `0.0213x`

Peak memory in this prototype was also similar between the two variants, so the sparse path did not buy a meaningful memory advantage.

## Why Dense-Packed Wins

The current HMoRA implementation uses `top_k` to change routing semantics, but it does not turn the heavy linear algebra into sparse execution.

In the checked-in HMoRA layer:

- the router computes expert weights and optionally keeps only top-k scores
- the expert activations are still laid out in packed expert dimensions
- the heavy work still happens through large dense `F.linear(...)` calls

See:

- [LLaMA-Factory/src/peft/tuners/hmora/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/hmora/layer.py:267)
- [LLaMA-Factory/src/peft/tuners/hmora/layer.py](/local/joeldag/repos/moe-study/LLaMA-Factory/src/peft/tuners/hmora/layer.py:487)

That execution pattern fits GPU hardware well because it stays close to a small number of large dense matrix multiplies.

The true-sparse prototype loses because it replaces those large dense operations with:

- token-to-expert indexing
- expert-specific sub-batches
- many smaller per-expert operations
- more irregular memory access
- more launch and dispatch overhead

That is exactly the kind of work that tends to underutilize an H100.

## Why `top_k` Barely Changes Throughput Today

The H100 numbers support the earlier code reading that current HMoRA is compute-dense.

Example dense-packed throughput slice:

- `adapter`, `hidden=4096`, `seq=1024`, `E=8`
  - `top_k=1`: `1,301,041 tok/s`
  - `top_k=2`: `1,266,875 tok/s`
  - `top_k=4`: `1,164,036 tok/s`
  - `top_k=8`: `1,165,526 tok/s`
- `block`, `hidden=4096`, `seq=1024`, `E=16`
  - `top_k=1`: `234,702 tok/s`
  - `top_k=2`: `243,022 tok/s`
  - `top_k=4`: `241,998 tok/s`
  - `top_k=8`: `239,869 tok/s`
  - `top_k=16`: `241,798 tok/s`

So `top_k` changes the effective mixture, but it does not reduce the main compute in the current implementation.

## GEMM Note

`GEMM` means `GEneral Matrix Multiply`. In practice, it is the core dense matrix multiplication used by linear layers, for example:

- `C = A x B`

Modern GPUs are highly optimized for large dense GEMMs. The current HMoRA path maps well to that because it keeps expert work packed into dense matrix multiplies. The true-sparse prototype breaks one efficient dense GEMM into many smaller expert-dispatched operations, which is why it performs much worse even though it computes fewer logical experts per token.

## Conclusion

The current HMoRA design should stay dense-packed.

What the study shows is:

- current HMoRA is routing-sparse, not compute-sparse
- that design is intentional enough to be considered faithful to the implementation
- a naive true-sparse execution path is not promising on GPU
- building a repo-integrated `hmora_sparse` tuner from this Python-dispatch approach is not justified

If sparse HMoRA is revisited later, it should be treated as a kernel-level project and not just a tuner variation. The likely requirements would be:

- grouped GEMM
- fused Triton or CUDA kernels
- expert batching without Python loops
- a design that preserves high GPU occupancy
