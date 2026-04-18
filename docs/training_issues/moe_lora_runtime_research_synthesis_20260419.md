# MoE+LoRA Runtime Research Synthesis

Date: 2026-04-19

## Open This First

If you need to regain the idea quickly, read only these sections in order:

1. "Current Verdict"
2. "Method Taxonomy"
3. "Most Important Results"
4. "Reproduction Commands"

Then open the supporting notes only if you need the deeper evidence trail.

## Why This Exists

This note consolidates the runtime and systems investigation around:

- `LoRA`
- `HMoRA`
- `AdaMoLE`
- `MoELoRA`
- `MixLoRA`
- the `mixlora_sparse_grouped_v2` research prototype

The goal was to understand which MoE+LoRA design is:

- best as a practical training default on current GPUs
- best as a long-run scalable direction for multi-task fine-tuning

This synthesis is the source of truth for the current conclusion. Some older notes remain important but are intermediate or background only.

## Method Taxonomy

| Method | Routing granularity | Logical sparsity | Compute sparsity | Practical runtime story |
|---|---|---:|---:|---|
| `LoRA` | none | no | no | fastest dense baseline |
| `HMoRA` | token or hybrid | yes | no | route-sparse, compute-dense, very GPU-friendly |
| `AdaMoLE` | token-level | yes | yes | true sparse dispatch, expensive gather/scatter path |
| `MoELoRA` | batch/task-level | yes | yes | sparser than HMoRA, cheaper than AdaMoLE, still dispatch-heavy |
| `MixLoRA current` | token-level FFN | yes | yes | stronger sparse base because frozen FFN work is shared |
| `mixlora_sparse_grouped_v2` | token-level FFN | yes | yes | grouped sparse prototype that improves MixLoRA but only wins in some scaling regimes |

The core distinction that drove most results is:

- `HMoRA` uses sparse routing semantics but keeps the heavy work in packed dense GEMMs.
- `AdaMoLE`, `MoELoRA`, and `MixLoRA` can do true sparse expert execution, but they pay dispatch overhead.

## Study Map

Open these only if you need the evidence behind a specific part of the conclusion:

- [hmora_paper_routing_vs_compute_audit_20260418.md](/local/joeldag/repos/moe-study/docs/training_issues/hmora_paper_routing_vs_compute_audit_20260418.md)
  Background audit showing that the paper describes sparse activation, while the checked-in HMoRA code is routing-sparse but compute-dense.
- [adamole_vs_lora_batchsize_investigation_20260418.md](/local/joeldag/repos/moe-study/docs/training_issues/adamole_vs_lora_batchsize_investigation_20260418.md)
  Why AdaMoLE needs much smaller batch sizes than plain LoRA despite similar trainable parameter counts.
- [hmora_sparse_h100_investigation_20260418.md](/local/joeldag/repos/moe-study/docs/training_issues/hmora_sparse_h100_investigation_20260418.md)
  Dense vs naive true-sparse HMoRA benchmark on H100.
- [mixlora_sparse_base_audit_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/mixlora_sparse_base_audit_20260419.md)
  Why MixLoRA was chosen as the sparse base for deeper systems work.
- [mixlora_sparse_v2_rank_sweep_h100_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/mixlora_sparse_v2_rank_sweep_h100_20260419.md)
  Intermediate high-rank sweep that first showed sparse v2 getting close to HMoRA.
- [mixlora_sparse_v2_extended_scalability_h100_20260419.md](/local/joeldag/repos/moe-study/docs/training_issues/mixlora_sparse_v2_extended_scalability_h100_20260419.md)
  Final detailed result for sparse MixLoRA vs HMoRA scaling.

Benchmark harnesses:

- [hmora_sparse_study.py](/local/joeldag/repos/moe-study/scripts/local/hmora_sparse_study.py)
- [moe_lora_runtime_study.py](/local/joeldag/repos/moe-study/scripts/local/moe_lora_runtime_study.py)
- [mixlora_sparse_v2_study.py](/local/joeldag/repos/moe-study/scripts/local/mixlora_sparse_v2_study.py)

## Current Verdict

There are two different winners depending on the question.

For practical use right now:

- `HMoRA` is the best MoE+LoRA default in this repo.
- `LoRA` is still the cleanest pure efficiency baseline if you do not need routed experts.
- `AdaMoLE` is not attractive from a systems perspective in the current implementation.
- `MoELoRA` is a better true-sparse compromise than `AdaMoLE`, but still much slower than `HMoRA`.

For long-run scalability:

- optimized sparse MixLoRA-style execution is now the most promising research direction.
- The crossover against HMoRA is real, but only in the high-rank, high-expert, low-`top_k`, long-sequence regime.

So the stable recommendation is:

- default training and comparisons now: `HMoRA`
- future scalable MoE+LoRA systems work: continue with sparse MixLoRA, not sparse HMoRA and not AdaMoLE

## Most Important Results

### 1. Why AdaMoLE is slow

Main takeaway:

- AdaMoLE is token-level compute-sparse, but the current implementation is routing-, indexing-, and scatter-heavy.
- Similar trainable parameter counts do not predict similar runtime.

Key supporting result:

- In a controlled adapter-only benchmark, AdaMoLE with `2` experts and fewer trainable parameters than LoRA was still `4.82x` slower.
- In the 10-language profile, the checked-in `adamole` run uses `batch_size=7` while `lora_r8` uses `batch_size=32`.

### 2. HMoRA is routing-sparse but compute-dense

Main takeaway:

- HMoRA's `top_k` changes routing semantics more than runtime because the heavy work stays in packed dense GEMMs.

Key supporting result from the H100 dense-vs-true-sparse benchmark:

- naive true-sparse HMoRA lost in every tested case
- adapter-tier sparse/dense mean throughput ratio: `0.0693x`
- block-tier sparse/dense mean throughput ratio: `0.0436x`

This ruled out a naive `hmora_sparse` direction for this repo.

### 3. Cross-method H100 runtime ranking

Artifact:

- `/tmp/moe_lora_runtime_study_full_h100_20260419`

Mean throughput across the full H100 runtime sweep:

| Routing mode | LoRA | HMoRA | MoELoRA | AdaMoLE |
|---|---:|---:|---:|---:|
| controlled | `1.66M tok/s` | `1.22M tok/s` | `318k tok/s` | `297k tok/s` |
| native | `1.60M tok/s` | `989k tok/s` | `251k tok/s` | `114k tok/s` |

Mean memory across the same sweep:

| Routing mode | LoRA | HMoRA | MoELoRA | AdaMoLE |
|---|---:|---:|---:|---:|
| controlled | `0.386 GB` | `0.406 GB` | `0.436 GB` | `0.461 GB` |
| native | `0.386 GB` | `0.407 GB` | `0.491 GB` | `0.817 GB` |

Interpretation:

- `HMoRA` is the best routed method on systems/runtime.
- `MoELoRA` is the best current true-sparse compromise.
- `AdaMoLE` is the weakest current routed method in systems terms.

### 4. Sparse MixLoRA v2 started as "promising but still behind"

Artifact:

- `/tmp/mixlora_sparse_v2_rank_sweep_h100_20260419`

Initial rank sweep result:

- for `seq_len=1024`, controlled active experts `=1`
  - `r=8`: sparse v2 / HMoRA = `0.393x`
  - `r=32`: `0.462x`
  - `r=64`: `0.626x`
  - `r=128`: `0.914x`

This was the first strong sign that sparse execution might overtake HMoRA at higher rank, but it had not crossed over yet in that earlier sweep.

### 5. Extended sweep proved the crossover is real

Artifacts:

- `/tmp/mixlora_sparse_v2_ext_controlled_h100_20260419`
- `/tmp/mixlora_sparse_v2_ext_native_h100_20260419`

Global average still favored HMoRA:

- controlled sparse/HMoRA mean throughput ratio: `0.735x`
- native sparse/HMoRA mean throughput ratio: `0.889x`

But the important result is the measured crossover region:

At `seq_len=1024`, earliest controlled-routing crossovers were:

| Experts | Active experts | First measured crossover rank |
|---|---:|---:|
| `32` | `1` | `192` |
| `32` | `2` | `256` |
| `32` | `4` | `384` |
| `64` | `1` | `192` |
| `64` | `2` | `192` |
| `64` | `4` | `256` |
| `64` | `8` | `512` |
| `128` | `1` | `192` |
| `128` | `2` | `192` |
| `128` | `4` | `256` |
| `128` | `8` | `384` |

No crossover was measured up to `r=512` for:

- `seq_len=256` in any tested case
- active experts `=16` at `seq_len=1024`

Best measured sparse-v2 case:

| Shape | HMoRA | sparse v2 | sparse/HMoRA |
|---|---:|---:|---:|
| `r=512, E=128, seq=1024, active=1` | `20.5k tok/s`, `22.08 GB` | `50.3k tok/s`, `16.89 GB` | `2.46x` |

This changed the conclusion from:

- "HMoRA wins everywhere we tested"

to:

- "HMoRA is the better average default, but sparse MixLoRA is the better asymptotic direction when rank, expert count, and context length are high and `top_k` stays small."

## Reproduction Commands

These are the commands that match the saved H100 study configurations.

### Cross-method runtime study

```bash
python3 scripts/local/moe_lora_runtime_study.py \
  --device cuda \
  --output-dir /tmp/moe_lora_runtime_study_full_h100_20260419
```

This run relies on the script defaults for the full matrix:

- methods: `lora,hmora,adamole,moelora`
- routing modes: `controlled_routing,native_routing`
- pass mode: `forward_backward`
- dtype: `bfloat16`
- hidden size: `4096`
- rank: `64`
- experts: `8,16,32`
- batch sizes: `1,8`
- sequence lengths: `256,1024`
- controlled active experts: `1,2,4`
- HMoRA top-k: `1,2,4`
- MoELoRA top-k: `1,2,4`

### Initial MixLoRA sparse-v2 rank sweep

```bash
python3 scripts/local/mixlora_sparse_v2_study.py \
  --variants lora_ffn_dense,hmora_ffn_dense_packed,mixlora_current,mixlora_sparse_grouped_v2 \
  --routing-modes controlled_routing,native_routing \
  --pass-mode forward_backward \
  --device cuda \
  --dtype bfloat16 \
  --hidden-sizes 4096 \
  --ffn-multipliers 4 \
  --ranks 8,16,32,64,128 \
  --num-experts-values 32 \
  --batch-size-values 8 \
  --seq-len-values 256,1024 \
  --controlled-active-experts-values 1,2 \
  --native-top-k-values 1,2 \
  --steps 10 \
  --warmup-steps 3 \
  --output-dir /tmp/mixlora_sparse_v2_rank_sweep_h100_20260419
```

### Extended sparse-v2 controlled sweep

```bash
python3 scripts/local/mixlora_sparse_v2_study.py \
  --variants hmora_ffn_dense_packed,mixlora_sparse_grouped_v2 \
  --routing-modes controlled_routing \
  --pass-mode forward_backward \
  --device cuda \
  --dtype bfloat16 \
  --hidden-sizes 4096 \
  --ffn-multipliers 4 \
  --ranks 128,192,256,384,512 \
  --num-experts-values 32,64,128 \
  --batch-size-values 8 \
  --seq-len-values 256,1024 \
  --controlled-active-experts-values 1,2,4,8,16 \
  --steps 10 \
  --warmup-steps 3 \
  --output-dir /tmp/mixlora_sparse_v2_ext_controlled_h100_20260419
```

### Extended sparse-v2 native slice

```bash
python3 scripts/local/mixlora_sparse_v2_study.py \
  --variants hmora_ffn_dense_packed,mixlora_sparse_grouped_v2 \
  --routing-modes native_routing \
  --pass-mode forward_backward \
  --device cuda \
  --dtype bfloat16 \
  --hidden-sizes 4096 \
  --ffn-multipliers 4 \
  --ranks 128,256,512 \
  --num-experts-values 32,128 \
  --batch-size-values 8 \
  --seq-len-values 256,1024 \
  --native-top-k-values 1,2,4 \
  --steps 10 \
  --warmup-steps 3 \
  --output-dir /tmp/mixlora_sparse_v2_ext_native_h100_20260419
```

### HMoRA dense-vs-true-sparse baseline

The saved H100 HMoRA study corresponds to this configuration:

```bash
python3 scripts/local/hmora_sparse_study.py \
  --tier both \
  --pass-mode forward_backward \
  --device cuda \
  --dtype bfloat16 \
  --hidden-sizes 2048,4096 \
  --ranks 8 \
  --num-experts-values 8,16,32 \
  --top-k-values 1,2,4,8,16,32 \
  --batch-size-values 1 \
  --seq-len-values 256,1024 \
  --ffn-multipliers 4 \
  --use-hydra-lora true \
  --steps 10 \
  --warmup-steps 3 \
  --output-dir /tmp/hmora_sparse_gpu_main
```

## Important Caveats

- These are benchmark harnesses, not full training-quality comparisons.
- The key conclusions are about execution behavior, scaling, and systems tradeoffs.
- Controlled routing isolates execution cost; native routing shows what happens under each method's own router behavior.
- The `/tmp` artifacts are useful on the current machine, but the durable conclusions are preserved in these docs.

## What To Do Next

If you want the best method to use now:

- use `HMoRA` for routed MoE+LoRA experiments
- use plain `LoRA` when routed experts are not necessary

If you want the best research direction to continue:

- continue sparse MixLoRA, not sparse HMoRA
- focus a `v3` prototype on:
  - reducing dispatch overhead
  - reducing memory blow-up as active experts rise
  - preserving the now-measured crossover gains at high rank and low `top_k`

If you want to write this up as an ablation later:

- treat the notes above as the evidence trail
- use this synthesis as the summary of record
- do not use the first rank-sweep note alone, because its conclusion was later superseded by the extended sweep
