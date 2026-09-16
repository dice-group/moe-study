# Gemma 2 / Falcon: 96-language extension

## Maintained source and historical evidence

The scientific settings in `96_languages.json` are unchanged. Operational paths,
resources, exclusions, and measured recovery allocator settings live in
`noctua2_runtime.json`, which is included in gate hashes. New submissions use
attempt-scoped logs. These source changes deliberately invalidate old gates:
do not re-sign historical receipts or relaunch the existing campaign blindly.

Reusable recovery allocator, monitor-cleanup, and checkpointing-probe fixes are
integrated here. Historical recovery-only scheduling and original/recovery source
mappings remain evidence, not a second maintained launcher. Existing runs must
be interpreted using their recorded source trees, not this updated source.

Generated benchmark YAMLs and large model/data artifacts remain local. A fresh
checkout still needs the frozen task panel and generated tasks; the custom
Belebele/Taxi1500 generator is not yet preserved here. Preparation must not be
claimed reproducible from Git alone until that dependency is restored.

This isolated campaign reproduces the seven final 96-language configurations on
the **base** Gemma 2 9B and Falcon 7B models. The original Llama results and
notebook are not overwritten. The scientific contract is
[96_languages.json](96_languages.json).

## Frozen comparison

- Seven configurations per model: the two MoE-LoRAs, HydraLoRA (eight heads),
  lower-LR MTL-LoRA, MoLA, LoRA r2, and LoRA r4; no new winner selection.
- Original per-configuration ranks, expert counts, top-k, alpha, dropout, routing
  settings and learning rates. MTL-LoRA uses `5e-5`; the other six use `2e-4`.
- One node / four A100s, global batch 48, BF16, sequence length 1024, seed 42,
  one-epoch linear schedule without warmup. Original 34h trainer target,
  34h30 allocation and USR1 warning 20 minutes before its end. Actual trainer
  runtime must fall within the original ±20-minute acceptance tolerance.
- Exactly the original selected documents, train/validation assignment and
  language IDs, separately retokenized with pinned native model/tokenizer
  snapshots. No vocabulary extension or changed sampling distribution.
- Falcon maps attention to `query_key_value,dense`; FFN to
  `dense_h_to_4h,dense_4h_to_h`. The two MoE-LoRAs retain attention-only scope;
  the other five retain attention plus FFN scope. Native fused QKV is not split.
- Full frozen benchmark panel: Belebele 40, FLORES 100 (50 per direction), SIB200
  51, Taxi1500 58, including English. Benchmark language-ID mode stays `no_ids`,
  as in the archived final runs; training/validation retain their routing IDs.
  Benchmark context limits remain the harness/model's native defaults and are
  recorded in raw harness outputs; no forced extension of Falcon's context.

## Gates and execution

1. Download/hash pinned assets; tokenize 50 original document-plan shards per model.
2. Verify document IDs, split and routing equality against the original shards,
   padding/labels, token bounds, all languages and merged row counts.
3. Prove the **exact Merlin Python and local Factory/PEFT imports** on an A100.
   Check accelerated attention against eager logits, including Gemma sliding-window
   behavior. If acceleration fails numerical parity, only validated eager is used.
   The installed FlashAttention wheel needs `GLIBC_2.32`, while these nodes provide
   2.28. A bounded CPU job rebuilds the **same 2.8.3 release**, SM80-only, following
   the [upstream build instructions](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3).
   It installs into the campaign's own runtime overlay, never into shared Merlin;
   a new A100 parity check decides whether it can be used.
4. First validate LoRA r2 per architecture, then the remaining six. Four-A100
   pilots measure 20 warmup + 100 steps for microbatches 1/2/3/4/6/12 and checkpointing
   variants, preserving global batch and the full one-epoch LR horizon. OOMs skip
   larger microbatches; other failures hold the gate. Repeat the fastest candidate;
   require ≤10% throughput disagreement and ≥10% free VRAM.
5. Require finite gradients/loss, complete adapter coverage, signal-triggered save,
   exact adapter tensor reload, Factory inference parity and five benchmark smoke
   tasks. Both the original `.bin` format and safetensors are understood.
6. Dataset capacity must exceed the projected 34h consumption with a 25% margin.
   **Hold affected runs if it cannot: no repetition, corpus expansion, LR change,
   or deliberately slower configuration is authorized.**
7. A short dependent gate-checker may submit the approved 34h production run.
   It rechecks code/config/data hashes and permits only measured systems changes.
   Production starts from the base model, never the pilot checkpoint. The gate is
   rechecked again inside the actual training allocation.
8. After training, an `afterany` acceptance job verifies Slurm completion, finite
   metrics, runtime, final-step direct adapter checkpoint and required save markers.
   Only successful acceptance releases the full validation loss and benchmark array.
   Fifteen disjoint single-GPU benchmark shards cover all 249 tasks; consolidation
   rejects missing/duplicate tasks, changed result files or checkpoint mismatches.
9. Produce one new dated workbook per architecture with seven trained configurations
   and its own unadapted baseline, parameter/document counts and validation evidence.
   Recompute common-seven ranks **without baselines** for Spearman/Kendall correlations
   against Llama and between the new architectures. These are descriptive, single-seed
   comparisons, not independent replicated evidence of architectural causation.

Node exclusions are limited to explicitly documented faults (`n2gpu1211` and
`n2gpu1213`, NHC/NVLink drain evidence on 2026-09-13). No old whitelist is reused. Each allocation performs GPU preflight;
training also records finite-gradient checks, per-step timing and GPU utilization.

## Commands

Run from the repository root with the dedicated environment and source paths:

```bash
export PYTHONPATH="$PWD:$PWD/LLaMA-Factory/src:$PWD/scripts/eval/latest_checkpoint/lm-evaluation-harness"
PY=/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python

$PY -m pipeline_tools.cross_arch_study status
$PY -m pytest -q pipeline/tests LLaMA-Factory/tests/model/test_nonmergeable_moe_adapters.py

# Preparation and submission are idempotence guarded, not commands to blindly rerun.
$PY -m pipeline_tools.cross_arch_study prepare
$PY -m pipeline_tools.cross_arch_study submit-preparation --submit
$PY -m pipeline_tools.cross_arch_pilot submit --submit
$PY -m pipeline_tools.cross_arch_launch arm --submit
```

Live jobs and submission history are in the campaign's `*_jobs.json` files under
`outputs/pipeline_runs/gemma2_falcon_96_languages_2026-09-13/`.
The first CPU arrays hit a validation import-path error **after** tokenization;
`repair-validation` reuses those data shards and records superseded dependencies.
Do not cancel or delete useful data just because the old Slurm task says `FAILED`.

Inspect `gates/<run_id>.json`, `runs/<run_id>/campaign_progress.jsonl`,
`runs/<run_id>/gpu_utilization.csv`, `acceptance/<run_id>/acceptance.json`, and
individual benchmark shard receipts. Code changes invalidate prior pilot evidence;
failed/partial submissions are deliberately not automatically retried.

## Interactive repair checks (2026-09-13)

Two one-node/four-A100 reservations (`34335332`, `34335366`) were used to
repair and test every model/configuration before another queued calibration.
All fourteen combinations passed three real optimizer steps, a signal-triggered
adapter save, exact checkpoint-tensor reload, Factory inference parity and five
benchmark smoke tasks. These short checks **are not production or calibration
gates**; throughput confirmation, longer stability and dataset-capacity checks
remain mandatory.

Repairs preserve the scientific settings:

- FSDP wrapping now keeps shared embedding/output weights together; gradient
  checks correctly aggregate across ranks with empty local adapter shards.
- MoELoRA retains language IDs and recomputes routing during activation
  checkpoint recomputation. A focused test verifies loss and adapter-gradient
  parity against non-checkpointed execution. Gemma's two MoELoRAs need
  checkpointing at microbatch one; their non-checkpointed attempts OOMed.
- Inference comparison casts parameters, not FP32 rotary-frequency buffers.
  MoLA now honors PEFT's `disable_adapter()` switch; its enabled computation
  is unchanged. Missing duplicate expert-alias warnings were checked against
  exact canonical tensor reload and actual adapter effect.
- Benchmark subprocesses resolve FLORES relative data paths from the harness
  directory. Verified job-local copies of the same model weights and Merlin
  packages avoid repeated shared-filesystem startup delays.

An isolated Gemma MoLA attempt raised a Python `unknown opcode` error; its
fresh identical retry passed. The cause was not established, so the failed
attempt is retained and longer calibration must still confirm stability.

Evidence lives under each model's `debug/`: `individual_07`,
`mola_retry_09` (Gemma MoLA), `gc_fixed_12` (Gemma MoELoRA),
`reload_13` (all fourteen reload/benchmark checks), and `eval_paths_14`
(bounded baseline and final-loss checks). Superseded queued attempts are
retained in the campaign's `attempt_history/`, not treated as successful runs.

New workbooks go beside the submitted workbooks in `result_analysis/reports/moe_study/`,
named `moe_study_final_<model>_96_languages_<completion-date>.xlsx`.
Correlation CSVs and source hashes go in the campaign's `correlations/` directory.
Nothing is committed, staged or pushed by this workflow.
