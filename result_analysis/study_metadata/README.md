# Study metadata: available information and gaps

Collected 2026-09-16 from the five canonical Llama study workbooks and their
saved training artifacts. This covers **159 training runs**: 55 rank-sweep,
68 expert-sweep, 22 mixed-follow-up, 7 final-96, and 7 final-200 runs.
It includes workbook entries excluded from selection, including the original
failed CoLA runs. It does not substitute later CoLA retries or include the
separate Gemma/Falcon extension. Baseline evaluations are not training runs.

No workbooks, benchmark values, loss values, training artifacts, or jobs were changed.

## What is available?

| Information | Available | Qualification / gap |
|---|---:|---|
| Actual trainer duration and optimizer steps | 159/159 runs | Saved `train_results.json` and `trainer_state.json`; not scheduler allocation duration |
| Input-token counter and calculated throughput | 159/159 | Counts input positions, including padding; not unique/non-padding content tokens |
| World size, microbatch, accumulation, global batch | 159/159 | Configuration and training startup logs; world size is 4 throughout |
| Sequence length | 159/159 | 1,024; token counters also equal steps × global batch × 1,024 for every run |
| Trainable and total model parameters | 159/159 | Recovered from training startup logs; not active-parameter counts |
| Implementation rank and alpha | 153/153 applicable | Six MoV configurations have no applicable LoRA rank/alpha |
| Target modules and routing settings | 159/159 | Saved adapter configurations; see the detailed CSV |
| Initial learning rate, independently supported | 153/159 | Six missing launch/profile records; their workbooks say `2e-4`, retained separately as unverified |
| A100 resource request | 153/159 | Six request records missing; GPU model directly recorded in preflight logs for 36/159 |
| Explicit requested node count | 32/159 | Absence of a recovered `--nodes` flag is not evidence of a multi-node run |
| Explicit active-time target | 36/159 | All mixed/final runs; original sweep active-time contract still needs confirmation |
| Scheduler wall-time limit | 117/159 | Kept separate from active-time target and measured runtime |
| Evaluated checkpoint reference and benchmark coverage | 159/159 | From original provenance/workbook metadata; coverage is counted from primary metrics |
| Prepared-dataset document/example counts and language proportions | All 5 datasets | 356 dataset-language rows; composition, not the amount actually consumed by each run |
| Local source-pool document counts | 356/356 dataset-language rows | Frozen shard plans; different from full FineWeb corpus metadata |
| Full-corpus reference document counts | 351/356 | Non-English panel metadata available; English absent from those CSVs |
| Prepared-dataset token-position counts | 4/5 datasets | Mixed expansion audit stores examples/documents, not token-position totals |
| Actual per-run, per-language training exposure | Not recovered | Need logged language/document/token counters or an exact replay of sampler/order |
| Actual non-padding training-token counts | Not recovered | Eight runs report an additional effective-throughput metric, but no uniformly recoverable token count/definition |
| Measured active parameters per token | Not recovered | Needs a clear definition and adapter/routing accounting; cannot infer as total parameters × top-k / experts |

## Simple table: final 96- and 200-language runs

All these runs use global batch 48 and sequence length 1,024. “Positions” means
the recorded input-token positions including padding, **not** verified non-padding
training tokens. Throughput is aggregated over the four training processes.

| Languages | Configuration | Trainer hours | Optimizer steps | Positions (M) | Positions/s | Trainable params (M) |
|---|---|---:|---:|---:|---:|---:|
| 96 | moelora_tr32_e16_k8 | 34.09 | 7,596 | 373.36 | 3,042 | 27.27 |
| 96 | hydralora_r2 | 34.08 | 10,349 | 508.67 | 4,146 | 34.47 |
| 96 | moelora_tr64_e32_k8 | 34.09 | 6,739 | 331.24 | 2,699 | 54.53 |
| 96 | mola_tr16_e8_k6 | 34.11 | 6,493 | 319.14 | 2,599 | 71.83 |
| 96 | mtllora_lr5e5_r8_nup16 | 34.09 | 19,981 | 982.11 | 8,002 | 187.95 |
| 96 | lora_r2 | 34.06 | 24,134 | 1,186.23 | 9,674 | 5.24 |
| 96 | lora_r4 | 34.06 | 24,141 | 1,186.58 | 9,677 | 10.49 |
| 200 | moelora_tr32_e16_k8 | 71.93 | 15,947 | 783.83 | 3,027 | 27.28 |
| 200 | hydralora_r2 | 71.92 | 21,829 | 1,072.94 | 4,144 | 34.47 |
| 200 | moelora_tr64_e32_k8 | 71.96 | 10,175 | 500.12 | 1,931 | 54.54 |
| 200 | mola_tr16_e8_k6 | 72.02 | 10,427 | 512.51 | 1,977 | 71.83 |
| 200 | mtllora_lr5e5_r8_nup16 | 71.84 | 30,555 | 1,501.84 | 5,807 | 189.98 |
| 200 | lora_r2 | 71.82 | 48,859 | 2,401.52 | 9,289 | 5.24 |
| 200 | lora_r4 | 71.85 | 48,745 | 2,395.91 | 9,263 | 10.49 |

## Prepared datasets, not measured training exposure

| Dataset | Language-script labels | Train documents | Train examples | Validation examples | Sampling fraction in frozen source plan |
|---|---:|---:|---:|---:|---|
| Original sweeps | 13 | 56,240 | 99,410 | 1,915 | 0.00025 |
| MoV expert sweep | 13 | 112,482 | 198,533 | 3,870 | 0.0005 |
| Mixed follow-up | 13 | 226,741 | 400,000 | 7,705 | Expanded to target size; not the copied base-plan fraction |
| Final 96 | 102 | 861,773 | 1,513,560 | 30,718 | 0.0025 |
| Final 200 | 215 | 2,219,542 | 3,875,724 | 78,809 | 0.005 |

The regime names count languages differently from the dataset's script-specific
labels and extra English label. Do not relabel the experiments as 102/215
languages without reconciling those definitions. English is deliberately
downsampled; source-pool proportions are **not** preserved exactly across all
languages. The saved final datasets contain 18 English train documents (96)
and 31 (200), plus one English validation document each. Those are prepared
counts, not proof of how many English examples any run actually saw.

## Interpretation and reporting limitations

- **Do not reuse generic throughput fields blindly.** For example, final-200
  `moelora_tr32_e16_k8` reports `train_steps_per_second=0.312`, but its 15,947
  actual steps / 258,937.4566 seconds is about **0.0616 steps/s**. The generic
  value corresponds to the planned 80,745-step limit. The new table uses actual
  counters, not this field. Runtime includes trainer overhead; it is not GPU-busy time.
  These are saved-run/session measurements, not total campaign costs including
  failed attempts, calibration, setup, and evaluation. No explicit checkpoint-resume
  flag was found in the recovered submission commands.
- **Batch size was not constant in the early sweeps.** Rank sweep: 4 for the
  seven HMoRA rows, 48 otherwise. Expert sweep: 56 for 14 MoLA rows, 32 for five
  CoLA rows, 44 for five MTL-LoRA rows, and 48 otherwise. All 36 mixed/final
  entries use 48. Report the actual values rather than a study-wide batch of 48.
- **Eight expert-sweep entries reached epoch 1:** `cola_r2_a1_b2`,
  `cola_r2_a3_b4`, `cola_r2_a5_b8`, and all five `mtllora_r2_nup{2,4,8,16,32}`.
  Their runtimes are about 3.30–3.49 hours. Confirm whether these finished on
  the epoch limit before the intended time budget. They have not been silently
  excluded. Mixed and final runs all report epochs below 1; this is not an
  item-level proof of unique data exposure or absence of sampler duplication.
- **Reuse the frozen dataset plans, not disabled-tokenization defaults.** The
  final training profiles still say `sample_fraction=0.00025`, but the dataset
  construction plans record **0.0025 (96)** and **0.005 (200)**. Both are retained
  in the CSV, with the profile field explicitly unverified. This collection does
  not change either source or any workbook.
- **Scaling is implementation-specific.** Rank/alpha come from saved adapter
  configs; MoELoRA rank is total rank. All applicable saved alpha values are 2r,
  but HMoRA uses alpha/sqrt(r), not alpha/r. The scaling column is the explicit
  adapter multiplier, not a measurement of total update magnitude.
- **Evaluation panels differ:** early/mixed stages have 42 primary units
  (9/18/9/6), final-96 has 249 (40/100/51/58), and final-200 has 408
  (65/166/84/93), in Belebele/FLORES/SIB200/Taxi1500 order. FLORES includes both
  directions. Matched-panel comparisons remain a separate analysis task.

## What you could provide

1. Any per-run language/document/token exposure logs, or a recorded sampler
   order, if these exist elsewhere. Without them we should report dataset
   composition and global input-position counters, not fabricate per-language
   consumption.
2. The original launch/resolved configs for rank-sweep
   `moelora_tr{16,32,64,128,256}_e8_k2` and mixed `moelora_tr16_e8_k2`, to confirm
   their initial LR and missing hardware request. The current workbooks say
   `2e-4`, but those six exact settings were not independently recovered.
3. The original sweep time-budget contract or archived scheduler accounting,
   including whether completing one epoch before the timer was allowed. Actual
   runtimes are already available for all runs.
4. An agreed definition or existing measurements for active parameters. Routed
   parameter participation and parameters actually computed by dense/fused
   implementations need not be the same quantity.

## Files and definitions

- [runs.csv](runs.csv): all 159 runs, including missing-value labels and sources.
- [datasets.csv](datasets.csv): the five prepared datasets and sampling sources.
- [languages.csv](languages.csv): 356 dataset-language rows, counts and shares.
- [sources.json](sources.json): per-run source paths, counters' hashes, commands,
  logs, saved adapter configs and parameter-count provenance.
- [dataset_sources.json](dataset_sources.json): dataset/panel/plan source hashes.

`MISSING` means not recovered, never zero. `N/A` means not applicable.
`trainer_runtime_hours = train_results.train_runtime / 3600`.
`input_token_positions_per_second = trainer_state.num_input_tokens_seen / train_runtime`.
`optimizer_steps_per_second = trainer_state.global_step / train_runtime`.
`trainer_gpu_hours_configured_world = train_runtime × world_size / 3600`; this is
not a scheduler-billing or utilization measurement. Dataset shares use the
train split as denominator. Local source-pool shares use the frozen shard plan;
full-corpus reference shares use only the available non-English panel counts.

The local collector is `tmp/collect_reviewer_evidence.py`; it writes only this
evidence directory and asserts that the five workbooks' hashes are unchanged.
No claim here revalidates benchmark quality, significance, or training health.
