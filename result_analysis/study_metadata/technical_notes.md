# Study metadata: technical notes and evidence

Collected 2026-09-16 from the five canonical Llama study workbooks and their
saved training artifacts. This covers **159 training runs**: 55 rank-sweep,
68 expert-sweep, 22 mixed-follow-up, 7 final-96, and 7 final-200 runs.
It includes workbook entries excluded from selection, including the original
failed CoLA runs. It does not substitute later CoLA retries or include the
separate Gemma/Falcon extension. Baseline evaluations are not training runs.

No workbooks, benchmark values, loss values, training artifacts, or jobs were changed.

## Start here: training counters and language sampling

The current reporting focus is **token counts, throughput, optimizer steps,
and language sampling proportions**. Use the columns below directly; the
remaining sections provide background and limitations.

| Number to report | CSV | Exact column(s) | Meaning |
|---|---|---|---|
| Tokens processed, including padding | [runs.csv](runs.csv) | `input_token_positions_seen` | Recorded input positions consumed during the run; available for all 159 runs. Not unique or non-padding tokens. |
| Token throughput | [runs.csv](runs.csv) | `input_token_positions_per_second` | Recorded input positions divided by actual trainer runtime, aggregated across the four training processes. Includes padding. |
| Optimizer updates | [runs.csv](runs.csv) | `optimizer_steps` | Completed optimizer steps, not microbatches. |
| Training duration | [runs.csv](runs.csv) | `trainer_runtime_hours` | Actual trainer runtime, not Slurm allocation time or GPU-busy time. |
| Prepared training-data language proportions | [languages.csv](languages.csv) | `dataset_train_document_share`, `dataset_train_example_share` | Each language's fraction of prepared training documents or tokenized examples. Multiply by 100 for percentages. |
| Counts behind those proportions | [languages.csv](languages.csv) | `train_documents`, `train_examples` | Prepared counts per language, not counts consumed by an individual run. |
| Prepared token-position proportions | [languages.csv](languages.csv) | `dataset_train_token_position_share`, `train_token_positions` | Available for four datasets; missing for the mixed expansion. These are not actual per-run language exposure or non-padding-token shares. |
| Dataset construction sampling rate | [datasets.csv](datasets.csv) | `sampling_fraction_from_dataset_plan`, `sampling_recipe_note`, `english_sampling_override_reference` | Construction keep-rate and exceptions, not a language's share of the training mixture. |

**How to match the files:** select a run using `stage` and `run_id` in
`runs.csv`. Its `dataset` value matches `dataset` in `datasets.csv` and
`languages.csv`; the latter has one row per `language` within each dataset.
For the final regimes, filter `stage` to `final_96` or `final_200`.

**Reporting boundary:** actual per-run non-padding tokens
(`nonpadding_trained_tokens`) and per-language consumption
(`measured_language_exposure`) are not recovered. Do not multiply the global
run counter by prepared language shares and present that as measured exposure.
Report the prepared mixture separately. English's median-document policy is
explained [below](#english-sampling-policy).

For a compact per-run table, use `stage`, `run_id`, `trainer_runtime_hours`,
`optimizer_steps`, `input_token_positions_seen`, and
`input_token_positions_per_second`. For its language-mixture companion, use
`dataset`, `language`, `train_documents`, `train_examples`,
`dataset_train_document_share`, and `dataset_train_example_share`.

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
| Explicit active-time target | 36/159 | All mixed/final runs; owner clarified early step calibration then signal caps, but per-run stopping settings still need classification |
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

### English sampling policy

English was assigned the **rounded median sampled document target of the
selected non-English language entries**, not the mean or a token-based share.
The sampling rate was that target divided by the available English document
count. See [the implementation](../../pipeline/stages/tokenize.py) in
`_build_language_rates_for_tokenize`.

The saved dataset checks confirm the policy before the train/validation split:

| Dataset | Observed non-English median documents | English documents | Train / validation |
|---|---:|---:|---:|
| Final 96 | 19 | 19 | 18 / 1 |
| Final 200 | 31.5 | 32 | 31 / 1 |

These small English allocations are intentional, not accidental omissions.
The policy balances document counts against the median language entry; it does
not equalize token counts, preserve English's natural corpus proportion, or
establish actual English exposure during partial-epoch training. The counts
come from the datasets' `tokenized/_check.json` files referenced in
[dataset_sources.json](dataset_sources.json).

## Interpretation and reporting limitations

See [open questions](open_questions.md) for the prioritized concerns and the
specific missing evidence to supply.

### Training-budget policy and observed variation

The experiment owner clarified the progression: initially, some runs were
calibrated by choosing an optimizer-step count expected to take **3.6 hours**.
Throughput variation meant that the resulting durations were only approximately
matched. Later runs switched to a Slurm signal-based time cap to make budget
enforcement more reliable. This describes the intended workflow; a complete
per-run classification of the two stopping strategies has not yet been made.

In recovered sweep scripts, Slurm allows 3h51m and sends the stop-and-save signal
900 seconds before the limit: nominally 3h36m (216 minutes) after job start.
That clock includes setup; recorded trainer runtime is a different measurement.
Step boundaries, shutdown, and saving can also affect the measured duration.
Some commands additionally cap training at one epoch, permitting earlier
completion. Whether this early completion was explicitly intended remains open.

Observed trainer durations from `runs.csv`:

| Sweep | Runs | Minimum (min) | Median (min) | Maximum (min) | Difference from nominal 216 min (min) | Within 10 min of 216 |
|---|---:|---:|---:|---:|---:|---:|
| Rank | 55 | 206.75 | 212.59 | 217.06 | -9.25 to +1.06 | 55 / 55 |
| Expert | 68 | 197.86 | 213.69 | 216.59 | -18.14 to +0.59 | 61 / 68 |

These are all original sweep workbook entries, including failed/excluded CoLA
runs, not a filtered set of healthy or promoted configurations. They combine
historical stopping strategies and must not be interpreted as a measured
before/after comparison of step calibration versus signal stopping. The largest
expert-sweep shortfall is approximately 8.4% of the nominal budget. For example,
MTL-LoRA `nup32` completed one epoch in 200.34 minutes, 15.66 minutes below 216.

The appropriate description is **approximately budget-matched early sweeps**,
with actual durations disclosed, rather than identical active compute time.
Step calibration is not inherently invalid, but these differences should not
be assumed to have no effect on close configuration rankings.

Reproduce the duration summary from `trainer_runtime_hours * 60`, grouped by
`stage` (`rank_sweep` and `expert_sweep`); subtract 216 for nominal deviations.

### Why target modules differ

As clarified by the experiment owner, some configurations did not fit in the
available GPU memory when adapters targeted all attention and FFN modules.
For some of these configurations, targets were therefore reduced to attention
modules only. Separately, approaches such as MoV have approach-specific target
modules; the same target list is not applicable to every method.

The per-run `target_modules` column in [runs.csv](runs.csv) records the actual
saved settings. These restrictions explain the configuration choices, but do
not remove the comparison limitation: where targets change alongside expert
count or rank, the results do not isolate the effect of that variable alone.

### Batch calibration and comparison scope

The experiment owner clarified that early calibration maximized throughput
without enforcing one global batch size across approaches. Later calibration
fixed the global batch size and optimized throughput within that constraint.
All recorded mixed/final runs use global batch 48; early sweeps retain their
actual, varying batch sizes. The later policy better controls optimization
differences. The early results remain observations of throughput-optimized
recipes under an approximate time budget, not comparisons with every
optimization setting held constant.

Different evaluation panels are intentional: according to the experiment
owner, each regime evaluates languages included in its training language set.
This is a language-selection rule, not proof that every training language has
a benchmark task or was encountered during a partial epoch. Configurations
within each regime share a panel. Comparing regimes therefore changes the
evaluation population; common-panel analysis is needed for claims on the same
languages, rather than for the existing within-regime rankings.

The owner attributes the observed 24–28% throughput reductions for three
final200 configurations relative to final96 to cluster communication
variability. This is the reported explanation, not an independently verified
cause: the collected counters do not include enough communication or hardware
telemetry to establish it. Keep the measured runtimes and throughput unchanged;
do not interpret these differences alone as failures or pure architecture costs.

HMoRA's alpha/sqrt(r) scaling is faithful to the original approach, rather
than a configuration error. It remains important to report that this differs
from alpha/r scaling: alpha=2r does not give every method a constant multiplier.

### Other reporting limitations

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
