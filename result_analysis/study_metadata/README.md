# Training statistics and language sampling

## File locations

Relative paths in the CSVs and evidence files are relative to the repository root. The original cluster location is:

```text
/scratch/hpc-prf-merlin/joel/moe-study
```

For example, `outputs/...` resolves to
`/scratch/hpc-prf-merlin/joel/moe-study/outputs/...`.
Large outputs and checkpoints may be cluster-local and absent from Git;
access to that directory is required to open them. Cloning the repository
alone does not provide those artifacts. Markdown links below are relative
to this README's directory.

## Reporting tables

The requested training statistics and active-parameter counts are in the CSV files below.
Training statistics cover **159 runs**: 55 rank-sweep, 68 expert-sweep,
22 mixed-follow-up, 7 final-96, and 7 final-200 runs.

**Note** It includes original failed/excluded CoLA sweep runs that were not are not in the paper, they can be ignored, they were inlcuded for completeness.

| Information | File | Column(s) |
|---|---|---|
| Tokens processed (input positions) | [runs.csv](runs.csv) | `input_token_positions_seen` |
| Optimizer steps | [runs.csv](runs.csv) | `optimizer_steps` |
| Throughput (tokens/second, all four GPUs combined) | [runs.csv](runs.csv) | `input_token_positions_per_second` |
| Training duration (hours) | [runs.csv](runs.csv) | `trainer_runtime_hours` |
| Configured training budget (hours) | [runs.csv](runs.csv) | `training_budget_hours`, with `budget_basis` (`job_clock` or `active_training`) |
| Language sampling proportions in the prepared training data | [languages.csv](languages.csv) | `dataset_train_document_share`, `dataset_train_example_share` |
| Active adapter parameters | [runs.csv](runs.csv) | `active_adapter_parameters`; for input-dependent routing, use `active_adapter_parameters_min` / `active_adapter_parameters_max` |

In `runs.csv`, identify each training by `stage` and `run_id`. For the final
experiments, select `final_96` or `final_200` in `stage`.

Use the run's `dataset` value to find its language distribution in `languages.csv`. 
Each row identifies a `language`; multiply the share by 100 to obtain a percentage. `train_documents` and `train_examples` give the corresponding counts. [datasets csv](datasets.csv) provides dataset totals and construction recipe notes. 
Construction sampling rates and the English
exception are documented in the technical notes.

The configured budget is distinct from measured training duration. `job_clock`
includes setup; `active_training` uses the configured training timer. Scheduler
time limits are retained in the evidence files, not the reporting CSV.

`world_size=4` means four training processes, one per GPU on one node.
`global_batch = micro_batch × gradient_accumulation × world_size`.
Documents are source texts; examples are tokenized training sequences, so their counts need not match. Optimizer steps count parameter updates, not microbatches.
Training duration is trainer runtime, not the total scheduler allocation time.

The names “96 languages” and “200 languages” identify training regimes. Their datasets contain 102 and 215 language-script labels respectively.

GPU allocations are verified as A100 for every run; use `allocated_gpu_type`.
`N/A` denotes inapplicable fields or a point estimate replaced by explicit bounds, not an unresolved measurement.

For definitions, provenance, and methodological notes, see [technical notes](technical_notes.md).

Active counts are derived from saved checkpoint shapes and routing settings, not measured averages: 123 runs have configured counts and 36 have bounds. `active_parameters_basis` distinguishes configured counts from bounds. The frozen backbone is listed separately in `frozen_backbone_parameters`; active counts are not FLOP estimates.
AdaMoLE uses adptive routing and has therefore no fixed number of active paramters. Therefore I choose to document min - max bounds of active paramters.

Detailed checkpoint paths, scheduler records, and source references are in `sources.json`, `dataset_sources.json`, `scheduler_accounting.json`, and `enrichment_audit.json`. Redundant audit columns were removed from the CSVs; the committed intermediate version preserves the original detailed tables.
