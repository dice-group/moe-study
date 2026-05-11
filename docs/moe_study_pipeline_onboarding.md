# MoE Study Pipeline Onboarding

This document replaces the older note in `papers/further/moe_study_pipeline_onboarding.md` and is intended to be the tracked onboarding reference for agents working on the MoE-study pipeline.

For repository-level goals, project split, environments, and the full approach inventory, read `docs/AGENTS.md` first. This file is the more operational companion focused on the MoE-study pipeline.

## Scope

The MoE-study work in this repo currently spans two main execution paths:

- pipeline profiles under `configs/pipeline/profiles/moe_study_profile.json`
- SLURM comparison jobs under `scripts/slurm/comparison/`

Use the pipeline profiles for controlled repo-integrated runs and smoke tests. Use the comparison path for broader multi-GPU cluster experiments.

## Main Profiles

Current MoE-study profiles:

- `moe_study_profile_otus_cluster`
- `moe_study_profile_lola_gpuvm`
- `moe_study_profile_noctua2_a100_10_langs`
- `moe_study_profile_noctua2_a100_4gpu_10_langs`

Defined in:

- `configs/pipeline/profiles/moe_study_profile.json`

## Main Entrypoint

```bash
python -m pipeline.run --config configs/pipeline --profile moe_study_profile_otus_cluster
```

## Current Stage Behavior

Pipeline stage order:

1. `sample`
2. `tokenizer_extend`
3. `embedding_init`
4. `tokenize`
5. `audit`
6. `train`
7. `embedding_verify`
8. `benchmark`
9. `baseline_eval`
10. `summary`
11. `report`

Note: the pipeline also supports an optional `final_eval_loss` stage (between `train` and `embedding_verify`) to compute a final validation loss from the tokenized dataset after training completes.

For the MoE-study profiles, most day-to-day work is concentrated in:

- `audit`
- `train`
- `summary`

Benchmark and report are currently disabled in the main study profiles.

## Config Resolution

Pipeline config bundle behavior:

- loads defaults from `configs/pipeline/defaults/*.json`
- loads profile fragments from `configs/pipeline/profiles/*.json`
- resolves profile inheritance at runtime in `pipeline/runtime.py`

Useful implication:

- when changing a profile, always inspect both the profile block and the inherited defaults

## Current `moe_study_profile_otus_cluster`

High-level intent:

- cluster execution on Otus via the pipeline `slurm` executor
- training from pre-tokenized data
- short controlled runs by default

Current executor setup:

- `type=slurm`
- `partition=gpu`
- `gres=gpu:h100:1`
- `cpus_per_task=32`
- `mem=128G`
- `time=27:00:00`
- Conda env `merlin`

Important interpretation:

- this profile is suitable for pipeline-driven cluster jobs
- it is currently a single-GPU executor profile, not a multi-node comparison launcher

## Current `moe_study_profile_lola_gpuvm`

This profile extends the Otus study profile but switches execution to local GPU.

Current behavior:

- `executor.type=local`
- `executor.cuda_visible_devices=1`
- `flash_attn=sdpa`

This is the easiest profile to use for local GPU validation runs.

## Current `moe_study_profile_noctua2_a100_10_langs`

This profile adapts the 10-language MoE-study cluster setup for Noctua2.

Current behavior:

- `executor.type=slurm`
- `partition=gpu`
- `gres=gpu:a100:1`
- `cpus_per_task=32`
- `mem=192G`
- Conda base `/opt/software/pc2/EB-SW/software/Miniforge3/24.1.2-0`
- Conda env `merlin`
- CUDA module `system/CUDA/12.6.0`
- training `flash_attn=sdpa`

Use this profile for A100 single-GPU pipeline checks and expert/head calibration probes on Noctua2.

For full-node Noctua2 calibration, use:

- `moe_study_profile_noctua2_a100_4gpu_10_langs`

This requests `gres=gpu:a100:4`, uses 4-process FSDP via `configs/accelerate/fsdp_4gpu_config.yaml`,
and is the preferred setup when a worst-case expert/head anchor does not fit on a single 40 GB A100.

## Current Run Matrix

The MoE-study profile currently includes these run ids:

- `lora`
- `hydralora`
- `moelora`
- `mixlora`
- `adamole`
- `adamoe_1` with `kind = "mola"`
- `mov` with `kind = "movlora"`
- `hmora`

This means MixLoRA is now part of the active study profile and is not just a requested future addition.

`moelpr` still exists in the repository as an implemented tuner, but it is considered obsolete for the current planned experiments and should be treated as legacy-only unless there is a specific reason to touch it.

## MixLoRA in the Study Pipeline

Current MixLoRA state in the study profile:

- run id: `mixlora`
- kind: `mixlora`
- attention LoRA targets: `q_proj,k_proj,v_proj,o_proj`
- FFN expert targets: `gate_proj,down_proj,up_proj`
- router aux loss enabled
- top-k router enabled with `top_k=2`

Current integration layers:

- tuner: `LLaMA-Factory/src/peft/tuners/mixlora`
- backend dispatch: `LLaMA-Factory/src/llamafactory/model/adapter.py`
- pipeline CLI mapping: `pipeline/training.py`
- train verification: `pipeline/checks.py`
- comparison job: `scripts/slurm/comparison/mixlora_job.sh`

Reference:

- `docs/mixlora_pipeline_integration_report.md`

## How Training Is Wired

Pipeline training command synthesis happens in:

- `pipeline/training.py`

Backend method dispatch happens in:

- `LLaMA-Factory/src/llamafactory/model/adapter.py`

Finetuning argument definitions live in:

- `LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py`

Train-stage adapter verification lives in:

- `pipeline/checks.py`

When a new approach is added, it is not enough to only implement the tuner. It also needs:

- pipeline argument mapping
- profile entries
- checkpoint verification support
- comparison job/config support if it is part of the study matrix

## Comparison Pipeline

The SLURM comparison path is separate from the generic pipeline executor and is more suitable for large runs.

Main files:

- `scripts/slurm/comparison/run_comparison.py`
- `scripts/slurm/comparison/matrix_runner.sh`
- `scripts/slurm/comparison/comparison_config*.json`
- `scripts/slurm/comparison/mixlora_job.sh`
- `scripts/slurm/comparison/accelerate_launch.sh`

Important current behavior:

- `run_comparison.py` automatically injects an `accelerate` config based on GPU count
- `mixlora_job.sh` is correct in that workflow
- this is the preferred path for multi-GPU comparison jobs

## Expert/Head Sweep Calibration

The expert/head-count sweep helper is:

- `scripts/analysis/expert_head_sweep.py`

For Noctua2 A100 calibration:

```bash
python scripts/analysis/expert_head_sweep.py write-profile --cluster noctua2
python scripts/analysis/expert_head_sweep.py validate-profile --cluster noctua2
python scripts/analysis/expert_head_sweep.py list-anchors --cluster noctua2
```

For full-node 4x A100 calibration, replace `noctua2` with `noctua2_4gpu`.

Submit one wave of batch probes:

```bash
python scripts/analysis/expert_head_sweep.py submit-wave --cluster noctua2 --max-submissions 8
```

Collect completed probes:

```bash
python scripts/analysis/expert_head_sweep.py collect-results --cluster noctua2 --cleanup-successful
```

After every active approach has at least one successful probe, write the calibrated profile:

```bash
python scripts/analysis/expert_head_sweep.py write-calibrated-profile --cluster noctua2
```

The Noctua2 probe path records GPU telemetry under:

- `outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_batch_search/`

Current Noctua2 A100 result:

- The single-GPU A100 path is not sufficient for the worst expert anchors.
- The working Noctua2 sweep profile is:
  - `moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_working_calibrated`
  - `configs/pipeline/profiles/moe_study_expert_head_sweep_r2_noctua2_4gpu_working_calibrated.generated.json`
- The current ready-for-longer-run profile with softer stability reductions is:
  - `moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_ready`
  - `configs/pipeline/profiles/moe_study_expert_head_sweep_r2_noctua2_4gpu_ready.generated.json`
- It uses 4x A100 FSDP and 65 working runs.
- AdaMoLE and MoE-LoRA are capped at 4 experts on 40 GB A100 because e8/e16/e32 anchors OOM at batch 1.
- The measured approach settings are in:
  - `outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_4gpu_batch_search_v2/approach_batch_settings_working.csv`
  - `outputs/pipeline_runs/noctua2_best_config_stability_soft_reduce_10step/approach_batch_settings_ready.csv`

Measured per-device batch settings for the ready profile:

| approach | max experts used | batch | grad accum | notes |
| --- | ---: | ---: | ---: | --- |
| MoE-LoRA | 4 | 1 | 12 | e8/e16/e32 OOM on 40 GB A100 |
| AdaMoLE | 4 | 1 | 12 | e8/e16/e32 OOM on 40 GB A100 |
| MoLA (`adamoe_1`) | 32 | 7 | 2 | softer reduction from bs8; bs10 failed |
| MoV | 32 | 11 | 1 | softer reduction from bs12; bs16 failed |
| HMoRA | 32 | 1 | 12 | bs2 failed; checkpointing-on failed at bs1 |
| CoLA | 32 | 8 | 1 | PiSSA initialization is slow, training succeeds |
| HydraLoRA | 32 | 2 | 6 | bs4 failed |
| MTL-LoRA | 32 | 11 | 1 | softer reduction from bs12; bs16 failed |

To regenerate the original maximum-utilization working profile from the collected probes:

```bash
python scripts/analysis/expert_head_sweep.py write-calibrated-profile \
  --cluster noctua2_4gpu \
  --results outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_4gpu_batch_search_v2/probe_results.csv \
  --extra-results outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_4gpu_reduced_anchor_search_e4/probe_results.csv \
  --target-global-batch 48 \
  --calibrated-profile moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_working_calibrated \
  --output-profile configs/pipeline/profiles/moe_study_expert_head_sweep_r2_noctua2_4gpu_working_calibrated.generated.json \
  --output-root outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_working_calibrated \
  --output-settings outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_4gpu_batch_search_v2/approach_batch_settings_working.csv \
  --cap-approach-max-experts adamole=4 \
  --cap-approach-max-experts moelora=4
```

## Known Gaps and Caveats

### 1. Missing `configs/moelpr_lang_map.json`

That file is still missing from the repo. Any profile or command path that assumes it exists needs to be checked carefully.

### 2. Generic pipeline Slurm vs comparison Slurm

There are two different cluster paths:

- generic pipeline `slurm` executor in `pipeline/executor.py`
- comparison launcher path in `scripts/slurm/comparison/`

Use the generic pipeline executor for standard pipeline jobs. Use the comparison path for multi-GPU comparison launches.

### 3. Benchmark scope is still limited

Current pipeline benchmarking is not yet the full desired multilingual evaluation stack. `SIB200` and `TAXI150` are still not fully integrated.

## Fast Commands

Inspect the profile:

```bash
sed -n '1,260p' configs/pipeline/profiles/moe_study_profile.json
```

List the current run matrix:

```bash
jq '.moe_study_profile_otus_cluster.training.runs[] | {id, kind, train, lora, params}' \
  configs/pipeline/profiles/moe_study_profile.json
```

Run only MixLoRA locally:

```bash
python -m pipeline.run \
  --config configs/pipeline \
  --profile moe_study_profile_lola_gpuvm \
  --run-id mixlora \
  --start-at train \
  --stop-after summary
```

Run the local quick smoke profile with MixLoRA:

```bash
python -m pipeline.run \
  --config configs/pipeline \
  --profile local_mini_quick \
  --run-id mixlora_4exp_k2 \
  --start-at sample \
  --stop-after summary
```

## Recommended Agent Workflow

1. Start from `docs/AGENTS.md` for repo-level context.
2. Inspect `configs/pipeline/profiles/moe_study_profile.json` before changing study behavior.
3. If touching MixLoRA, also read `docs/mixlora_pipeline_integration_report.md`.
4. For cluster comparison work, inspect the relevant `comparison_config*.json` and `mixlora_job.sh`.
5. Verify changes with at least one local pipeline run or focused backend test before expanding to cluster runs.
