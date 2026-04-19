# Multitask Track 2 A100 Cluster Runbook

This is the canonical rollout note for Track 2 on the A100 cluster.

## Revisions

- branch: `feature/multitask-track2-pipeline`
- root commit: `d87eaafbb168a559954897424539717db0f6a861`
- `LLaMA-Factory` commit: `30e8c6e0ed3ffa4cf378f24c43d55d8d58497a1c`

## Required Local Asset

- `MTL-LoRA/commonsense_170k_taskid.json`

Do not submit any Track 2 run until that file exists in the cluster workspace.

## Required External Access

- Hugging Face access to `meta-llama/Llama-3.1-8B`
- remote-code trust for benchmark tasks such as `social_iqa`

## Environment Rules

- use the `merlin` conda env
- keep the default Hugging Face model cache behavior
- do not set `HF_HOME`
- set only scratch-backed `HF_DATASETS_CACHE` and `TMPDIR`
- set `LM_EVAL_TRUST_REMOTE_CODE=true`

Recommended submission environment:

```bash
export LM_EVAL_TRUST_REMOTE_CODE=true
export HF_DATASETS_CACHE=/scratch/hpc-prf-merlin/joel/moe-study-cache/hf_datasets_track2
export TMPDIR=/scratch/hpc-prf-merlin/joel/moe-study-cache/tmp
unset HF_HOME
```

## Cluster Profiles

- `multitask_track2_cluster_smoke`
  - quick Track 2 task mix
  - all 11 methods
  - `max_steps=5`
  - `save_steps=5`
  - `eval_steps=5`
  - `benchmark.limit=5`
  - `resize_vocab=false`
  - output root: `outputs/pipeline_runs/multitask_track2_cluster_smoke_a100`
- `multitask_track2_cluster_pilot`
  - full Track 2 task mix
  - all 11 methods
  - `max_steps=50`
  - `save_steps=50`
  - `eval_steps=50`
  - `benchmark.limit=5`
  - `resize_vocab=false`
  - output root: `outputs/pipeline_runs/multitask_track2_cluster_pilot_a100`
- `multitask_track2_cluster_full_s1`
  - full single-seed run
  - all 11 methods
  - no benchmark limit
  - `resize_vocab=false`
  - output root: `outputs/pipeline_runs/multitask_track2_cluster_full_s1_a100`

All three profiles use:

- `partition=gpu`
- `gres=gpu:a100:1`
- `mem=128G`
- `cpus_per_task=32`
- `time=27:00:00`
- modules:
  - `toolchain/foss/2024a`
  - `system/CUDA/12.6.0`
  - `lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0`
- `conda_base=/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3`
- `conda_env=merlin`
- `training.train.flash_attn=sdpa`

## Canonical Outputs

Inspect these files after every run:

- `sampled/manifest.json`
- `sampled/task_map.json`
- `runs/<run_id>/train_results.json`
- `benchmarks/benchmark_summary.json`
- `benchmarks/benchmark_metrics.csv`
- `summary.csv`
- `gate.json`
- `state/submit_chain_jobs.json`

Interpretation rules:

- `benchmarks/benchmark_summary.json` and `benchmarks/benchmark_metrics.csv` are the canonical Track 2 evaluation outputs
- `summary.csv` and `gate.json` are operational sanity outputs, not the scientific ranking source
- benchmark artifacts after a timed-out training job are not a pass
- `state/submit_chain_jobs.json` is the source of truth for the Slurm job names and job ids created by `--submit-chain`

## Commands

Sync the repo:

```bash
git checkout feature/multitask-track2-pipeline
git pull
git submodule update --init --recursive
```

Run preflight:

```bash
python scripts/cluster/preflight_multitask_track2.py
```

Run the all-method 5-step smoke:

```bash
python -m pipeline.run \
  --config configs/pipeline \
  --profile multitask_track2_cluster_smoke \
  --start-at sample \
  --stop-after summary \
  --submit-chain
```

Run the pilot sample once:

```bash
python -m pipeline.run \
  --config configs/pipeline \
  --profile multitask_track2_cluster_pilot \
  --start-at sample \
  --stop-after sample \
  --submit-chain
```

Run pilot waves from `train` through `summary`:

```bash
python -m pipeline.run \
  --config configs/pipeline \
  --profile multitask_track2_cluster_pilot \
  --start-at train \
  --stop-after summary \
  --run-id lora_r8 \
  --run-id mtllora \
  --run-id hmora \
  --submit-chain
```

Wave layout:

- Wave A: `lora_r8`, `mtllora`, `hmora`
- Wave B: `lora_r64`, `cola`, `hydralora`
- Wave C: `moelora`, `adamole`, `adamoe_1`, `mov`, `mixlora`

## Failure Policy

- missing `MTL-LoRA/commonsense_170k_taskid.json`: stop immediately and stage the asset
- benchmark remote-code failure on `social_iqa`: fix env or trust settings, then rerun the same phase
- training timeout with benchmark output present: treat as failure
- repeated failure inside one wave: stop and investigate before later waves
- if a method fails even at `batch_size=1`, do not silently change the study configuration
