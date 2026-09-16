# Execution safeguards

These are shared pipeline capabilities. Model choices, Slurm resources, GPU
requirements, and scientific budgets belong in experiment profiles.

- `preflight.enabled` opts into GPU checks. Configure `required_gpu_count`,
  `required_gpu_name`, `timeout_seconds`, and optional ECC, NVLink, and
  distributed checks. CPU stages must override inherited preflight to disabled.
- `training.tokenized_local_copy` supports a bounded transfer, Arrow count/size
  checks, and SHA256 checks for explicitly listed `verify_files`. This is not a
  full dataset-content checksum. `cleanup` requires `job_scoped: true`; source
  and destination must be disjoint. Lock files remain to avoid lock-inode races.
- `train_acceptance.enabled` opts into the stage-owned acceptance check. Supply
  `target_runtime_seconds` and `runtime_tolerance_seconds`, plus optional log
  patterns, checkpoint-save-marker requirements, and Slurm completion checks.
  A receipt records observed evidence and rejection reasons; it does not establish
  statistical independence or model quality.
- Submitted chains use `train -> afterany acceptance -> afterok evaluation` for
  each run. Without acceptance, the previous `afterany` evaluation behavior stays.
  The chain passes the exact training log and job ID to acceptance.
- New logs live beneath `logs/attempts/<attempt_id>/` (or the configured log
  root). Checkpoints, summaries, and other stage artifacts keep stable paths.
  Existing logs and receipts are not migrated or rewritten.

Preview a chain using the normal entrypoint:

```bash
python -m pipeline.run --config configs/pipeline --profile <profile> \
  --submit-chain --start-at train --stop-after benchmark --dry-run
```

CPU regression checks (no scheduler submission):

```bash
python -m pytest -q pipeline/tests/test_execution_safety.py \
  pipeline/tests/test_eval_preflight_hardening.py \
  pipeline/tests/test_train_acceptance.py
```

GPU/NCCL readiness still requires a separate allocated-node check. Do not update
sealed campaign source hashes or reuse acceptance receipts after changing code;
future submissions must record their actual source revision.
