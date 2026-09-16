# Two-chain recovery — 2026-09-15

Only the Falcon LoRA r2 acceptance/evaluation chain and Gemma MoELoRA E32
confirmation/promotion chain were repaired. No running trainings, frozen code,
scientific settings, datasets, or completed calibration trials were changed.
Nothing was staged, committed, or pushed.

## Replacement jobs

| Stage | Old job | Replacement |
| --- | --- | --- |
| Falcon LoRA r2 acceptance | 34349876, failed | 34426145 |
| Falcon LoRA r2 benchmark, 15 shards / concurrency 4 | 34349878, cancelled | 34426146 |
| Falcon LoRA r2 final validation loss | 34349879, cancelled | 34426147 |
| Falcon LoRA r2 collector | 34349880, cancelled | 34426148 |
| Falcon final report | 34384759, cancelled | 34426151 |
| Gemma E32 calibration confirmation | 34375226, timeout | 34426149 |
| Gemma E32 gated production promoter | 34375237, cancelled | 34426150 |

Falcon training **34349875** is retained: COMPLETED, exit 0:0, final step 16734.
The original acceptance failure was stale Slurm accounting (`RUNNING` after the
training had ended). A read-only rerun of the exact acceptance function now
passes every check, with current accounting `COMPLETED`; additional campaign
telemetry checks also pass. The scheduled replacement acceptance job will write
the canonical receipt before evaluation is allowed to start.

Gemma reuses all seven completed candidate receipts, with verified unchanged
source signatures. The fastest eligible candidate is `b1_gc1`: microbatch 1,
gradient accumulation 12, four GPUs, GBS 48, gradient checkpointing enabled.
Measured median step: 43.3038 seconds; free VRAM margin: 23.25%.
Its incomplete confirmation and checkpoint were moved, not deleted, to
`archive/gemma/interrupted_confirmation_34375226/`, with a SHA-256 inventory.
The retry starts a **fresh 120-step confirmation**, not a resumed partial trial.

The confirmation requests **one node / four A100s / three hours**. Expected
training portion is about 87 minutes after startup, plus save/reload and smoke
checks; queue waiting is separate. The existing gates must still pass before
the promoter submits unchanged 34-hour production training, acceptance,
benchmarks, final loss, collection, and eventually the Gemma report/correlations.
Only previously documented bad nodes n2gpu1211 and n2gpu1213 are excluded.

## Verification and current status

- Six orchestration unit tests passed (target isolation, idempotence, ambiguous
  submission fail-closed behavior, terminal state checks, candidate replay).
- Dry runs verified both frozen campaigns before live submission.
- Slurm confirms the intended node/GPU counts and valid `afterok` dependencies.
- The restored Falcon report depends on the new LoRA r2 collector and the four
  other unfinished Falcon collectors; already verified complete results are reused.
- Unrelated original production/evaluation entries and recovery pilot/promoter
  entries are unchanged against the archived pre-repair receipts.
- At 22:57 CEST, both retry entry jobs were PENDING (Priority);
  downstream jobs were PENDING (Dependency), not DependencyNeverSatisfied.
  Actual GPU execution of these retries is not yet verified.

## Artifacts and reproducibility

`repair_jobs.json` journals every new submission immediately. `archive/` retains
the old receipts, failed acceptance evidence, and interrupted confirmation.
`repair.py` is a local operational helper outside both frozen code signatures.
Use the original checkout/PYTHONPATH for `falcon`, and the frozen recovery
checkout/PYTHONPATH for `gemma` and `report`. Without `--submit`, checks are
read-only; completed journaled actions never resubmit themselves.

Tests (from the repository root, using its normal campaign PYTHONPATH):

```bash
/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python -m unittest discover \
  -s tmp/cross_arch_repairs_20260915 -p test_repair.py -v
```
