# Historical source and run records

Captured on 2026-09-16 against maintained source commit
`0bd25fb228738bead58b5338c9f1664d3a2d9642`. These files document lineage, not
current job health or successful completion. No jobs or results were changed.

## Records

- `artifact_sources.json`: original/recovery source location for each run and baseline.
- `*_production_jobs.json`, `*_evaluation_jobs.json`: submission snapshots from
  the corresponding campaign directories. Status fields are historical.
- `repair_jobs.json`, `repair_notes.md`: September 15 replacement-job journal
  and its contemporaneous explanation.
- `external_artifacts.sha256`: references and hashes for larger receipts and
  local repair evidence. Run `sha256sum -c` from the repository root to check them.

## Source recovery

The two patches are historical evidence, not maintained alternative launchers.
Both apply independently to the commit above in an isolated checkout:

- `pre_cleanup_sources.patch` restores files captured before the September 16
  evaluation, execution, and campaign cleanup. It was generated from the three
  local pre-cleanup source archives; it does not delete subsequently added files.
- `historical_recovery.patch` restores the selected recovery-specific campaign
  modules, GPU preflight, and scientific JSON. It is not a complete recovery
  checkout and must not be treated as sufficient to reproduce every source hash.

Both passed `git apply --check`. Never apply them to an active campaign checkout
or modify old gate signatures to accept the maintained code. Use the recorded
source tree and frozen receipts to interpret existing runs.

## Remaining preservation limits

The original outputs remain under
`outputs/pipeline_runs/gemma2_falcon_96_languages_2026-09-13/`; the recovery
checkout remains under `tmp/cross_arch_recovery_20260914/source/`. The repair
workspace remains under `tmp/cross_arch_repairs_20260915/`.

These directories still contain unique evidence, datasets, and checkpoints.
Git does not back them up. Do not delete them until a complete external archive
has been verified. Full recovery source/signature parity has not been re-audited.
Custom generated Belebele/Taxi1500 tasks still need their generator and exact
inputs preserved; generated YAMLs intentionally remain local.

The maintained workflow passed 23 lightweight fixture tests in isolation.
The inference check timed out; fresh GPU gates are required before new training.
Workbook alpha/rank metadata corrections and reviewer evidence tables remain
separate follow-up work, not completed by this preservation pass.
