# Rank Sweep Pipeline Report

Updated: 2026-05-13 14:58:41 UTC

- Workbook: `outputs/pipeline_runs/moe_study_rank_sweep_pipeline_report/rank_pipeline_report.xlsx`
- Source manifest: `outputs/pipeline_runs/moe_study_rank_sweep_pipeline_report/source_manifest.csv`
- Audit status: `outputs/pipeline_runs/moe_study_rank_sweep_pipeline_report/audit_status.csv`
- Clean runs in main report: `55`
- Audit-only runs/attempts: `10`

## Clean Runs By Approach

| approach | runs |
|---|---:|
| `cola` | 8 |
| `hmora` | 7 |
| `hydralora` | 8 |
| `lora` | 10 |
| `moelora` | 6 |
| `mola` | 7 |
| `movlora` | 1 |
| `mtllora` | 8 |

## Audit Status Counts

| status | count |
|---|---:|
| `probe_or_incomplete` | 10 |

Main workbook sheets contain only clean runs with 42 benchmark tasks. Audit sheets document probe-only, missing, incomplete, and infeasible attempts.
