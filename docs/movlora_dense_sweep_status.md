# MoVLoRA Dense Expert Sweep Status

Updated: 2026-05-12 13:45 Europe/Brussels

## Scope

This replaces the earlier MoVLoRA expert sweep rows that varied `top_k`.
For the clean expert analysis, MoVLoRA is run with dense mixing only:

| run_id | experts | top_k | rank | lora target | status |
| --- | ---: | ---: | ---: | --- | --- |
| `mov_dense_e2` | 2 | 0 | 2 | `k_proj,v_proj,down_proj` | training submitted |
| `mov_dense_e4` | 4 | 0 | 2 | `k_proj,v_proj,down_proj` | training submitted |
| `mov_dense_e8` | 8 | 0 | 2 | `k_proj,v_proj,down_proj` | training submitted |
| `mov_dense_e16` | 16 | 0 | 2 | `k_proj,v_proj,down_proj` | training submitted |
| `mov_dense_e32` | 32 | 0 | 2 | `k_proj,v_proj,down_proj` | training submitted |

`top_k=0` is the dense-mixing setting; this is intentional.

## Profiles And Data

| artifact | path |
| --- | --- |
| source dense profile | `configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_movlora_dense_expert_sweep_sf0005.generated.json` |
| calibrated profile | `configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_movlora_dense_expert_sweep_sf0005_calibrated.generated.json` |
| clean tokenized data | `outputs/pipeline_runs/moe_study_movlora_dense_expert_sweep_sf0005/tokenized_clean` |
| calibration root | `outputs/pipeline_runs/moe_study_movlora_dense_expert_sweep_sf0005_batch_calibration` |
| calibrated output root | `outputs/pipeline_runs/moe_study_movlora_dense_expert_sweep_sf0005_calibrated` |

Tokenization used `sample_fraction=0.0005` from `outputs/sharded_samples`.
The tokenization merge and parallel health check completed successfully.

## Calibration

Calibration used 20 training steps for each dense run and batch sizes
`1,2,4,8,10,12,14,16` on 4 A100 GPUs.

| run_id | selected per-device batch | grad_accum | global batch | failed larger batches | note |
| --- | ---: | ---: | ---: | --- | --- |
| `mov_dense_e2` | 12 | 1 | 48 | 14, 16 | selected highest stable batch |
| `mov_dense_e4` | 12 | 1 | 48 | 14, 16 | selected highest stable batch |
| `mov_dense_e8` | 12 | 1 | 48 | 14, 16 | selected highest stable batch |
| `mov_dense_e16` | 12 | 1 | 48 | 14, 16 | selected highest stable batch |
| `mov_dense_e32` | 12 | 1 | 48 | 14, 16 | selected highest stable batch |

Batch sizes 14 and 16 failed with CUDA OOM during probing.
The successful probes reached high GPU utilization during the actual training steps.

## Submitted Jobs

Each run has an independent train -> final eval loss -> benchmark chain.
Final eval and benchmark are configured with strict final-checkpoint checks:
`strict_final_checkpoint=true`, `require_direct_save_marker=true`,
`checkpoint_step_source=trainer_state`.

| run_id | train job | final eval job | benchmark job |
| --- | ---: | ---: | ---: |
| `mov_dense_e2` | 32635285 | 32635286 | 32635287 |
| `mov_dense_e4` | 32635288 | 32635289 | 32635290 |
| `mov_dense_e8` | 32635294 | 32635295 | 32635296 |
| `mov_dense_e16` | 32635291 | 32635292 | 32635293 |
| `mov_dense_e32` | 32635282 | 32635283 | 32635284 |

The training jobs were running when this note was written; eval and benchmark
jobs were pending on their training dependencies.
