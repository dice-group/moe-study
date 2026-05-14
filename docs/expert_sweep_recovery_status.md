# Expert Sweep Recovery Status

Updated: 2026-05-13 Europe/Brussels

This note tracks the recovery reruns for the expert sweep rows whose original
checkpoints or benchmark outputs were not clean enough for final analysis.

## Current Summary

| family | recovery profile | scope | status |
| --- | --- | --- | --- |
| `moelora` | `configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_moelora_attention_sizefsdp_calibrated.generated.json` | attention-target rerun for `e8/e16/e32`, excluding infeasible `e32_k16` | complete: 11/11 clean adapter checkpoints, final eval, benchmark |
| `movlora` | `configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_movlora_dense_expert_sweep_sf0005_calibrated.generated.json` | dense mixing only, experts `2,4,8,16,32`, `top_k=0` | complete: 5/5 clean adapter checkpoints, final eval, benchmark |
| `hmora` | `configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_hmora_attention_calibrated.generated.json` | attention-target rerun for `e2_k1` and `e32_k1/k2/k4/k8/k16` | training running; final eval and benchmark dependency-pending |
| `hydralora` | `configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_hydralora_heads32_rerun.generated.json` | all-target rerun for `hydralora_r2_heads32` only | training running; final eval, benchmark, and report dependency-pending |
| `remaining_problematic` | `configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_remaining_problematic_recovery.generated.json` | `moelora_r2_e2_k1`, `adamole_r2_e8`, attention-only `adamole_r2_e16`, attention-only `adamoe_1_r2_e32_k16` | complete: 4/4 clean adapter checkpoints, final eval, benchmark |
| `mtllora` | `configs/pipeline/profiles/moe_study_mtllora_clean_recovery_eval.generated.json` | recovered `checkpoint-2260_adapter` dirs for `nup2/nup4/nup8/nup16/nup32` from completed final saves | clean final eval complete; benchmarks running |

Clean checkpoint means the adapter directory contains
`.adapter_direct_save_complete`, produced by the adapter-only SLURM/JIT signal
save path.

## Completed MoELoRA Reruns

Output root:
`outputs/pipeline_runs/moe_study_moelora_attention_sizefsdp_calibrated`

| run_id | step | train runtime sec | checkpoint | final eval | benchmark |
| --- | ---: | ---: | --- | --- | --- |
| `moelora_r2_e8_k1` | 1837 | 12840.0 | `checkpoint-1837_adapter` | yes | yes |
| `moelora_r2_e8_k2` | 1588 | 12828.4 | `checkpoint-1588_adapter` | yes | yes |
| `moelora_r2_e8_k4` | 1241 | 12853.4 | `checkpoint-1241_adapter` | yes | yes |
| `moelora_r2_e16_k1` | 1702 | 12834.4 | `checkpoint-1702_adapter` | yes | yes |
| `moelora_r2_e16_k2` | 1363 | 12838.7 | `checkpoint-1363_adapter` | yes | yes |
| `moelora_r2_e16_k4` | 1047 | 12852.2 | `checkpoint-1047_adapter` | yes | yes |
| `moelora_r2_e16_k8` | 882 | 12843.7 | `checkpoint-882_adapter` | yes | yes |
| `moelora_r2_e32_k1` | 1581 | 12848.7 | `checkpoint-1581_adapter` | yes | yes |
| `moelora_r2_e32_k2` | 1273 | 12861.3 | `checkpoint-1273_adapter` | yes | yes |
| `moelora_r2_e32_k4` | 923 | 12839.1 | `checkpoint-923_adapter` | yes | yes |
| `moelora_r2_e32_k8` | 749 | 12833.3 | `checkpoint-749_adapter` | yes | yes |

`moelora_r2_e32_k16` remains excluded for now as infeasible on the current A100
setup.

## Completed Dense MoVLoRA Reruns

Output root:
`outputs/pipeline_runs/moe_study_movlora_dense_expert_sweep_sf0005_calibrated`

| run_id | experts | top_k | step | train runtime sec | checkpoint | final eval | benchmark |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `mov_dense_e2` | 2 | 0 | 2441 | 12705.5 | `checkpoint-2441_adapter` | yes | yes |
| `mov_dense_e4` | 4 | 0 | 2391 | 12769.9 | `checkpoint-2391_adapter` | yes | yes |
| `mov_dense_e8` | 8 | 0 | 2399 | 12800.6 | `checkpoint-2399_adapter` | yes | yes |
| `mov_dense_e16` | 16 | 0 | 2287 | 12760.2 | `checkpoint-2287_adapter` | yes | yes |
| `mov_dense_e32` | 32 | 0 | 2359 | 12775.5 | `checkpoint-2359_adapter` | yes | yes |

The earlier `mov_e*_k*` top-k rows should be treated as historical bookkeeping,
not as the clean MoVLoRA expert sweep. The clean MoVLoRA analysis should use
the dense rows above.

## Active HMoRA Reruns

Output root:
`outputs/pipeline_runs/moe_study_hmora_attention_calibrated`

HMoRA was switched to attention targets after the original all-target rows were
not reliable. Calibration selected per-device batch size `2`, gradient
accumulation `6`, and disabled gradient checkpointing for the six recovery
rows. The train -> final-eval-loss -> benchmark chains are submitted.

| run_id | train job | final eval job | benchmark job | current state |
| --- | ---: | ---: | ---: | --- |
| `hmora_r2_e2_k1` | 32635596 | 32635602 | 32635608 | training running |
| `hmora_r2_e32_k1` | 32635597 | 32635603 | 32635609 | training running |
| `hmora_r2_e32_k2` | 32635598 | 32635604 | 32635610 | training running |
| `hmora_r2_e32_k4` | 32635599 | 32635605 | 32635611 | training running |
| `hmora_r2_e32_k8` | 32635600 | 32635606 | 32635612 | training running |
| `hmora_r2_e32_k16` | 32635601 | 32635607 | 32635613 | training running |

As of this update, no HMoRA signal-saved adapter checkpoint exists yet; that is
expected until the SLURM signal window is reached. The eval and benchmark jobs
are dependency-pending and should start only after successful training exit.

## Active HydraLoRA Heads32 Rerun

Output root:
`outputs/pipeline_runs/moe_study_hydralora_heads32_rerun`

The original `hydralora_r2_heads32` row came from a short probe and should not
be used. The rerun keeps all LoRA targets, uses the updated no-prefetch sharded
FSDP config, excludes slow nodes, and relies on adapter-only SLURM/JIT signal
saving for the final checkpoint.

| run_id | train job | final eval job | benchmark job | report job | current state |
| --- | ---: | ---: | ---: | ---: | --- |
| `hydralora_r2_heads32` | 32636254 | 32636255 | 32636256 | 32636257 | training running on `n2gpu1203`; dependents pending |

## Completed Remaining Problematic Reruns

Output root:
`outputs/pipeline_runs/moe_study_remaining_problematic_recovery`

The corresponding 20-step probe profile was
`configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_remaining_problematic_probe.generated.json`.
The probes validated the runnable variants before full-budget submission.

| run_id | target policy | probe result | step | runtime h | checkpoint | final eval | benchmark |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `moelora_r2_e2_k1` | all targets | passed 20 steps | 1546 | 3.5415 | `checkpoint-1546_adapter` | yes | 42/42 |
| `adamole_r2_e8` | all targets | passed 20 steps | 195 | 3.5690 | `checkpoint-195_adapter` | yes | 42/42 |
| `adamole_r2_e16` | attention only | all-target OOM; attention-only passed 20 steps | 215 | 3.5540 | `checkpoint-215_adapter` | yes | 42/42 |
| `adamoe_1_r2_e32_k16` | attention only | all-target MLP shape mismatch; attention-only passed 20 steps | 121 | 3.5588 | `checkpoint-121_adapter` | yes | 42/42 |

`adamole_r2_e32` remains excluded for now: all-target and attention-only probes
both OOM at per-device batch size `1` on 4x A100. Do not schedule it again
without a model/configuration change.

## MTL-LoRA Checkpoint Trust Note

The five existing MTL-LoRA rows completed one full epoch close to the target
budget. The original eval/benchmark outputs should not be used because the old
adapter files had saved tensor keys including the adapter name (`.default`).
PEFT expects saved adapter files without that suffix and reinserts it during
load, so the old files triggered missing-key warnings and were not applied
cleanly in eval.

| run_id | train step | runtime h | final save type | final eval | benchmark tasks | trust decision |
| --- | ---: | ---: | --- | --- | ---: | --- |
| `mtllora_r2_nup2` | 2260/2260 | 3.43 | recovered `checkpoint-2260_adapter` | clean, loss `2.3130574226379395` | old partial 24 | running job 32636523 |
| `mtllora_r2_nup4` | 2260/2260 | 3.38 | recovered `checkpoint-2260_adapter` | clean, loss `2.3139915466308594` | old partial 22 | running job 32636524 |
| `mtllora_r2_nup8` | 2260/2260 | 3.41 | recovered `checkpoint-2260_adapter` | clean, loss `2.3136045932769775` | old partial 22 | running job 32636525 |
| `mtllora_r2_nup16` | 2260/2260 | 3.49 | recovered `checkpoint-2260_adapter` | clean, loss `2.3136799335479736` | old partial 22 | running job 32636526 |
| `mtllora_r2_nup32` | 2260/2260 | 3.34 | recovered `checkpoint-2260_adapter` | clean, loss `2.313389301300049` | old partial 25 | running job 32636527 |

The recovered adapter directories each contain 896 MTL-LoRA tensors, zero
`.default` keys, and the expected `num_up_projections`. All five clean final
eval reruns loaded `checkpoint-2260_adapter` with no missing-key warnings and no
tracebacks. Retraining is not required unless the running benchmark reruns reveal
a new failure.
