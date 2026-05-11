# Noctua2 Rank Sweep Status

Updated: 2026-05-10 14:04:52 UTC

Profile: `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated`
CSV: `outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated/rank_sweep_completion_status.csv`

## Summary

- Intended runs: 55
- Training adapters present: 55/55
- Eval outputs complete: 55/55
- Eval outputs missing or partial: 0

## Missing Or Partial Eval

All intended eval outputs are present.

## Timeboxed Training Checkpoints

| Run ID | Kind | Step | Max steps | Epoch | Root |
|---|---|---:|---:|---:|---|
| `lora_r2` | `lora` | 1880 | 2072 | 0.9076279369166399 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r4` | `lora` | 1886 | 2072 | 0.9105246218216929 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r8` | `lora` | 1898 | 2072 | 0.9163179916317992 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r16` | `lora` | 1889 | 2072 | 0.9119729642742195 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r32` | `lora` | 1929 | 2072 | 0.9312841969745735 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r64` | `lora` | 1904 | 2072 | 0.9192146765368523 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r128` | `lora` | 1463 | 2072 | 0.706393594334688 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r256` | `lora` | 1408 | 2072 | 0.6798374441717298 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r512` | `lora` | 1301 | 2072 | 0.6281736611274293 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `lora_r1024` | `lora` | 1417 | 2072 | 0.6841829960165775 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_kvdown` |
| `cola_r2` | `cola` | 966 | 2072 | 0.4664225646803203 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `cola_r4` | `cola` | 966 | 2072 | 0.4664225646803203 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `cola_r8` | `cola` | 968 | 2072 | 0.46738824286806424 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `cola_r16` | `cola` | 961 | 2072 | 0.46400836921096045 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `cola_r32` | `cola` | 957 | 2072 | 0.4620770128354726 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `cola_r64` | `cola` | 919 | 2072 | 0.44372912726833785 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `cola_r128` | `cola` | 877 | 2072 | 0.4234498853257152 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `cola_r256` | `cola` | 1144 | 2072 | 0.5523679233895304 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_kvdown` |
| `hydralora_r2` | `hydralora` | 948 | 2072 | 0.45763939174511226 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `hydralora_r4` | `hydralora` | 892 | 2072 | 0.4306924717337947 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `hydralora_r8` | `hydralora` | 910 | 2072 | 0.4393835754234901 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `hydralora_r16` | `hydralora` | 904 | 2072 | 0.43648654086025834 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `hydralora_r32` | `hydralora` | 901 | 2072 | 0.4350380235786424 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `hydralora_r64` | `hydralora` | 866 | 2072 | 0.41813865529312355 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `hydralora_r128` | `hydralora` | 817 | 2072 | 0.39447953969339716 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `hydralora_r256` | `hydralora` | 1350 | 2072 | 0.6518327767271557 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_kvdown` |
| `moelora_tr16_e8_k2` | `moelora` | 1733 | 2072 | 0.8367601496801191 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly` |
| `moelora_tr32_e8_k2` | `moelora` | 1684 | 2072 | 0.8131010340803927 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly` |
| `moelora_tr64_e8_k2` | `moelora` | 1698 | 2072 | 0.8198607813946003 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly` |
| `moelora_tr128_e8_k2` | `moelora` | 1689 | 2072 | 0.8155152295497525 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly` |
| `moelora_tr256_e8_k2` | `moelora` | 1720 | 2072 | 0.8304832414597835 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly` |
| `moelora_tr512_e8_k2` | `moelora` | 1685 | 2072 | 0.8135838731742647 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly` |
| `mola_tr16_e8_k6` | `mola` | 687 | 2072 | 0.331643736422882 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mola_tr32_e8_k6` | `mola` | 686 | 2072 | 0.3311609944484673 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mola_tr64_e8_k6` | `mola` | 665 | 2072 | 0.3210234129857591 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mola_tr128_e8_k6` | `mola` | 668 | 2072 | 0.32247163890900316 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mola_tr256_e8_k6` | `mola` | 300 | 2072 | 0.14485172816159014 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mola_tr512_e8_k6` | `mola` | 296 | 2072 | 0.14292037178610228 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mola_tr1024_e8_k6` | `mola` | 617 | 2072 | 0.29791172091900375 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_kvdown` |
| `movlora_e8` | `movlora` | 1977 | 2072 | 0.9544576762149984 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `hmora_tr16_e8_k2` | `hmora` | 17402 | 24853 | 0.7001971592966644 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full` |
| `hmora_tr32_e8_k2` | `hmora` | 17356 | 24853 | 0.6983462761034885 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full` |
| `hmora_tr64_e8_k2` | `hmora` | 17402 | 24853 | 0.7001971592966644 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full` |
| `hmora_tr128_e8_k2` | `hmora` | 16959 | 24853 | 0.6823723494145576 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full` |
| `hmora_tr256_e8_k2` | `hmora` | 17402 | 24853 | 0.7001971592966644 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full` |
| `hmora_tr512_e8_k2` | `hmora` | 17201 | 24853 | 0.6921096044743089 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full` |
| `hmora_tr1024_e8_k2` | `hmora` | 14595 | 24853 | 0.58725304792178 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full` |
| `mtllora_r2` | `mtllora` | 1860 | 2072 | 0.8979000724112962 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mtllora_r4` | `mtllora` | 1801 | 2072 | 0.8694882523334406 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mtllora_r8` | `mtllora` | 1819 | 2072 | 0.8781783070485999 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mtllora_r16` | `mtllora` | 1818 | 2072 | 0.877695526231091 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mtllora_r32` | `mtllora` | 1832 | 2072 | 0.884454457676215 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mtllora_r64` | `mtllora` | 1769 | 2072 | 0.8540392661731574 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mtllora_r128` | `mtllora` | 1159 | 2072 | 0.55961050979761 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated` |
| `mtllora_r256` | `mtllora` | 1479 | 2072 | 0.7141190198366394 | `moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_kvdown` |
