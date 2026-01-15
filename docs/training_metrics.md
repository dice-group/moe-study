# Training Metrics (W&B)

This document lists every metric emitted during training for the five PEFT approaches in this repo. The names below
match what appears in W&B.

Sources:
- Transformers Trainer: standard training/eval metrics (loss, lr, runtime, throughput).
- `LLaMA-Factory/src/peft/metrics.py`: routing metrics aggregated per logging interval.
- `LLaMA-Factory/src/llamafactory/train/sft/trainer.py`: aux losses and language-prior metrics.

Notes:
- Routing metrics are token-weighted averages over the logging interval and are prefixed with `train/`.
- Target metrics are only meaningful when language targets are provided; otherwise they are logged as zeros.

## Common Trainer Metrics (all approaches)
| Metric | What it measures | Healthy | Unhealthy |
| --- | --- | --- | --- |
| `loss` or `train/loss` | Main training loss | Downward trend or stable plateau | Diverging, NaN, or oscillating wildly |
| `learning_rate` | LR schedule value | Matches expected schedule | Flat when it should decay, or spikes |
| `epoch` | Training progress in epochs | Monotonic increase | Stuck or jumps unexpectedly |
| `eval/loss` (if eval) | Eval loss on validation data | Tracks or improves vs train | Large train/eval gap or rising eval loss |
| `train_runtime`, `train_steps_per_second`, `train_samples_per_second` (end of run) | Throughput summary | Stable across runs | Sudden drops, large regressions |

## AdaMoLE (`train/adamole/*`, `adamole_aux_loss`)
| Metric | What it measures | Healthy | Unhealthy |#
| --- | --- | --- | --- |
| `train/adamole/active_token_pct` | Tokens routed to at least one expert | Near 1.0 | Low values (many tokens unrouted) |
| `train/adamole/zero_routed_pct` | Tokens routed to no expert | Near 0.0 | High values (dead routing) |
| `train/adamole/experts_per_token` | Avg experts selected per token | Matches intended gating | Very low or very high vs intent |
| `train/adamole/expert_load_cv` | Load balance variation across experts | Low (close to 0) | High (>1) indicates imbalance |
| `train/adamole/threshold_mean` | Mean routing threshold | Stable, within expected range | Very high -> over-pruning; very low -> too permissive |
| `train/adamole/threshold_std` | Threshold variability | Moderate and stable | Large swings indicate unstable gating |
| `train/adamole/margin_mean` | Mean adapted logit margin (gate - threshold) | Positive, stable | Near 0 or negative -> weak routing |
| `adamole_aux_loss` | Load-balancing aux loss | Non-zero, stable | Exploding or collapsing to 0 early |

## MoLA / AdaMoE (`train/mola/*`, `mola_aux_loss`)
| Metric | What it measures | Healthy | Unhealthy |
| --- | --- | --- | --- |
| `train/mola/null_token_pct` | Tokens whose top-k includes null expert | Low unless intentionally high | High values -> router over-uses null |
| `train/mola/null_path_pct` | Fraction of expert assignments to null expert | Low | High indicates collapse to null |
| `train/mola/expert_load_cv` | Load balance variation across true experts | Low | High (>1) indicates imbalance |
| `train/mola/router_entropy` | Avg router entropy over experts | Moderate (not 0, not max) | Near 0 -> collapse; near max -> no specialization |
| `train/mola/active_expert_frac` | Fraction of experts receiving tokens | High | Low -> many dead experts |
| `mola_aux_loss` | MoLA load-balancing aux loss | Non-zero, stable | Exploding or collapsing to 0 early |

## CoLA (`train/cola/*`)
| Metric | What it measures | Healthy | Unhealthy |
| --- | --- | --- | --- |
| `train/cola/expert_load_cv` | Load balance variation | Low | High imbalance |
| `train/cola/active_expert_frac` | Fraction of experts active | High | Low -> dead experts |
| `train/cola/router_entropy` | Router entropy | Moderate | Collapse (near 0) or uniform (near max) |
| `train/cola/topk_weight_mean` | Mean weight of selected experts | Around `1/top_k` | Near 1 -> single expert dominates |
| `train/cola/expert_load_max_frac` | Largest expert load fraction | Close to `1/num_experts` | Very high -> one expert dominates |
| `train/cola/expert_load_min_frac` | Smallest expert load fraction | Non-zero | Near 0 -> unused experts |
| `train/cola/language_target_hit_rate` | Top-1 equals language target | High if guidance enabled | Low indicates guidance not working |
| `train/cola/language_target_prob_mean` | Prob mass on target expert | Moderate/high | Near 0 -> routing ignores target |
| `train/cola/language_target_neglogp` | -log p(target) | Decreasing | Large values indicate mismatch |
| `train/cola/language_target_token_frac` | Tokens with valid targets | Near 1 | Near 0 -> missing language ids/map |
| `train/cola/router_grad_present_frac` | Router params with grads | Near 1.0 | Low -> router frozen/missing grads |
| `train/cola/language_prior_loss` | Aux loss to align router with targets | Decreasing | High or flat with guidance enabled |

## HydraLoRA (`train/hydralora/*`)
Head router metrics:
| Metric | What it measures | Healthy | Unhealthy |
| --- | --- | --- | --- |
| `train/hydralora/head_load_cv` | Head load balance variation | Low | High imbalance |
| `train/hydralora/head_active_frac` | Fraction of heads active | High | Low -> dead heads |
| `train/hydralora/head_router_entropy` | Head router entropy | Moderate | Collapse or uniform |
| `train/hydralora/head_load_max_frac` | Largest head load fraction | Close to `1/num_heads` | Very high -> head collapse |
| `train/hydralora/head_load_min_frac` | Smallest head load fraction | Non-zero | Near 0 -> unused heads |
| `train/hydralora/head_target_hit_rate` | Head target top-1 hit | High when guidance enabled | Low indicates guidance mismatch |
| `train/hydralora/head_target_prob_mean` | Prob mass on target head | Moderate/high | Near 0 -> guidance ignored |
| `train/hydralora/head_target_neglogp` | -log p(target head) | Decreasing | Large values |
| `train/hydralora/head_target_token_frac` | Tokens with valid head targets | Near 1 | Near 0 -> missing language ids/map |

Expert router metrics:
| Metric | What it measures | Healthy | Unhealthy |
| --- | --- | --- | --- |
| `train/hydralora/expert_load_cv` | Expert load balance variation | Low | High imbalance |
| `train/hydralora/expert_active_frac` | Fraction of experts active | High | Low -> dead experts |
| `train/hydralora/expert_router_entropy` | Expert router entropy | Moderate | Collapse or uniform |
| `train/hydralora/expert_topk_weight_mean` | Mean weight of selected experts | Around `1/top_k` | Near 1 -> single expert dominates |
| `train/hydralora/expert_load_max_frac` | Largest expert load fraction | Close to `1/num_experts` | Very high -> collapse |
| `train/hydralora/expert_load_min_frac` | Smallest expert load fraction | Non-zero | Near 0 -> unused experts |
| `train/hydralora/expert_target_hit_rate` | Expert target top-1 hit | High when guidance enabled | Low indicates mismatch |
| `train/hydralora/expert_target_prob_mean` | Prob mass on target expert | Moderate/high | Near 0 -> guidance ignored |
| `train/hydralora/expert_target_neglogp` | -log p(target expert) | Decreasing | Large values |
| `train/hydralora/expert_target_token_frac` | Tokens with valid expert targets | Near 1 | Near 0 -> missing language ids/map |

## MoE-LPR (`train/moelpr/*`, `moelpr_aux_loss`)
| Metric | What it measures | Healthy | Unhealthy |
| --- | --- | --- | --- |
| `train/moelpr/expert_load_cv` | Expert load balance variation | Low | High imbalance |
| `train/moelpr/router_entropy` | Router entropy | Moderate | Collapse or uniform |
| `train/moelpr/expert_{i}_top_lang_id` | Most routed language id for expert `i` | Stable per expert | Rapidly changing or invalid ids |
| `train/moelpr/expert_{i}_top_lang_frac` | Share of expert `i` routed to its top language | Moderate/high | Near 1 for all experts (over-specialization) |
| `train/moelpr/expert_{i}_lang_entropy` | Language diversity per expert | Lower for specialization | Very high -> no specialization |
| `train/moelpr/expert_{i}_num_langs` | Count of languages routed to expert `i` | >1 for shared experts | 0 or 1 for all experts |
| `train/moelpr/original_primary_expert` | Expert with most original-language tokens | Usually base expert (id 0) | Not base during Stage 2 |
| `train/moelpr/original_primary_frac` | Share of original tokens on primary expert | High | Low indicates leakage to new experts |
| `train/moelpr/original_to_base_pct` | Original tokens routed to base expert | High in Stage 2 | Low indicates routing drift |
| `train/moelpr/new_primary_expert` | Expert with most new-language tokens | Non-base expert | Base expert dominates new langs |
| `train/moelpr/new_primary_frac` | Share of new tokens on primary expert | Moderate | Very low -> no specialization |
| `moelpr_aux_loss` | Stage 1 load-balance or Stage 2 LPR loss | Non-zero, stable | Exploding or collapsing early |
