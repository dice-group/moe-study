# MoE Study Eval Results

Generated: 2026-01-28 14:03 UTC

## Source projects
- W&B: `dice-nlp/moe-study-comparison`
- W&B: `dice-nlp/moe-study-comparison_eval`

## Eval loss (latest per tier/approach/model)
### 10_langs
| Approach | 1B | 3B | 8B |
|---|---|---|---|
| adamole | — | — | — |
| cola | — | — | — |
| hydralora | — | — | — |
| moelpr | — | — | — |
| mola | — | — | — |

### 96_langs
| Approach | 1B | 3B | 8B |
|---|---|---|---|
| adamole | — | — | — |
| cola | — | — | — |
| hydralora | — | — | — |
| moelpr | — | — | — |
| mola | — | — | — |

### 200_langs
| Approach | 1B | 3B | 8B |
|---|---|---|---|
| adamole | — | — | — |
| cola | — | — | — |
| hydralora | — | — | — |
| moelpr | — | — | — |
| mola | — | — | — |


## Aggregated eval (Llama-3.1-8B)
FLORES uses `byte_perplexity` (word perplexity can be unstable across scripts).
### high resource (10_langs)
| Approach | FLORES avg byte_ppl | n_flores | Belebele avg acc_norm | n_belebele |
|---|---|---|---|---|
| adamole | 2.5547 | 18 | 0.4980 | 9 |
| cola | 2.5662 | 18 | 0.4552 | 9 |
| hydralora | 2.5556 | 18 | 0.5099 | 9 |
| moelpr | — | 0 | — | 0 |
| mola | 2.5274 | 18 | 0.5157 | 9 |

### mid resource (96_langs)
| Approach | FLORES avg byte_ppl | n_flores | Belebele avg acc_norm | n_belebele |
|---|---|---|---|---|
| adamole | 3.1314 | 92 | 0.4398 | 34 |
| cola | 3.1706 | 92 | 0.4238 | 34 |
| hydralora | 3.1195 | 92 | 0.4489 | 34 |
| moelpr | — | 0 | — | 0 |
| mola | 3.0987 | 92 | 0.4453 | 34 |

### low resource (200_langs)
| Approach | FLORES avg byte_ppl | n_flores | Belebele avg acc_norm | n_belebele |
|---|---|---|---|---|
| adamole | 2.9833 | 156 | 0.4406 | 57 |
| cola | 3.0167 | 156 | 0.4101 | 57 |
| hydralora | 2.9833 | 156 | 0.4332 | 57 |
| moelpr | — | 0 | — | 0 |
| mola | 2.9656 | 156 | 0.4413 | 57 |


### Best per tier
- high: FLORES best = mola (2.5274)
- high: Belebele best = mola (0.5157)
- mid: FLORES best = mola (3.0987)
- mid: Belebele best = hydralora (0.4489)
- low: FLORES best = mola (2.9656)
- low: Belebele best = mola (0.4413)

## Plots
### Eval loss (per model size)
![](plots/eval_loss_1B.png)
![](plots/eval_loss_3B.png)
![](plots/eval_loss_8B.png)

### FLORES byte perplexity
![](plots/flores_byte_ppl.png)

### Belebele acc_norm
![](plots/belebele_acc_norm.png)

## Notes
- Latest run per group is chosen by `_timestamp` in W&B summary (fallback: updated_at).
- Resource tiers map: 10_langs=high, 96_langs=mid, 200_langs=low.
- Belebele aggregation uses `acc_norm` over the task list in `configs/lm_eval_tasks_{tier}_langs.txt`.
