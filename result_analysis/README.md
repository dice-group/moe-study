# MoE Study Result Analysis

Start with [`notebooks/moe_study_final_analysis.ipynb`](notebooks/moe_study_final_analysis.ipynb). It is the canonical end-to-end analysis for:

1. selecting configured LoRA ranks;
2. selecting expert, head, or projection counts;
3. ranking the mixed follow-up configurations; and
4. comparing the final 96- and 200-language trainings.

All workbooks used by this notebook are stored in [`reports/moe_study/`](reports/moe_study/). Generated cluster outputs remain local under `outputs/` and are not part of the canonical result interface.

## Statistical method

- Configurations are ranked within each paired benchmark-language evaluation unit.
- Ranks are averaged within each benchmark.
- The two FLORES directions receive equal weight.
- Belebele, FLORES, SIB200, and Taxi1500 each receive 25% weight in the macro rank.
- Friedman tests provide the per-benchmark omnibus tests.
- Paired Wilcoxon signed-rank tests use Holm correction across all tested pairs within each benchmark.
- In the final 96- and 200-language analysis, the baseline is treated as a normal sixth result.

## Additional material

- [`notebooks/validation/`](notebooks/validation/): operational and validity audits; some require local pipeline outputs.
- [`notebooks/routing_analysis/`](notebooks/routing_analysis/): routing-specific analyses.

Superseded exploratory notebooks were removed from the current tree to avoid conflicting sources of truth. They remain recoverable through Git history.
