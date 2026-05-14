# Result Analysis Notebooks

This folder contains the notebook-based analysis for the rank and expert sweep reports.

## `expert_sweep_analysis_significane.ipynb`

Main significance analysis for the expert sweep. It asnwers:
- which expert counts perform best overall
- how the result changes with and without CoLA
- which concrete approaches perform best inside the selected expert counts
- whether the differences are significant using Friedman, Wilcoxon signed-rank, and Holm correction

This notebook was checked in detail and should be treated as the main source for expert-count conclusions.

## `rank_sweep_analysis_signifance.ipynb`

Main significance analysis for the rank sweep. It answers:

- which configured rank values perform best overall
- which concrete approaches perform best inside the selected rank values
- whether the differences are significant using Friedman, Wilcoxon signed-rank, and Holm correction

This notebook was checked in detail and should be treated as the main source for configured-rank conclusions.

## `visualizations_rank_expert_sweep.ipynb`

Compact visualization notebook for the rank and expert sweep findings.
Use this notebook to quickly inspect:
- the final top-rank and top-expert findings
- mean-score vs Friedman-rank differences
- per-benchmark trend plots for expert count and configured rank
- exact scores, normalized scores, and Friedman-style average ranks

This notebook is mainly for visualization and communication. 
It was less deeply audited than the two significance notebooks and was mostly AI-assisted, so final statistical claims should still be taken from the two dedicated significance notebooks.
