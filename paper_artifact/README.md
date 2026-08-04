# Paper resources: generated tables and figures

This directory regenerates every table and figure in the paper from the
workbooks under `reports/moe_study/`. All counts exclude CoLA, whose failed and
high-loss runs are not representative and which the paper drops entirely.

## Inputs

- `moe_study_rank_sweep.xlsx`
- `moe_study_expert_sweep.xlsx`
- `moe_study_mixed_followup.xlsx`
- `moe_study_final_96_languages_2026-07-25.xlsx`
- `moe_study_final_200_languages_2026-07-25.xlsx`

The scripts keep the four primary metrics used by
`notebooks/moe_study_final_analysis.ipynb`, reverse lower-is-better metrics, and
rank systems inside paired evaluation cases. FLORES directions are ranked
separately and then given equal weight; the four benchmark families receive
equal weight in macro rankings. No cross-workbook score normalization is used.

The sweep loader drops the base model, which is not a training run there. The
broader-regime loader keeps it, because in those regimes the unadapted model is
one of the six compared systems. It also replaces the workbooks' legacy
resource lookup with the finalized FineWeb2 subset labels and fails if any
evaluated language has no canonical tier; English is the sole explicit
override because the pipeline supplies it separately from the FineWeb2 table.

## Two kinds of run

- **Training run** — one configuration trained once. 142 in total, 125 unique
  (fourteen follow-up configurations repeat an earlier setting at the longer
  budget, and the two one-axis sweeps share three settings). Configurations are
  compared by canonical key — method, per-expert rank, expert count, top-k —
  because the stages spell the same configuration differently
  (`moelora_r2_e16_k8` in the expert sweep is `moelora_tr32_e16_k8` in the
  follow-up, and AdaMoE runs appear as `adamoe_1_*` in the expert sweep and as
  `mola_*` elsewhere). Intersecting raw run ids
  finds only eight of these repeats.
- **Evaluation run** — one model measured on one benchmark task, where a task is
  a benchmark--language pair and the two FLORES directions count separately.
  9,528 in total. Repeated settings are counted once per training run, not once
  per configuration: each was trained again at the longer budget and evaluated
  on its own.

## Generated tables and figures

| File | Role |
|---|---|
| tab0_study_funnel.tex | Stage coverage, training runs, evaluation runs |
| tabA0_method_coverage.tex | Uneven method-grid coverage |
| tab1_rq1_design_factors.tex | Compact rank/expert-count selection summary |
| tabA1_rq1_full_ranks.tex | Full RQ1 average-rank matrices |
| tabA1_rq1_omnibus.tex | RQ1 Friedman tests and Kendall's W |
| tab2_rq2_family_comparison.tex | Six-method comparison on the shared rank panel |
| tab2_rq2_mixing_effects.tex | Preserved-parent versus mixed-descendant analysis |
| tab3_rq2_followup_ranking.tex | Paper-sized follow-up leaderboard |
| tabA2_rq2_full_ranking.tex | Complete 22-configuration leaderboard |
| tab4_rq2_statistical_tests.tex | Follow-up omnibus and top-three post-hoc tests |
| tab5_rq3_final_ranking.tex | **RQ3/RQ4 main table**: six systems in two regimes, each position paired with its better/tie/worse verdict against the base model |
| tab7_rq3_transfer.tex | Shortlist position at each of the three stages |
| tab8_rq4_resource.tex | Win rate against the base model by language-resource level |
| tabA3_rq3_vs_baseline_tests.tex | Full broader-regime tests against the base model |
| fig1_rq3_transfer.pdf | **Figure 1**: how the shortlist reorders across stages |

Every LaTeX table under `output/tables/` has an audit CSV under `output/data/`.

Two CSVs back numbers that the Results text quotes without a table of their own,
because a raw bits-per-byte difference is not a quantity a reader can size:

| File | Role |
|---|---|
| tab6_rq4_vs_baseline.csv | Verdicts against the base model, plus the base median, the median relative gain, the win rate, and the difference as a share of the margin above chance |
| tab8b_rq4_flores_relative_by_resource.csv | Median relative FLORES gain per resource tier, which separates the Section 5.4 trend from the dynamic range of the metric |

Chance levels for the headroom column are declared per benchmark in
`configs/config.yaml`, and only where the floor is unambiguous.
The figure is written straight to `paper_latex/data/figures/`. Generated files
should not be hand-edited; tables must be copied into
`paper_latex/data/tables/` after regeneration, which `build.sh` does not do.

## Statistical protocol

RQ1 and RQ2 use per-benchmark Friedman tests with Kendall's W, followed by
two-sided Wilcoxon signed-rank tests, each stated family Holm-corrected. These
are exploratory: the same observations selected the values being compared.

RQ3 and RQ4 run one Friedman test per benchmark over all six systems, then
Wilcoxon tests on all fifteen system pairs, Holm-corrected inside each regime
and benchmark. Testing every pair keeps the correction family independent of
which systems rank highest.

## Reproduction

    cd paper_resources
    conda env create -f env/environment.yml
    conda activate moe-paper-resources
    python scripts/run_all.py

The pipeline asserts the per-stage run counts, the cross-stage unique-identifier
accounting, complete follow-up coverage, the executed expert-count ordering
(32, 16, 4, 8, 2), and the exact system set in each broader-regime panel.

## Directory layout

    paper_resources/
    ├── configs/config.yaml
    ├── env/environment.yml
    ├── scripts/
    │   ├── common.py
    │   ├── run_all.py
    │   ├── fig1_rq3_transfer.py
    │   ├── tab0_study_funnel.py
    │   ├── tab1_rq1_design_factors.py
    │   ├── tab2_rq2_family_comparison.py
    │   ├── tab2_rq2_mixing_effects.py
    │   ├── tab3_rq2_followup_ranking.py
    │   ├── tab4_rq2_statistical_tests.py
    │   ├── tab5_rq3_final.py
    │   └── tabA0_method_coverage.py
    └── output/
        ├── data/
        └── tables/
