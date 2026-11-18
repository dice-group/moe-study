# Paper artifact

This folder contains the analysis material behind the paper. Saved notebook
outputs, tables, figures, and audit data are included.

The study covers 146 training runs (129 unique configurations) and 10,842
evaluation runs. CoLA is excluded throughout. The two broader regimes, with 96
and 200 training languages, compare eight systems each: five configurations
promoted by the ten-language search, plain LoRA at r2 and r4 as a non-routed
baseline, and the unadapted base model.

## Start with the notebooks

The notebooks give a readable walk-through in the same order as the paper.

1. [Study overview](notebooks/00_study_overview.ipynb) explains what was trained
   and evaluated.
2. [Design factors](notebooks/01_rq1_design_factors.ipynb) studies rank and
   expert count.
3. [Method comparison](notebooks/02_rq2_method_comparison.ipynb) compares the
   methods and selected settings.
4. [Broader regimes](notebooks/03_rq3_broader_regimes.ipynb) checks the selected
   systems with 96 and 200 languages.
5. [Localization](notebooks/04_rq4_localization.ipynb) shows where adaptation
   helps and where it hurts.

The [notebook guide](notebooks/README.md) explains how the notebooks relate to
the scripts and paper results.

## Find the main files

- [Audit data](output/data/) contains the CSV files behind the reported results.
- [Paper tables](output/tables/) contains generated LaTeX tables.
- [Paper figures](output/figures/) contains PDF and PNG versions of the figures.
- [Analysis scripts](scripts/) contains the code used to create the outputs.
- [Analysis settings](configs/config.yaml) records metrics, systems, and tests.
- [Software environment](env/environment.yml) records the needed packages.
- [Result workbooks](input/reports/) contains the raw result files read by the
  analysis.
- [Language data](input/languages/) contains language lists and resource labels.
- [Run metadata](metadata/) records, per training run, the optimizer steps,
  tokens, throughput, trainable and active parameters, rank, alpha, and
  language-sampling proportions. It backs the resource-accounting and
  language-mixture tables in the appendix.

The scripts are the source of truth for reported numbers. Each generated table
has a matching audit CSV. The notebooks use the same analysis functions and
present the results in a more readable form.

## Regenerate the outputs

From the repository root:

    conda env create -f paper_artifact/env/environment.yml
    conda activate moe-paper-resources
    python paper_artifact/scripts/run_all.py

This rewrites `output/tables/`, `output/data/`, and `output/figures/` from the
workbooks in `input/reports/` and the tables in `metadata/`. The notebooks can be
run from `notebooks/` with the same environment.

The launch configurations for the training runs are under
[`configs/pipeline/profiles/`](../configs/pipeline/profiles/) at the repository
root. The plain-LoRA baselines are in
[`moe_study_rank_expert_sweep_configs/moe_study_final_lora_r2_r4_96_200_v1.generated.json`](../configs/pipeline/profiles/moe_study_rank_expert_sweep_configs/moe_study_final_lora_r2_r4_96_200_v1.generated.json).

Routing matrices are an optional part of the artifact. When included, they are
under `output/routing_analysis/`.
