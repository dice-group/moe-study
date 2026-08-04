# Paper artifact

This folder contains the analysis material behind the paper. Saved notebook
outputs, tables, figures, and audit data are included.

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

The scripts are the source of truth for reported numbers. Each generated table
has a matching audit CSV. The notebooks use the same analysis functions and
present the results in a more readable form.

Routing matrices are an optional part of the artifact. When included, they are
under `output/routing_analysis/`.
