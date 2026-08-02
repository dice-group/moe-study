# Paper reference notebooks

A readable walk-through of every analysis behind the paper, in the order the
paper tells it. These are **presentation, not implementation**: each notebook
imports `paper_resources/scripts/` and calls the same functions that write the
`.tex` tables, so a number here cannot drift from a number in the paper.

| Notebook | Question | Backs |
|---|---|---|
| `00_study_overview.ipynb` | What was trained and evaluated, and how does the funnel narrow? | Tables 1, A0, A0b |
| `01_rq1_design_factors.ipynb` | Which design factors matter — rank and expert count? | Tables 2, A1 |
| `02_rq2_method_comparison.ipynb` | Which methods lead, and do selected factors compose? | Tables 3–6, A2 |
| `03_rq3_broader_regimes.ipynb` | Does the shortlist still lead at 96 and 200 languages? | Tables 7, 8, A3 |
| `04_rq4_localization.ipynb` | Where does adaptation help, and where does it hurt? | Tables 7, 9, A3, A4, Figure 2 |

## Running them

Select the `moe-paper-resources` environment as the kernel (in VS Code, or via
`conda activate moe-paper-resources` and your own Jupyter front end — the env
ships `ipykernel` but no notebook server of its own).

The setup cell locates the repository root by walking up from the working
directory, so the notebooks run from anywhere inside the repo.

Outputs are committed. Re-execute after regenerating the pipeline:

    cd paper_resources && python scripts/run_all.py

Several cells read the audit CSVs under `paper_resources/output/data/` rather
than recomputing, so a stale pipeline shows up as stale notebook output.

## Relationship to the other notebooks

`notebooks/moe_study_final_analysis.ipynb` and
`notebooks/best_rank_expert_selection.ipynb` are the original exploratory
notebooks. They are the historical record of how the selection was actually
made, with the ad-hoc code that made it. Keep them; they are not superseded.
These `paper/` notebooks are the cleaned-up account of the same work, and the
pipeline they call is the single source of truth for anything quoted in the
paper.

## Reading discipline carried through the notebooks

- Search-stage results (RQ1, RQ2) are **exploratory**: the same observations
  selected the values being compared. Orderings there describe the search.
- The 96- and 200-language regimes differ in several factors at once, so their
  differences are an interaction between method and regime, not an effect of
  language count.
- Every configuration is trained **once**. The tests describe consistency across
  evaluation cases, not uncertainty across training runs.
- Nothing is randomized, so RQ4's relationships are associations.
