"""RQ2 mixed-follow-up leaderboard, in paper-sized and complete forms."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


def _tt(run_id: str) -> str:
    return rf"\texttt{{{C.latex_escape(run_id)}}}"


def _integer_or_dash(value: float) -> str:
    return "--" if pd.isna(value) else str(int(value))


def _rows(ranking: pd.DataFrame, promoted: set[str]) -> list[str]:
    rows = []
    for run_id, row in ranking.iterrows():
        cells = [
            str(int(row["overall_rank"])),
            _tt(run_id),
            C.latex_escape(row["approach_label"]),
            _integer_or_dash(row["configured_rank"]),
            _integer_or_dash(row["component_count"]),
            _integer_or_dash(row["top_k"]),
            f"{row['Belebele']:.2f}",
            f"{row['FLORES']:.2f}",
            f"{row['SIB200']:.2f}",
            f"{row['Taxi1500']:.2f}",
            # Two decimals everywhere, so the full table and the shortlist that
            # repeats five of its rows print the same numbers.
            f"{row['macro_avg_rank']:.2f}",
        ]
        if run_id in promoted:
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        rows.append(" & ".join(cells) + r" \\")
    return rows


def _shortlist_rows(ranking: pd.DataFrame) -> list[str]:
    columns = ["Belebele", "FLORES", "SIB200", "Taxi1500", "macro_avg_rank"]
    minima = ranking[columns].min(axis=0)
    rows = []
    for run_id, row in ranking.iterrows():
        cells = [
            C.latex_escape(row["approach_label"]),
            _integer_or_dash(row["configured_rank"]),
            _integer_or_dash(row["component_count"]),
            _integer_or_dash(row["top_k"]),
        ]
        for column in columns:
            value = f"{row[column]:.2f}"
            if row[column] == minima[column]:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        rows.append(" & ".join(cells) + r" \\")
    return rows


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    data = C.load_primary(C.results_path(cfg, "mixed_followup_xlsx"), cfg)
    ranking, _ = C.followup_ranking(data, cfg["analysis"]["benchmarks"])
    labels = cfg["rq2"]["labels"]
    posthoc_family = cfg["rq2"]["posthoc_family"]
    promoted = cfg["rq2"]["promoted"]

    actual_top = ranking.head(len(posthoc_family)).index.tolist()
    if actual_top != posthoc_family:
        raise ValueError(
            f"Follow-up post-hoc family changed: expected {posthoc_family}, got {actual_top}"
        )
    actual_promoted = ranking.head(len(promoted)).index.tolist()
    if actual_promoted != promoted:
        raise ValueError(f"Promoted set changed: expected {promoted}, got {actual_promoted}")
    expected_macro = np.array([7.2847222222, 7.4791666667, 7.5])
    if not np.allclose(ranking.head(3)["macro_avg_rank"].to_numpy(), expected_macro, atol=1e-9):
        raise ValueError("Follow-up macro ranks no longer match the notebook")

    ranking["approach_label"] = ranking["approach"].map(labels)
    ranking["configured_rank"] = [C.parse_configured_rank(run_id) for run_id in ranking.index]
    ranking["component_count"] = [C.parse_component_count(run_id) for run_id in ranking.index]
    ranking["top_k"] = [C.parse_top_k(run_id) for run_id in ranking.index]

    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")
    # Header terms fixed 2026-07-31. "Family" collided with the MoE--LoRA family
    # the paper names throughout, and the first column holds run identifiers,
    # which are not configurations: fourteen follow-up runs repeat an earlier
    # configuration at the longer budget.
    full_header = [
        r"\# & Run ID & Method & Rank $r$ & Experts & Top-$k$ & Belebele & FLORES & SIB-200 & Taxi1500 & Overall \\"
    ]
    full_caption = (
        "Average position of every mixed follow-up configuration. Lower is better, "
        f"and bold marks the {C.spell_count(len(promoted))} configurations carried "
        "to the broader validation."
    )
    C.write_table(
        table_dir / "tab3_rq2_followup_ranking.tex",
        "lcccrrrrr",
        [r"Method & Rank $r$ & Experts & Top-$k$ & Belebele & FLORES & SIB-200 & Taxi1500 & Overall \\"],
        _shortlist_rows(ranking.head(len(promoted))),
        f"The {C.spell_count(len(promoted))} configurations carried to the broader validation, with "
        "their average positions among all 22 follow-up configurations. Lower is "
        "better, best per column in bold.",
        "tab:rq2-followup",
        environment="table*",
        placement="tp",
    )
    C.write_table(
        table_dir / "tabA2_rq2_full_ranking.tex",
        "rlllcccrrrr",
        full_header,
        _rows(ranking, set(promoted)),
        full_caption,
        "tab:rq2-followup-full",
        environment="table*",
        placement="tp",
        col_sep="3pt",
    )
    output = ranking.reset_index(names="run_id")
    C.write_csv(output, data_dir / "tab3_rq2_followup_ranking.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
