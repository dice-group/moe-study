"""RQ2 comparison on the shared six-family rank panel."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    data = C.load_primary(C.results_path(cfg, "rank_sweep_xlsx"), cfg)
    data["configured_rank"] = data["run_id"].map(C.parse_configured_rank)
    levels = cfg["analysis"]["rank_levels"]
    data = data[data["configured_rank"].isin(levels)].copy()

    complete_families = sorted(
        set.intersection(
            *[
                set(data.loc[data["configured_rank"].eq(level), "kind"])
                for level in levels
            ]
        )
    )
    expected = ["hmora", "hydralora", "lora", "moelora", "mola", "mtllora"]
    if complete_families != expected:
        raise ValueError(
            f"Shared rank-panel families changed: expected {expected}, got {complete_families}"
        )
    data = data[data["kind"].isin(complete_families)].copy()

    pivot = data.pivot(
        index=C.UNIT_COLS + ["configured_rank"], columns="kind", values="score"
    )
    if pivot.isna().any().any():
        raise ValueError("Incomplete shared family panel")
    unit_ranks = pivot.rank(axis=1, ascending=False, method="average")
    family_ranks = unit_ranks.groupby(level="score_family").mean()
    benchmark_ranks = C.combine_flores(
        family_ranks, cfg["analysis"]["benchmarks"]
    )
    summary = benchmark_ranks.T
    summary["macro_avg_rank"] = benchmark_ranks.mean()
    summary = summary.sort_values("macro_avg_rank")
    expected_order = ["moelora", "mola", "hydralora", "mtllora", "lora", "hmora"]
    expected_macro = np.array(
        [2.68287037, 2.71990741, 3.25347222, 3.66898148, 3.73032407, 4.94444444]
    )
    if summary.index.tolist() != expected_order or not np.allclose(
        summary["macro_avg_rank"].to_numpy(), expected_macro, atol=1e-8
    ):
        raise ValueError("Shared-panel family ranking no longer matches the audited result")

    labels = {
        "hmora": "HMoRA",
        "hydralora": "HydraLoRA",
        "lora": "LoRA",
        "moelora": "MOELoRA",
        "mola": "AdaMoE",
        "mtllora": "MTL-LoRA",
    }
    body = []
    minima = summary.min(axis=0)
    for family, row in summary.iterrows():
        cells = [labels[family]]
        for column in [*cfg["analysis"]["benchmarks"], "macro_avg_rank"]:
            value = f"{row[column]:.2f}"
            if row[column] == minima[column]:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        body.append(" & ".join(cells) + r" \\")

    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")
    C.write_table(
        table_dir / "tab2_rq2_family_comparison.tex",
        "lrrrrr",
        # Full benchmark names make this tabular 240pt wide against a 219pt
        # column, so the single-column table abbreviates them. The caption and
        # the surrounding text name all four benchmarks in full.
        [r"Method & Bele. & FLO. & SIB & Taxi & Overall \\"],
        body,
        "Average position of the six methods with complete coverage of ranks "
        "2--64 on Belebele, FLORES, SIB-200 and Taxi1500. Lower is better, "
        # The caption carries the pointer to the appendix coverage matrix so
        # that the main-text prose does not have to spend a sentence on it.
        r"best per column in bold. Coverage matrix in Table~\ref{tab:method-coverage}.",
        "tab:rq2-family",
        environment="table",
        placement="t",
        col_sep="5pt",
    )
    output = summary.reset_index(names="family")
    output.insert(1, "method", output["family"].map(labels))
    C.write_csv(output, data_dir / "tab2_rq2_family_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
