"""RQ1 tables: configured-rank and expert-count effects."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import common as C


def prepare_rank_panel(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    data = C.load_primary(C.results_path(cfg, "rank_sweep_xlsx"), cfg)
    data["level"] = data["run_id"].map(C.parse_configured_rank)
    levels = cfg["analysis"]["rank_levels"]
    data = data[data["level"].isin(levels)].copy()
    common = sorted(set.intersection(*[set(data.loc[data.level.eq(level), "kind"]) for level in levels]))
    data = data[data["kind"].isin(common)].copy()
    return data, common


def prepare_component_panel(cfg: dict) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    data = C.load_primary(C.results_path(cfg, "component_sweep_xlsx"), cfg)
    data["level"] = data["run_id"].map(C.parse_component_count)
    levels = cfg["analysis"]["component_levels"]
    data = data[data["level"].isin(levels)].copy()
    common = sorted(set.intersection(*[set(data.loc[data.level.eq(level), "kind"]) for level in levels]))
    data = data[data["kind"].isin(common)].copy()
    # Averaging over configurations drops run_id, which the parameter summary
    # still needs, so the run-level frame is returned alongside.
    runs = data[["run_id", "kind", "level"]].drop_duplicates()
    data = (
        data.groupby(["benchmark"] + C.UNIT_COLS + ["kind", "level"], as_index=False)
        .agg(score=("score", "mean"), configurations_averaged=("run_id", "nunique"))
    )
    return data, common, runs


def _best_two(series: pd.Series, lower_is_better: bool) -> tuple[int, int]:
    ordered = series.sort_values(ascending=lower_is_better)
    return int(ordered.index[0]), int(ordered.index[1])


def _winner_list(frame: pd.DataFrame, lower_is_better: bool) -> str:
    winners = []
    for benchmark, row in frame.iterrows():
        value = row.idxmin() if lower_is_better else row.idxmax()
        winners.append(f"{benchmark}: {int(value)}")
    return "; ".join(winners)


def _full_rank_rows(title: str, by_benchmark: pd.DataFrame, macro: pd.Series) -> list[str]:
    """One block of the RQ1 panel table.

    Bold marks the best value in each column, matching the convention used by
    every other ranking table in the paper. Bolding the whole winning row
    instead would claim per-benchmark wins the numbers do not support: E=32
    has the best overall average but is only fourth of five on SIB-200.
    """
    rows = [rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}}}} \\"]
    best_per_benchmark = {b: by_benchmark.loc[b].idxmin() for b in by_benchmark.index}
    best_macro = macro.idxmin()
    for level in by_benchmark.columns:
        cells = [str(int(level))]
        for benchmark in by_benchmark.index:
            value = f"{by_benchmark.loc[benchmark, level]:.2f}"
            cells.append(rf"\textbf{{{value}}}" if best_per_benchmark[benchmark] == level else value)
        macro_value = f"{macro.loc[level]:.2f}"
        cells.append(rf"\textbf{{{macro_value}}}" if best_macro == level else macro_value)
        rows.append(" & ".join(cells) + r" \\")
    return rows


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    benchmarks = cfg["analysis"]["benchmarks"]
    rank_data, rank_kinds = prepare_rank_panel(cfg)
    component_data, component_kinds, component_runs = prepare_component_panel(cfg)
    # Adapter sizes back the parameter contrast stated in the RQ1 prose. They
    # stay in the audit CSV: per-level counts are a wide spread rather than one
    # number, which the paper table has no room to qualify.
    rank_params = C.parameter_summary(
        rank_data, C.load_run_params(C.results_path(cfg, "rank_sweep_xlsx")), "level"
    )
    component_params = C.parameter_summary(
        component_runs, C.load_run_params(C.results_path(cfg, "component_sweep_xlsx")), "level"
    )

    rank_levels = cfg["analysis"]["rank_levels"]
    component_levels = cfg["analysis"]["component_levels"]
    rank_by_bench, rank_macro, rank_tests = C.average_rank_summary(
        rank_data, "level", rank_levels, benchmarks
    )
    component_by_bench, component_macro, component_tests = C.average_rank_summary(
        component_data, "level", component_levels, benchmarks
    )
    # The component workbook now contains every run used by the executed
    # selection, so the panel reproduces the executed ordering directly.
    expected_component_order = [32, 16, 4, 8, 2]
    if component_macro.index.tolist() != expected_component_order:
        raise ValueError(
            "Expert-count ordering changed: "
            f"expected {expected_component_order}, got {component_macro.index.tolist()}"
        )
    rank_robust_by_bench, rank_robust_macro = C.architecture_collapsed_normalized_summary(
        rank_data, "level", rank_levels, benchmarks
    )
    component_robust_by_bench, component_robust_macro = C.architecture_collapsed_normalized_summary(
        component_data, "level", component_levels, benchmarks
    )

    summary_records = []
    analyses = [
        ("Rank", "Architecture blocks", rank_macro, rank_by_bench, True, cfg["analysis"]["executed_rank_choices"]),
        ("Rank", "Architecture-collapsed", rank_robust_macro, rank_robust_by_bench, False, cfg["analysis"]["executed_rank_choices"]),
        ("Expert count", "Architecture blocks", component_macro, component_by_bench, True, cfg["analysis"]["executed_component_choices"]),
        ("Expert count", "Architecture-collapsed", component_robust_macro, component_robust_by_bench, False, cfg["analysis"]["executed_component_choices"]),
    ]
    for factor, analysis, macro, by_benchmark, lower, executed in analyses:
        best, runner = _best_two(macro, lower)
        winners = _winner_list(by_benchmark, lower)
        summary_records.append(
            {
                "factor": factor,
                "analysis": analysis,
                "best": best,
                "runner_up": runner,
                "benchmark_winners": winners,
                "executed_followup_choices": ",".join(map(str, executed)),
            }
        )

    # The architecture-collapsed view stays in the audit CSV as an internal
    # robustness check and is deliberately not shown in the paper table.
    compact_rows = [
        " & ".join(
            [
                r"Rank $r$",
                f"{int(rank_macro.index[0])} (next: {int(rank_macro.index[1])})",
                ", ".join(map(str, cfg["analysis"]["executed_rank_choices"])),
            ]
        )
        + r" \\",
        " & ".join(
            [
                "Expert count",
                f"{int(component_macro.index[0])} (next: {int(component_macro.index[1])})",
                ", ".join(map(str, cfg["analysis"]["executed_component_choices"])),
            ]
        )
        + r" \\",
    ]

    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")
    C.write_table(
        table_dir / "tab1_rq1_design_factors.tex",
        "lcc",
        [r"Factor & Selection analysis & Follow-up values \\"],
        compact_rows,
        "Design-factor values carried into the mixed follow-up.",
        "tab:rq1-design",
        placement="tbp",
    )

    full_rows = _full_rank_rows(r"Rank $r$", rank_by_bench, rank_macro)
    full_rows.append(r"\addlinespace")
    full_rows.extend(_full_rank_rows("Expert count", component_by_bench, component_macro))
    C.write_table(
        table_dir / "tabA1_rq1_full_ranks.tex",
        "lccccc",
        [r"Setting & Belebele & FLORES & SIB-200 & Taxi1500 & Overall \\"],
        full_rows,
        "Average position of each rank and expert count on the balanced panels; "
        "lower is better, best per column within each factor in bold.",
        "tab:rq1-full-ranks",
        placement="tbp",
        col_sep="3pt",
    )

    test_rows = []
    test_records = []
    factors = [(r"Rank $r$", rank_tests), ("Expert count", component_tests)]
    for index, (factor, tests) in enumerate(factors):
        if index:
            test_rows.append(r"\addlinespace")
        # The factor reads as a block heading rather than a column repeated once
        # per benchmark, matching the layout of tab:rq1-full-ranks above.
        test_rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{factor}}}}} \\")
        for row in tests.itertuples(index=False):
            p_text = "$" + C.fmt_p(row.p_holm) + "$" + C.sig_stars(row.p_holm)
            test_rows.append(
                f"{C.benchmark_label(row.benchmark)} & {row.n_blocks} & {row.friedman_q:.2f} & {row.kendalls_w:.3f} & {p_text} " + r"\\"
            )
            record = row._asdict()
            record["factor"] = factor
            test_records.append(record)
    C.write_table(
        table_dir / "tabA1_rq1_omnibus.tex",
        "lrrrr",
        # The statistic names are the widest thing in the table and the columns
        # below them are half their width, so they move into the caption and the
        # table fits a single text column.
        [r"Benchmark & $n$ & $Q$ & $W$ & $p_{\mathrm{Holm}}$ \\"],
        test_rows,
        "Friedman tests for the balanced rank and expert-count panels. $Q$ is the "
        "Friedman statistic and $W$ Kendall's coefficient of concordance.",
        "tab:rq1-omnibus",
        placement="tbp",
        col_sep="3pt",
    )

    C.write_csv(pd.DataFrame(summary_records), data_dir / "tab1_rq1_design_factors.csv")
    full_csv = pd.concat(
        {
            "configured_rank": rank_by_bench.T.assign(macro=rank_macro).join(rank_params),
            "component_count": component_by_bench.T.assign(macro=component_macro).join(component_params),
        },
        names=["factor", "value"],
    ).reset_index()
    C.write_csv(full_csv, data_dir / "tabA1_rq1_full_ranks.csv")
    C.write_csv(pd.DataFrame(test_records), data_dir / "tabA1_rq1_omnibus.csv")

    audit = pd.DataFrame(
        [
            {"factor": "configured_rank", "common_architectures": ",".join(rank_kinds), "count": len(rank_kinds)},
            {"factor": "component_count", "common_architectures": ",".join(component_kinds), "count": len(component_kinds)},
        ]
    )
    C.write_csv(audit, data_dir / "rq1_panel_audit.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
