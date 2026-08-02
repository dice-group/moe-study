"""Broader-regime tables: RQ3 ranking and transfer, RQ4 localization.

Every table here is built from the two broader-regime workbooks, which contain
the five promoted configurations and the unadapted base model. Regimes are
never compared numerically with each other: they use different evaluation
populations, so only the ordering inside a regime is meaningful.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import common as C

MARKS = {"better": r"\gainmark", "tie": r"\tiemark", "worse": r"\lossmark"}



def _labels(cfg: dict) -> dict[str, str]:
    mapping = {cfg["rq3"]["baseline_run_id"]: cfg["rq3"]["baseline_label"]}
    for entry in cfg["rq3"]["finalists"]:
        mapping[entry["run_id"]] = entry["label"]
    return mapping


def _load_regimes(cfg: dict) -> dict[int, pd.DataFrame]:
    panels = {}
    for regime in cfg["rq3"]["regimes"]:
        data = C.load_final_panel(C.results_path(cfg, f"final_{regime}_xlsx"), cfg)
        expected = {cfg["rq3"]["baseline_run_id"], *[e["run_id"] for e in cfg["rq3"]["finalists"]]}
        if set(data["run_id"]) != expected:
            raise ValueError(f"Unexpected systems in the {regime}-language panel: {set(data['run_id'])}")
        panels[regime] = data
    return panels


def _ranking_rows(
    ranking: pd.DataFrame,
    labels: dict[str, str],
    benchmarks: list[str],
    verdicts: pd.DataFrame,
    baseline: str,
) -> list[str]:
    """One row per system: average position per benchmark, each followed by the
    verdict of that system's paired test against the base model, then overall.

    The two tables this replaces described the same systems on the same
    benchmarks, so the position and its verdict belong in the same cell pair.
    The base model is the reference of the comparison, so its verdict cells stay
    empty rather than carrying a self-comparison.
    """
    columns = benchmarks + ["macro_avg_rank"]
    best = {column: ranking[column].min() for column in columns}
    rows = []
    for run_id, row in ranking.iterrows():
        cells = []
        for column in columns:
            text = f"{row[column]:.2f}"
            if row[column] == best[column]:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
            if column in benchmarks:
                cells.append(
                    ""
                    if run_id == baseline
                    else MARKS[verdicts.loc[(run_id, column), "verdict"]]
                )
        rows.append(" & ".join([C.latex_escape(labels[run_id]), *cells]) + r" \\")
    return rows


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    benchmarks = cfg["analysis"]["benchmarks"]
    alpha = cfg["analysis"]["alpha"]
    baseline = cfg["rq3"]["baseline_run_id"]
    regimes = cfg["rq3"]["regimes"]
    labels = _labels(cfg)
    panels = _load_regimes(cfg)
    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")

    rankings = {regime: C.final_ranking(panels[regime], benchmarks) for regime in regimes}
    tests = {regime: C.final_pairwise_tests(panels[regime], benchmarks) for regime in regimes}
    verdicts = {
        regime: C.verdict_vs_baseline(tests[regime], baseline, alpha).set_index(["method", "benchmark"])
        for regime in regimes
    }
    method_order = [entry["run_id"] for entry in cfg["rq3"]["finalists"]]

    # ------------------------------------------------- ranking and vs. base
    # Ordering and verdicts come from the same panels, so they share one table:
    # each benchmark gets a right-aligned position and a left-aligned verdict
    # mark, tied together by the r@{\,}l pairs in the column spec.
    body: list[str] = []
    for index, regime in enumerate(regimes):
        if index:
            body.append(r"\addlinespace")
        body.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{regime} training languages}}}} \\")
        body.extend(
            _ranking_rows(rankings[regime], labels, benchmarks, verdicts[regime], baseline)
        )
    benchmark_header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{C.benchmark_label(benchmark)}}}" for benchmark in benchmarks
    )
    C.write_table(
        table_dir / "tab5_rq3_final_ranking.tex",
        "l" + r"r@{\,}l" * len(benchmarks) + "c",
        [rf"System & {benchmark_header} & Overall \\"],
        body,
        "Each system within its training regime, showing its average position "
        "on every benchmark (lower is better, best per column in bold) followed by the "
        rf"outcome of its paired comparison against the base model, "
        # Trimmed 2026-07-31 for the 8-page limit: the base model being the
        # unmarked reference and positions not crossing regimes are both stated
        # in the Section 5.3 body text that introduces this table.
        rf"{MARKS['better']}~better, {MARKS['tie']}~not distinguishable, "
        # The caption carries the pointers to the underlying tests so that the
        # main-text prose does not have to spend a sentence on them.
        rf"{MARKS['worse']}~worse. Underlying tests in "
        r"Tables~\ref{tab:rq3-vs-baseline-tests} and~\ref{tab:rq3-omnibus}.",
        "tab:rq3-final",
        environment="table*",
        placement="t",
        # Ten columns of two-digit numbers leave a full-width float looking
        # sparse, so the separation is opened up to fill more of the page.
        col_sep="10pt",
    )
    C.write_csv(
        pd.concat({regime: rankings[regime] for regime in regimes}, names=["regime", "run_id"]).reset_index(),
        data_dir / "tab5_rq3_final_ranking.csv",
    )
    # The Results text quotes these differences in relative terms and as win
    # rates, because a raw bits-per-byte difference is not something a reader can
    # size. Those magnitudes join the verdicts here so that every number in the
    # RQ4 paragraph comes out of one regenerated file.
    magnitudes = {
        regime: C.baseline_magnitudes(
            panels[regime], baseline, benchmarks, cfg["analysis"].get("chance_levels")
        ).set_index(["method", "benchmark"])
        for regime in regimes
    }
    vs_baseline = pd.concat(
        {
            regime: verdicts[regime].join(
                magnitudes[regime].drop(columns="median_difference")
            )
            for regime in regimes
        },
        names=["regime"],
    ).reset_index()
    C.write_csv(vs_baseline, data_dir / "tab6_rq4_vs_baseline.csv")
    C.write_csv(
        pd.concat({regime: tests[regime] for regime in regimes}, names=["regime", "row"]).reset_index(),
        data_dir / "tabA3_rq3_pairwise_tests.csv",
    )

    # ------------------------------------------------------- appendix tests
    # The Results text quotes Kendall's W for the broader regimes, so the
    # omnibus statistics need a table of their own rather than living only in
    # the released CSV.
    omnibus_rows = []
    omnibus_records = []
    for index, regime in enumerate(regimes):
        if index:
            omnibus_rows.append(r"\addlinespace")
        # The regime is a block heading rather than a column repeated once per
        # benchmark, which is the same shape as tab:rq4-resource below.
        omnibus_rows.append(
            rf"\multicolumn{{5}}{{l}}{{\textit{{{regime} training languages}}}} \\"
        )
        first = tests[regime].groupby("benchmark", sort=False).first().reset_index()
        for row in first.itertuples(index=False):
            omnibus_rows.append(
                " & ".join(
                    [
                        C.latex_escape(C.benchmark_label(row.benchmark)),
                        str(row.n_units),
                        f"{row.friedman_q:.2f}",
                        f"{row.kendalls_w:.3f}",
                        "$" + C.fmt_p(row.p_friedman) + "$" + C.sig_stars(row.p_friedman),
                    ]
                )
                + r" \\"
            )
            record = row._asdict()
            record["regime"] = regime
            omnibus_records.append(record)
    C.write_table(
        table_dir / "tabA3_rq3_omnibus.tex",
        "lrrrr",
        [r"Benchmark & $n$ & $Q$ & $W$ & $p$ \\"],
        omnibus_rows,
        "Friedman tests over the six systems in each broader regime. $Q$ is the "
        "Friedman statistic and $W$ Kendall's coefficient of concordance.",
        "tab:rq3-omnibus",
        placement="tbp",
        col_sep="3pt",
    )
    C.write_csv(pd.DataFrame(omnibus_records), data_dir / "tabA3_rq3_omnibus.csv")

    # Stacking the regimes made this 48 rows tall while using two thirds of the
    # text width, so it needed a page to itself. Putting them side by side halves
    # the height, uses the full width, and puts the comparison a reader actually
    # wants -- one configuration across the two regimes -- on a single line.
    measurements: dict[tuple, list[str]] = {}
    for regime in regimes:
        subset = tests[regime]
        subset = subset[subset["config_a"].eq(baseline) | subset["config_b"].eq(baseline)]
        for row in subset.itertuples(index=False):
            method = row.config_b if row.config_a == baseline else row.config_a
            sign = -1.0 if row.config_a == baseline else 1.0
            measurements[(regime, row.benchmark, method)] = [
                f"{sign * row.median_difference:+.4f}",
                f"{sign * row.rank_biserial:+.2f}",
                "$" + C.fmt_p(row.p_holm) + "$" + C.sig_stars(row.p_holm),
            ]

    appendix_rows = []
    for benchmark_index, benchmark in enumerate(benchmarks):
        if benchmark_index:
            appendix_rows.append(r"\addlinespace")
        appendix_rows.append(
            rf"\multicolumn{{7}}{{l}}{{\textit{{{C.latex_escape(C.benchmark_label(benchmark))}}}}} \\"
        )
        for method in method_order:
            row_cells = [C.latex_escape(labels[method])]
            for regime in regimes:
                row_cells.extend(measurements[(regime, benchmark, method)])
            appendix_rows.append(" & ".join(row_cells) + r" \\")

    regime_header = " & ".join(
        [""] + [rf"\multicolumn{{3}}{{c}}{{{regime} training languages}}" for regime in regimes]
    )
    cmidrules = "".join(
        rf"\cmidrule(lr){{{2 + 3 * index}-{4 + 3 * index}}}" for index in range(len(regimes))
    )
    C.write_table(
        table_dir / "tabA3_rq3_vs_baseline_tests.tex",
        "l" + "rrr" * len(regimes),
        [
            regime_header + r" \\",
            cmidrules,
            # "Configuration", not "Method" (2026-07-31): the rows are
            # method-plus-settings, which is what the caption already calls them.
            r"Configuration" + r" & Median diff. & Effect & $p_{\mathrm{Holm}}$" * len(regimes) + r" \\",
        ],
        appendix_rows,
        "Every configuration against the base model in the broader regimes. "
        "Median difference and rank-biserial effect are signed so that positive "
        "favours the configuration.",
        "tab:rq3-vs-baseline-tests",
        environment="table*",
        placement="tp",
        col_sep="3pt",
    )

    # ------------------------------------------------------------- transfer
    followup = C.load_primary(C.results_path(cfg, "mixed_followup_xlsx"), cfg)
    followup_ranking, _ = C.followup_ranking(followup, benchmarks)
    transfer = pd.DataFrame(
        [
            {
                "run_id": entry["run_id"],
                "label": entry["label"],
                "followup_rank_of_22": int(
                    followup_ranking.loc[entry["followup_run_id"], "overall_rank"]
                ),
            }
            for entry in cfg["rq3"]["finalists"]
        ]
    )
    transfer["followup"] = transfer["followup_rank_of_22"].rank().astype(int)
    for regime in regimes:
        adapted = rankings[regime].drop(index=baseline)
        places = {run_id: place for place, run_id in enumerate(adapted.index, 1)}
        transfer[str(regime)] = transfer["run_id"].map(places)

    stages = ["followup", *[str(regime) for regime in regimes]]
    correlations = [
        {
            "pair": f"{a} vs {b}",
            "spearman_rho": round(float(spearmanr(transfer[a], transfer[b]).statistic), 2),
        }
        for index, a in enumerate(stages)
        for b in stages[index + 1 :]
    ]
    transfer_rows = [
        " & ".join(
            [C.latex_escape(row["label"]), *[str(row[stage]) for stage in stages]]
        )
        + r" \\"
        for row in transfer.to_dict("records")
    ]
    rho_text = "; ".join(
        f"{entry['pair'].replace('followup', '10 languages')}: $\\rho={entry['spearman_rho']:.2f}$"
        for entry in correlations
    )
    C.write_table(
        table_dir / "tab7_rq3_transfer.tex",
        "lccc",
        # The "langs" abbreviation appeared nowhere else in the paper. A spanning
        # "Training languages" header would cost this float two rows on an
        # already-full page, so the caption carries the column meaning instead.
        [r"Configuration & 10 & 96 & 200 \\"],
        transfer_rows,
        "Position of each promoted configuration among the five, at the selection "
        "stage and in the two broader regimes. Columns give the number of "
        "training languages.",
        "tab:rq3-transfer",
        placement="t",
    )
    C.write_csv(transfer, data_dir / "tab7_rq3_transfer.csv")
    C.write_csv(pd.DataFrame(correlations), data_dir / "tab7_rq3_transfer_correlations.csv")

    # ------------------------------------------------------- resource groups
    # Average position hides how often a method actually beats the base model,
    # and it moves non-monotonically for some methods. The share of evaluation
    # cases won against the base model is both easier to read and, unlike the
    # position gradient, monotone for every configuration in both regimes.
    groups = cfg["rq3"]["resource_groups"]
    resource_rows: list[str] = []
    resource_frames = {}
    group_sizes = {}
    for regime in regimes:
        data = panels[regime]
        labelled = data[data["resource_availability"].isin(groups)]
        group_sizes[regime] = (
            labelled.drop_duplicates("language")
            .groupby("resource_availability")["language"]
            .size()
            .reindex(groups)
        )
        pivot = labelled.pivot_table(
            index=["resource_availability", *C.UNIT_COLS],
            columns="run_id",
            values="score",
        )
        wins = pivot.drop(columns=baseline).sub(pivot[baseline], axis=0).gt(0)
        resource_frames[regime] = wins.groupby(level="resource_availability").mean() * 100

    for index, regime in enumerate(regimes):
        if index:
            resource_rows.append(r"\addlinespace")
        sizes = group_sizes[regime]
        # Tier sizes as a compact slash-separated triple, so the group header
        # stays inside one column and the table needs no down-scaling.
        header = "/".join(str(sizes[group]) for group in groups)
        resource_rows.append(
            rf"\multicolumn{{4}}{{l}}{{\textit{{{regime} training languages}} ({header} languages)}} \\"
        )
        frame = resource_frames[regime]
        for run_id in method_order:
            cells = [f"{frame.loc[group, run_id]:.0f}" for group in groups]
            resource_rows.append(" & ".join([C.latex_escape(labels[run_id]), *cells]) + r" \\")
    C.write_table(
        table_dir / "tab8_rq4_resource.tex",
        "lccc",
        [r"Configuration & High & Medium & Low \\"],
        resource_rows,
        "Percentage of evaluation cases won against the base model, by the "
        "resource level of the language.",
        "tab:rq4-resource",
        placement="tbp",
    )
    C.write_csv(
        pd.concat(resource_frames, names=["regime", "resource_availability"]).reset_index(),
        data_dir / "tab8_rq4_resource.csv",
    )

    # The win rates above and the Figure 2 trend both live on absolute scores,
    # where FLORES gives its weakest languages the most room to move. The same
    # comparison on a relative scale is what the Section 5.4 footnote rests on,
    # so it is regenerated rather than computed by hand.
    C.write_csv(
        pd.concat(
            {
                regime: C.relative_gain_by_resource(panels[regime], baseline, "FLORES", groups)
                for regime in regimes
            },
            names=["regime", "resource_availability"],
        ).reset_index(),
        data_dir / "tab8b_rq4_flores_relative_by_resource.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
