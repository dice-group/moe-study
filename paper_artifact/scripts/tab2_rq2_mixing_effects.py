"""RQ2 parent-to-mixed comparison requested by the paper storyline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import common as C


def _tt(run_id: str) -> str:
    return rf"\texttt{{{C.latex_escape(run_id)}}}"


def _delta(value: float, bold_positive: bool = False) -> str:
    text = f"{value:+.2f}"
    return rf"\textbf{{{text}}}" if bold_positive and value > 0 else text


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    data = C.load_primary(C.results_path(cfg, "mixed_followup_xlsx"), cfg)
    coverage = data.groupby("run_id").size()
    if len(coverage) != 22 or not coverage.eq(42).all():
        raise ValueError(f"Expected 22 complete follow-up runs with 42 rows each; got {coverage.to_dict()}")

    benchmarks = cfg["analysis"]["benchmarks"]
    ranking, _ = C.followup_ranking(data, benchmarks)
    expected_ids = set(ranking.index)
    labels = cfg["rq2"]["labels"]

    detailed_records = []
    main_records = []
    body = []
    for group in cfg["rq2"]["parent_groups"]:
        parent = group["parent"]
        children = group["children"]
        missing = ({parent} | set(children)) - expected_ids
        if missing:
            raise ValueError(f"Parent mapping references missing follow-up runs: {sorted(missing)}")

        for child in children:
            record = {
                "architecture": labels[group["architecture"]],
                "configured_rank": group["configured_rank"],
                "parent": parent,
                "mixed_child": child,
                "parent_macro_rank": ranking.loc[parent, "macro_avg_rank"],
                "mixed_macro_rank": ranking.loc[child, "macro_avg_rank"],
            }
            for benchmark in benchmarks:
                record[f"delta_{benchmark}"] = ranking.loc[parent, benchmark] - ranking.loc[child, benchmark]
            record["delta_macro"] = ranking.loc[parent, "macro_avg_rank"] - ranking.loc[child, "macro_avg_rank"]
            detailed_records.append(record)

        group_records = [record for record in detailed_records if record["parent"] == parent]
        best = min(group_records, key=lambda record: record["mixed_macro_rank"])
        main_records.append(best)
        cells = [
            best["architecture"],
            str(group["configured_rank"]),
            _tt(parent),
            _tt(best["mixed_child"]),
            *[_delta(best[f"delta_{benchmark}"]) for benchmark in benchmarks],
            _delta(best["delta_macro"], bold_positive=True),
        ]
        body.append(" & ".join(cells) + r" \\")

    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")
    C.write_table(
        table_dir / "tab2_rq2_mixing_effects.tex",
        "lcllrrrrr",
        [
            # "Method" rather than "Architecture" (2026-07-31): the rows are
            # method names, and "architecture" is the internal config key, not
            # a term the paper uses.
            r"Method & Rank $r$ & Preserved parent & Best mixed descendant & Belebele & FLORES & SIB-200 & Taxi1500 & Overall \\"
        ],
        body,
        "Effect of combining the selected rank and expert-count values. Cells are "
        "the parent's average position minus the descendant's, so positive values "
        "favour mixing and bold marks a positive overall effect.",
        "tab:rq2-mixing",
        environment="table*",
        placement="tp",
        # The run-id columns make this the widest of the RQ2 tables. Tightening
        # the column separation buys back enough room for footnotesize, which is
        # a readable step up from the scriptsize this used to need.
        size="footnotesize",
        col_sep="3pt",
    )
    C.write_csv(pd.DataFrame(main_records), data_dir / "tab2_rq2_mixing_effects.csv")
    C.write_csv(pd.DataFrame(detailed_records), data_dir / "tab2_rq2_mixing_effects_all_descendants.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
