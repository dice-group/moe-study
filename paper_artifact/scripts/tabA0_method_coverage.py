"""Appendix coverage table for the uneven method grids."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import common as C


def _levels(values: pd.Series) -> str:
    levels = sorted({int(value) for value in values.dropna()})
    if not levels:
        return "--"
    if len(levels) > 2 and all(right == 2 * left for left, right in zip(levels, levels[1:])):
        return f"{levels[0]}--{levels[-1]}"
    return ", ".join(map(str, levels))


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    exclude = set(cfg["analysis"].get("exclude_kinds", []))
    rank = pd.read_excel(C.results_path(cfg, "rank_sweep_xlsx"), sheet_name="run_configs")
    component = pd.read_excel(
        C.results_path(cfg, "component_sweep_xlsx"), sheet_name="run_configs"
    )
    rank = rank[~rank["kind"].isin(exclude)]
    component = component[~component["kind"].isin(exclude)]
    rank["rank"] = rank["run_id"].map(C.parse_configured_rank)
    component["component"] = component["run_id"].map(C.parse_component_count)

    # The ``mola`` family key is the AdaMoE implementation, whose run ids carry a
    # historical ``mola``/``adamoe_1`` prefix. Rows stay alphabetical by label.
    labels = {
        "mola": "AdaMoE",
        "adamole": "AdaMoLE",
        "hmora": "HMoRA",
        "hydralora": "HydraLoRA",
        "lora": "LoRA",
        "moelora": "MOELoRA",
        "movlora": "MoV",
        "mtllora": "MTL-LoRA",
    }
    shared = {"hmora", "hydralora", "lora", "moelora", "mola", "mtllora"}
    notes = {
        "mola": "Complete shared rank panel",
        "adamole": "Higher ranks unstable; $E{=}32$ infeasible",
        "hmora": "Complete shared rank panel",
        "hydralora": "Complete shared rank panel",
        "lora": "Single adapter, so no expert count",
        "moelora": "Shared rank panel; $E{=}32$, $k{=}16$ infeasible",
        "movlora": "Rank is not defined",
        "mtllora": "Complete shared rank panel",
    }

    records = []
    rows = []
    for family in labels:
        family_rank = rank[rank["kind"].eq(family)]
        family_component = component[component["kind"].eq(family)]
        rank_levels = _levels(family_rank["rank"])
        component_levels = _levels(family_component["component"])
        record = {
            "family": family,
            "method": labels[family],
            "rank_settings": rank_levels,
            "rank_runs": len(family_rank),
            "component_settings": component_levels,
            "component_runs": len(family_component),
            "shared_rank_panel": family in shared,
            "note": notes[family],
        }
        records.append(record)
        rows.append(
            " & ".join(
                [
                    labels[family],
                    rank_levels,
                    str(len(family_rank)) if len(family_rank) else "--",
                    component_levels,
                    str(len(family_component)) if len(family_component) else "--",
                    "yes" if family in shared else "no",
                    notes[family],
                ]
            )
            + r" \\"
        )
    rows.append(r"\midrule")
    rows.append(
        " & ".join(
            ["Total", "--", str(len(rank)), "--", str(len(component)), "--", ""]
        )
        + r" \\"
    )

    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")
    C.write_table(
        table_dir / "tabA0_method_coverage.tex",
        "lccccc@{\\hskip 6pt}l",
        [
            r" & \multicolumn{2}{c}{Rank search} & \multicolumn{2}{c}{Expert-count search} & & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"Method & Settings & Runs & Settings & Runs & Shared panel & Note \\",
        ],
        rows,
        "Rank and expert-count coverage of each method in the one-factor "
        "searches, with the number of training runs behind each. A dash marks a "
        "factor the method does not define. Counts exclude CoLA "
        "(Appendix~\\ref{sec:appendix-coverage}).",
        "tab:method-coverage",
        environment="table*",
        placement="tp",
        size="footnotesize",
        col_sep="3pt",
    )
    C.write_csv(pd.DataFrame(records), data_dir / "tabA0_method_coverage.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
