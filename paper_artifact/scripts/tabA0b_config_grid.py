"""Appendix table listing the configurations actually trained per method.

Table~\\ref{tab:method-coverage} gives the endpoints of each grid; this table
gives the grid itself, so that the per-stage run counts in
Table~\\ref{tab:study-funnel} can be reconstructed method by method.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import common as C


def _ints(values) -> list[int]:
    return sorted({int(value) for value in values if not pd.isna(value)})


def _join(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def _fixed_context(frame: pd.DataFrame) -> str:
    """Describe the expert count and top-k held fixed during the rank search."""
    experts = _ints(frame["component"])
    top_k = _ints(frame["top_k"])
    if not experts:
        return ""
    parts = [rf"$E{{=}}{experts[0]}$"] if len(experts) == 1 else []
    if len(top_k) == 1:
        parts.append(rf"$k{{=}}{top_k[0]}$")
    return " at " + ", ".join(parts) if parts else ""


def _rank_cell(frame: pd.DataFrame) -> str:
    ranks = _ints(frame["rank"])
    if not ranks:
        # MoV defines no rank, so its single row is described by its expert count
        # alone; a method absent from the search has no row at all.
        context = _fixed_context(frame)
        return f"no rank{context}" if len(frame) else "--"
    return rf"$r \in \{{{_join(ranks)}\}}$" + _fixed_context(frame)


def _component_cell(frame: pd.DataFrame) -> str:
    """Group the expert-count search by expert count, listing the top-k values."""
    if frame.empty:
        return "--"
    groups = []
    for experts in _ints(frame["component"]):
        block = frame[frame["component"].eq(experts)]
        top_k = _ints(block["top_k"])
        if top_k:
            groups.append(rf"${experts}$:~{_join(top_k)}")
        else:
            groups.append(rf"${experts}$")
    separator = "; " if any(":" in group for group in groups) else ", "
    return separator.join(groups)


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    exclude = set(cfg["analysis"].get("exclude_kinds", []))
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

    frames = {}
    for stage, key in [("rank", "rank_sweep_xlsx"), ("component", "component_sweep_xlsx")]:
        frame = pd.read_excel(C.results_path(cfg, key), sheet_name="run_configs")
        frame = frame[~frame["kind"].isin(exclude)].copy()
        frame["rank"] = frame["run_id"].map(C.parse_configured_rank)
        frame["component"] = frame["run_id"].map(C.parse_component_count)
        frame["top_k"] = frame["run_id"].map(C.parse_top_k)
        frames[stage] = frame

    records = []
    rows = []
    for family, label in labels.items():
        rank_frame = frames["rank"][frames["rank"]["kind"].eq(family)]
        component_frame = frames["component"][frames["component"]["kind"].eq(family)]
        rank_cell = _rank_cell(rank_frame)
        component_cell = _component_cell(component_frame)
        records.append(
            {
                "family": family,
                "method": label,
                "rank_configurations": rank_cell,
                "rank_runs": len(rank_frame),
                "component_configurations": component_cell,
                "component_runs": len(component_frame),
            }
        )
        rows.append(
            " & ".join([label, rank_cell, str(len(rank_frame) or "--"),
                        component_cell, str(len(component_frame) or "--")])
            + r" \\"
        )

    totals = (len(frames["rank"]), len(frames["component"]))
    expected = cfg["study_funnel"]["expected_runs"]
    if totals != (expected["rank"], expected["component"]):
        raise ValueError(f"Grid enumeration misses runs: {totals} vs {dict(expected)}")
    rows.append(r"\midrule")
    rows.append(" & ".join(["Total", "", str(totals[0]), "", str(totals[1])]) + r" \\")

    C.write_table(
        C.output_dir(cfg, "tables") / "tabA0b_config_grid.tex",
        "llcll",
        [r"Method & Rank search & Runs & Expert-count search ($E$:~top-$k$) & Runs \\"],
        rows,
        "Every configuration trained in the two one-factor searches. $r$ is the "
        "rank of one expert, $E$ the number of experts, and $k$ the number of "
        "experts routed to per token; methods without a router define no $k$. "
        "The rank search holds $E$ and $k$ fixed, and the expert-count search "
        "holds $r{=}2$. Counts exclude CoLA "
        "(Appendix~\\ref{sec:appendix-coverage}).",
        "tab:config-grid",
        environment="table*",
        placement="tp",
        size="footnotesize",
        resize_to=r"\textwidth",
        col_sep="4pt",
    )
    C.write_csv(pd.DataFrame(records), C.output_dir(cfg, "data") / "tabA0b_config_grid.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
