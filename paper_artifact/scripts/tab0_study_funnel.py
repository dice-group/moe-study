"""Experimental-coverage table for the staged study.

Reports training runs and evaluation runs separately: a training run is one
configuration trained once, an evaluation run is one model measured on one
benchmark task (FLORES directions count separately).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import common as C


def _tasks_per_model(path: Path, cfg: dict) -> int:
    data = C.load_final_panel(path, cfg)
    per_model = data.groupby("run_id")["task"].nunique()
    if per_model.nunique() != 1:
        raise ValueError(f"Uneven task coverage in {path.name}: {per_model.to_dict()}")
    return int(per_model.iloc[0])


def main(config_path: Path = C.CONFIG_PATH) -> None:
    cfg = C.load_config(config_path)
    exclude = set(cfg["analysis"].get("exclude_kinds", []))
    funnel_cfg = cfg["study_funnel"]
    expected = funnel_cfg["expected_runs"]

    rank = pd.read_excel(C.results_path(cfg, "rank_sweep_xlsx"), sheet_name="run_configs")
    component = pd.read_excel(C.results_path(cfg, "component_sweep_xlsx"), sheet_name="run_configs")
    followup = pd.read_excel(C.results_path(cfg, "mixed_followup_xlsx"), sheet_name="run_configs")

    # Excluded kinds are dropped from every stage before counting, so all
    # reported totals describe the method pool that appears in the paper.
    rank = rank[~rank["kind"].isin(exclude)]
    component = component[~component["kind"].isin(exclude)]
    followup = followup[~followup["kind"].isin(exclude)]

    observed = {"rank": len(rank), "component": len(component), "followup": len(followup)}
    if observed != dict(expected):
        raise ValueError(f"Unexpected run counts: {observed}, expected {dict(expected)}")
    if set(rank.run_id) & set(component.run_id):
        raise ValueError("Rank and component sweeps unexpectedly overlap")

    # Configurations are compared by canonical key rather than by run id, because
    # the stages spell the same configuration differently (see
    # C.canonical_config_key). Every row here is a training run that was trained
    # and evaluated on its own; the keys only say which runs share a setting.
    for frame in (rank, component, followup):
        frame["config_key"] = [
            C.canonical_config_key(run_id, kind)
            for run_id, kind in zip(frame["run_id"], frame["kind"])
        ]
    sweep_keys = set(rank.config_key) | set(component.config_key)
    unique_configs = sweep_keys | set(followup.config_key)
    repeated = sweep_keys & set(followup.config_key)
    if (
        len(unique_configs) != funnel_cfg["expected_unique_configs"]
        or len(repeated) != funnel_cfg["expected_repeated_followup_configs"]
    ):
        raise ValueError(
            "Unexpected cross-stage accounting: "
            f"unique_configs={len(unique_configs)}, repeated={len(repeated)}"
        )

    regimes = cfg["rq3"]["regimes"]
    finalists = cfg["rq3"]["finalists"]
    final_runs = len(finalists) * len(regimes)
    final_hours = funnel_cfg["final_budget_hours"]

    sweep_tasks = _tasks_per_model(C.results_path(cfg, "mixed_followup_xlsx"), cfg)
    final_tasks = {
        regime: _tasks_per_model(C.results_path(cfg, f"final_{regime}_xlsx"), cfg)
        for regime in regimes
    }

    training_runs = sum(observed.values()) + final_runs
    unique_runs = len(unique_configs) + final_runs
    # Every training run is evaluated separately, including runs that repeat an
    # earlier setting at a longer budget: those are distinct models with their
    # own scores. One base-model evaluation set is added per distinct evaluation
    # population.
    evaluation_runs = (
        (sum(observed.values()) + 1) * sweep_tasks
        + sum((len(finalists) + 1) * final_tasks[regime] for regime in regimes)
    )

    rows = [
        {
            "stage": "One-axis rank sweep",
            "languages": funnel_cfg["sweep_languages"],
            "hours": funnel_cfg["rank_budget_hours"],
            "runs": observed["rank"],
            "role": "Effect of rank",
        },
        {
            "stage": "One-axis expert-count sweep",
            "languages": funnel_cfg["sweep_languages"],
            "hours": funnel_cfg["component_budget_hours"],
            "runs": observed["component"],
            "role": "Effect of expert count",
        },
        {
            "stage": "Mixed follow-up",
            "languages": funnel_cfg["sweep_languages"],
            "hours": funnel_cfg["followup_budget_hours"],
            "runs": observed["followup"],
            "role": "Joint configurations and shortlist",
        },
        *[
            {
                "stage": f"Broader validation ({regime} languages)",
                "languages": regime,
                "hours": final_hours[regime],
                "runs": len(finalists),
                "role": "Test the shortlist at scale",
            }
            for regime in regimes
        ],
    ]
    funnel = pd.DataFrame(rows)

    body = [
        " & ".join(
            [
                C.latex_escape(row.stage),
                C.latex_escape(row.languages),
                f"{row.hours:g}",
                str(row.runs),
                C.latex_escape(row.role),
            ]
        )
        + r" \\"
        for row in funnel.itertuples(index=False)
    ]

    # The totals belong in the table rather than in the caption, so that the
    # caption states what the table is and the numbers stay countable.
    body.append(r"\midrule")
    body.append(
        " & ".join(
            [
                "Total",
                "--",
                "--",
                str(training_runs),
                f"{unique_runs} unique configurations; "
                f"{evaluation_runs:,}".replace(",", "{,}") + " evaluation runs",
            ]
        )
        + r" \\"
    )

    table_dir = C.output_dir(cfg, "tables")
    data_dir = C.output_dir(cfg, "data")
    C.write_table(
        table_dir / "tab0_study_funnel.tex",
        "lcccl",
        [r"Stage & Languages & Hours/run & Runs & Purpose \\"],
        body,
        "Stages of the study, with training coverage, per-run budget, and run counts.",
        "tab:study-funnel",
        environment="table*",
        placement="tp",
        col_sep="4pt",
    )

    funnel.loc[len(funnel)] = {
        "stage": "TOTAL",
        "languages": "--",
        "hours": float("nan"),
        "runs": training_runs,
        "role": f"{unique_runs} unique; {evaluation_runs} evaluation runs",
    }
    C.write_csv(funnel, data_dir / "tab0_study_funnel.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    main(parser.parse_args().config)
