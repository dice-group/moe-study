#!/usr/bin/env python3
"""Extract V2 tasks that have no primary-metric Llama-3.1-8B baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKBOOKS = {
    "v2_96": ROOT
    / "outputs/pipeline_runs/moe_study_final_combined_top5_v2_96/moe_study_final_combined_top5_v2_96.xlsx",
    "v2_200": ROOT
    / (
        "outputs/pipeline_runs/moe_study_final_combined_top5_v2_200_interim/"
        "moe_study_final_combined_top5_v2_200_interim.xlsx"
    ),
}


def family(task: str) -> str:
    if task.startswith("belebele_"):
        return "Belebele"
    if task.startswith("flores_en_perplexity_"):
        return "FLORES"
    if task.startswith("sib200_"):
        return "SIB200"
    if task.startswith("taxi1500_"):
        return "Taxi1500"
    return "Other"


def primary_metric(task: str) -> str:
    if task.startswith("belebele_"):
        return "acc_norm"
    if task.startswith("flores_en_perplexity_"):
        return "bits_per_byte"
    return "acc"


def missing_for_workbook(path: Path) -> set[str]:
    data = pd.read_excel(path, sheet_name="benchmark_raw")
    baseline = data[data["is_baseline"].fillna(False).astype(bool)].copy()
    experiments = data[~data["is_baseline"].fillna(False).astype(bool)].copy()
    missing: set[str] = set()
    for task in sorted(set(experiments["task"].astype(str))):
        rows = baseline[
            baseline["task"].astype(str).eq(task)
            & baseline["metric"].astype(str).str.startswith(primary_metric(task))
        ]
        if rows.empty or not pd.to_numeric(rows["value"], errors="coerce").notna().any():
            missing.add(task)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/pipeline_runs/moe_study_v2_missing_baseline_recovery_20260725",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    missing = {name: missing_for_workbook(path) for name, path in WORKBOOKS.items()}
    union = sorted(set().union(*missing.values()))
    by_family = {
        benchmark: sorted(task for task in union if family(task) == benchmark)
        for benchmark in ["Belebele", "FLORES", "SIB200", "Taxi1500"]
    }
    expected = {"Belebele": 4, "FLORES": 10, "SIB200": 5, "Taxi1500": 92}
    actual = {benchmark: len(tasks) for benchmark, tasks in by_family.items()}
    if len(missing["v2_96"]) != 73 or len(missing["v2_200"]) != 111:
        raise ValueError(f"Unexpected workbook gaps: { {key: len(value) for key, value in missing.items()} }")
    if actual != expected or len(union) != 111:
        raise ValueError(f"Unexpected missing-task union: total={len(union)}, breakdown={actual}")
    if not missing["v2_96"].issubset(missing["v2_200"]):
        raise ValueError("V2-96 missing tasks are not a subset of V2-200")

    task_files: dict[str, str] = {}
    shards = {
        "belebele": by_family["Belebele"],
        "flores": by_family["FLORES"],
        "sib200": by_family["SIB200"],
    }
    taxi = by_family["Taxi1500"]
    for index in range(4):
        shards[f"taxi_{index:02d}"] = taxi[index * 23 : (index + 1) * 23]
    for shard, tasks in shards.items():
        path = output_root / f"tasks_{shard}.txt"
        path.write_text("\n".join(tasks) + "\n", encoding="utf-8")
        task_files[shard] = str(path)
    (output_root / "tasks_union.txt").write_text("\n".join(union) + "\n", encoding="utf-8")

    payload = {
        "status": "validated",
        "model": "meta-llama/Llama-3.1-8B",
        "missing_counts": {name: len(tasks) for name, tasks in missing.items()},
        "union_count": len(union),
        "benchmark_breakdown": actual,
        "missing_by_panel": {name: sorted(tasks) for name, tasks in missing.items()},
        "missing_by_benchmark": by_family,
        "shards": {name: {"count": len(tasks), "tasks_file": task_files[name]} for name, tasks in shards.items()},
    }
    (output_root / "missing_baseline_tasks.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "missing_by_panel"}, indent=2))


if __name__ == "__main__":
    main()
