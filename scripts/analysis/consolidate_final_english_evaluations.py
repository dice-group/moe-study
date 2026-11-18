#!/usr/bin/env python3
"""Validate the final English extension and merge its baseline rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RECOVERY_ROOT = ROOT / "outputs/pipeline_runs/moe_study_final_english_eval_20260725"
BASELINE_METRICS = ROOT / "outputs/baselines/l-3-1-8b/metrics_long.jsonl"
TASKS = {
    "belebele_eng_Latn",
    "sib200_eng_Latn",
    "taxi1500_eng_Latn",
    "taxi1500_v2_eng_Latn",
}
RUNS = {
    "hydralora_r2",
    "moelora_tr32_e16_k8",
    "moelora_tr64_e32_k8",
    "mola_tr16_e8_k6",
    "mtllora_lr5e5_r8_nup16",
}
PRIMARY_METRICS = {
    "belebele_eng_Latn": "acc_norm,none",
    "sib200_eng_Latn": "acc,none",
    "taxi1500_eng_Latn": "acc,none",
    "taxi1500_v2_eng_Latn": "acc,none",
}
PANEL_SPECS = {
    "v1_96": (186, {"belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_eng_Latn"}),
    "v1_200": (309, {"belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_eng_Latn"}),
    "v2_96": (249, {"belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_v2_eng_Latn"}),
    "v2_200": (408, {"belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_v2_eng_Latn"}),
}


def load_result(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(output_dir.rglob("results_*.json"))
    candidates.extend(sorted(output_dir.rglob("no_language_ids.json")))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f"{output_dir}: expected exactly one result JSON, got {len(candidates)}")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"{candidates[0]}: invalid lm-eval result")
    return candidates[0], payload


def validate_payload(path: Path, payload: dict[str, Any]) -> None:
    results = payload["results"]
    observed = {str(task) for task in results}
    if observed != TASKS:
        raise ValueError(
            f"{path}: task mismatch; missing={sorted(TASKS-observed)}, "
            f"extra={sorted(observed-TASKS)}"
        )
    for task, metric in PRIMARY_METRICS.items():
        value = results[task].get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path}: missing/non-finite primary metric {task}/{metric}")


def validate_raw() -> tuple[list[str], dict[str, Any]]:
    raw_files: list[str] = []
    for size in ("96", "200"):
        benchmark_root = RECOVERY_ROOT / size / "benchmarks"
        observed_runs = {path.name for path in benchmark_root.iterdir() if path.is_dir()}
        if observed_runs != RUNS:
            raise ValueError(
                f"{size}: run mismatch; missing={sorted(RUNS-observed_runs)}, "
                f"extra={sorted(observed_runs-RUNS)}"
            )
        for run_id in sorted(RUNS):
            path, payload = load_result(benchmark_root / run_id)
            validate_payload(path, payload)
            raw_files.append(str(path.resolve()))

    baseline_path, baseline_payload = load_result(RECOVERY_ROOT / "baseline_raw")
    validate_payload(baseline_path, baseline_payload)
    raw_files.append(str(baseline_path.resolve()))
    return raw_files, baseline_payload


def read_cached_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(str(row.get("task", "")), str(row.get("metric", "")))] = row
    return rows


def merge_baseline(payload: dict[str, Any]) -> int:
    rows = read_cached_rows(BASELINE_METRICS)
    added = 0
    for task in sorted(TASKS):
        metrics = payload["results"][task]
        wanted = (
            ("acc_norm,none", "acc,none")
            if task.startswith("belebele_")
            else ("acc,none", "f1,none")
        )
        for metric in wanted:
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            rows[(task, metric)] = {
                "run_id": "baseline::l-3-1-8b",
                "task": task,
                "metric": metric,
                "value": float(value),
                "wandb_project": "",
                "wandb_prefix": "",
                "model": "meta-llama/Llama-3.1-8B",
                "model_slug": "l-3-1-8b",
            }
            added += 1
    if added != 8:
        raise ValueError(f"Expected 8 baseline metric rows, got {added}")

    BASELINE_METRICS.parent.mkdir(parents=True, exist_ok=True)
    temporary = BASELINE_METRICS.with_name(f".{BASELINE_METRICS.name}.tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(rows[key], ensure_ascii=False) + "\n" for key in sorted(rows)),
        encoding="utf-8",
    )
    os.replace(temporary, BASELINE_METRICS)
    return added


def validate_workbooks() -> None:
    expected_ids = {*RUNS, "baseline::l-3-1-8b"}
    for panel, (task_count, english_tasks) in PANEL_SPECS.items():
        workbook = (
            ROOT
            / f"outputs/pipeline_runs/moe_study_final_combined_top5_{panel}"
            / f"moe_study_final_combined_top5_{panel}.xlsx"
        )
        with zipfile.ZipFile(workbook) as archive:
            corrupt = archive.testzip()
            if corrupt is not None or "xl/workbook.xml" not in archive.namelist():
                raise ValueError(f"{workbook}: invalid XLSX archive ({corrupt})")
        data = pd.read_excel(workbook, sheet_name="benchmark_raw")
        observed_ids = set(data["run_id"].astype(str))
        if observed_ids != expected_ids:
            raise ValueError(f"{panel}: unexpected run IDs {sorted(observed_ids)}")
        for run_id in sorted(expected_ids):
            run_rows = data[data["run_id"].astype(str).eq(run_id)]
            if run_rows["task"].nunique() != task_count:
                raise ValueError(
                    f"{panel}/{run_id}: expected {task_count} tasks, "
                    f"got {run_rows['task'].nunique()}"
                )
            observed_english = set(run_rows["task"].astype(str)) & TASKS
            if observed_english != english_tasks:
                raise ValueError(
                    f"{panel}/{run_id}: English task mismatch "
                    f"{sorted(observed_english)}"
                )
            if run_rows[run_rows["task"].isin(english_tasks)]["value"].isna().any():
                raise ValueError(f"{panel}/{run_id}: missing English value")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-workbooks", action="store_true")
    args = parser.parse_args()
    if args.validate_workbooks:
        validate_workbooks()
        print("PASS: all four English-augmented workbooks are complete")
        return

    raw_files, baseline_payload = validate_raw()
    added = merge_baseline(baseline_payload)
    manifest = {
        "status": "complete",
        "tasks": sorted(TASKS),
        "runs": sorted(RUNS),
        "sizes": [96, 200],
        "num_adapter_evaluations": 10,
        "num_baseline_metric_rows": added,
        "baseline_metrics": str(BASELINE_METRICS.resolve()),
        "baseline_metrics_sha256": hashlib.sha256(BASELINE_METRICS.read_bytes()).hexdigest(),
        "raw_result_files": raw_files,
    }
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    (RECOVERY_ROOT / "consolidation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
