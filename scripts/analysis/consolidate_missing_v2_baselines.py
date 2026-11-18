#!/usr/bin/env python3
"""Merge validated missing-task baseline shards into the report baseline cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODEL = "meta-llama/Llama-3.1-8B"
SLUG = "l-3-1-8b"
SHARDS = ["belebele", "flores", "sib200", "taxi_00", "taxi_01", "taxi_02", "taxi_03"]


def read_tasks(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_result(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(output_dir.rglob("results_*.json"))
    if len(candidates) != 1:
        raise ValueError(f"{output_dir}: expected one result JSON, got {len(candidates)}")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"Invalid lm-eval result: {candidates[0]}")
    return candidates[0], payload


def wanted_metrics(task: str) -> tuple[str, ...]:
    if task.startswith("belebele_"):
        return ("acc_norm,none",)
    if task.startswith("flores_en_perplexity_"):
        return ("word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none")
    if task.startswith(("sib200_", "taxi1500_")):
        return ("acc,none", "f1,none")
    raise ValueError(f"Unknown task family: {task}")


def existing_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows[(str(row.get("task", "")), str(row.get("metric", "")))] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery-root",
        type=Path,
        default=ROOT / "outputs/pipeline_runs/moe_study_v2_missing_baseline_recovery_20260725",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "outputs/baselines",
    )
    args = parser.parse_args()
    recovery = args.recovery_root.resolve()
    baseline_root = args.baseline_root.resolve()
    expected_union = read_tasks(recovery / "tasks_union.txt")
    if len(expected_union) != 111:
        raise ValueError(f"Expected 111 tasks, got {len(expected_union)}")

    new_rows: dict[tuple[str, str], dict[str, Any]] = {}
    raw_files: list[str] = []
    observed_union: set[str] = set()
    for shard in SHARDS:
        selected = read_tasks(recovery / f"tasks_{shard}.txt")
        result_path, payload = load_result(recovery / "shards" / shard)
        results = payload["results"]
        observed = {str(task) for task in results}
        if observed != selected:
            raise ValueError(
                f"{shard}: task mismatch; missing={sorted(selected-observed)[:5]}, "
                f"extra={sorted(observed-selected)[:5]}"
            )
        observed_union.update(observed)
        raw_files.append(str(result_path))
        for task in sorted(selected):
            metrics = results[task]
            if not isinstance(metrics, dict):
                raise ValueError(f"{shard}/{task}: metrics are not an object")
            found_primary = False
            for metric in wanted_metrics(task):
                value = metrics.get(metric)
                if not isinstance(value, (int, float)):
                    continue
                new_rows[(task, metric)] = {
                    "run_id": f"baseline::{SLUG}",
                    "task": task,
                    "metric": metric,
                    "value": float(value),
                    "wandb_project": "",
                    "wandb_prefix": "",
                    "model": MODEL,
                    "model_slug": SLUG,
                }
                if metric in {"acc_norm,none", "bits_per_byte,none", "acc,none"}:
                    found_primary = True
            if not found_primary:
                raise ValueError(f"{shard}/{task}: primary metric is missing")
    if observed_union != expected_union:
        raise ValueError("The seven baseline shards do not cover the exact 111-task union")

    model_dir = baseline_root / SLUG
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / "metrics_long.jsonl"
    merged = existing_rows(metrics_path)
    merged.update(new_rows)
    metrics_path.write_text(
        "".join(json.dumps(merged[key], ensure_ascii=False) + "\n" for key in sorted(merged)),
        encoding="utf-8",
    )
    manifest = {
        "status": "complete",
        "model": MODEL,
        "model_slug": SLUG,
        "num_new_tasks": len(expected_union),
        "num_new_metric_rows": len(new_rows),
        "metrics_long_jsonl": str(metrics_path),
        "raw_result_files": raw_files,
    }
    baseline_root.mkdir(parents=True, exist_ok=True)
    (baseline_root / "manifest.json").write_text(json.dumps({"rows": [manifest]}, indent=2) + "\n")
    (recovery / "baseline_consolidation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
