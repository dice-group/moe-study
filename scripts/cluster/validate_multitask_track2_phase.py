#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPECTED_RUNS = [
    "lora_r8",
    "lora_r64",
    "cola",
    "hydralora",
    "moelora",
    "adamole",
    "adamoe_1",
    "mov",
    "mixlora",
    "hmora",
    "mtllora",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_output_root(phase: str) -> Path:
    mapping = {
        "smoke": Path("outputs/pipeline_runs/multitask_track2_cluster_smoke_a100"),
        "pilot": Path("outputs/pipeline_runs/multitask_track2_cluster_pilot_a100"),
        "full_single_seed": Path("outputs/pipeline_runs/multitask_track2_cluster_full_s1_a100"),
    }
    return mapping[phase]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Track 2 rollout outputs.")
    parser.add_argument("--phase", choices=["smoke", "pilot", "full_single_seed"], required=True)
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = repo_root / (args.output_root or str(_expected_output_root(args.phase)))
    benchmark_summary_path = output_root / "benchmarks" / "benchmark_summary.json"
    metrics_csv_path = output_root / "benchmarks" / "benchmark_metrics.csv"

    failures: list[str] = []
    if not benchmark_summary_path.exists():
        failures.append(f"missing benchmark summary: {benchmark_summary_path}")
    if not metrics_csv_path.exists():
        failures.append(f"missing benchmark metrics csv: {metrics_csv_path}")
    for run_id in EXPECTED_RUNS:
        train_results = output_root / "runs" / run_id / "train_results.json"
        if not train_results.exists():
            failures.append(f"missing train results for {run_id}: {train_results}")

    if benchmark_summary_path.exists():
        payload = _load_json(benchmark_summary_path)
        run_rows = {
            str(row.get("run_id")): row
            for row in payload.get("runs", [])
            if isinstance(row, dict) and str(row.get("run_id", "")).strip()
        }
        for run_id in EXPECTED_RUNS:
            row = run_rows.get(run_id)
            if row is None:
                failures.append(f"missing benchmark row for {run_id}")
                continue
            if int(row.get("num_tasks_expected", 0)) != 16:
                failures.append(f"{run_id} expected 16 tasks")
            if int(row.get("num_tasks_seen", 0)) != 16:
                failures.append(f"{run_id} did not see all 16 tasks")
            if list(row.get("missing_tasks", [])) != []:
                failures.append(f"{run_id} has missing tasks: {row.get('missing_tasks')}")

            if args.phase in {"smoke", "pilot"}:
                if list(row.get("unscored_tasks", [])) != ["stsb"]:
                    failures.append(f"{run_id} should have only ['stsb'] unscored in {args.phase}")
            else:
                if int(row.get("num_tasks_scored", 0)) != 16:
                    failures.append(f"{run_id} did not score all 16 tasks")
                if list(row.get("unscored_tasks", [])) != []:
                    failures.append(f"{run_id} has unscored tasks in full_single_seed: {row.get('unscored_tasks')}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1

    print(json.dumps({"phase": args.phase, "output_root": str(output_root), "status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
