#!/usr/bin/env python3
"""Create a clearly labelled interim Macro3 200-language benchmark workbook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline_tools.reporting.common import safe_json_load
from pipeline_tools.reporting.sources import (
    iter_eval_rows,
    load_baseline_rows,
    load_baseline_rows_from_csv,
    load_baseline_rows_from_lm_eval_json,
    load_resource_map,
    resolve_baseline_lm_eval_path,
)
from pipeline_tools.reporting.workbook import build_workbook, _merge_baseline_rows_with_precedence


RUN_ROOT = ROOT / "outputs/pipeline_runs/moe_study_final_macro3_noctua2_a100_4gpu_200_langs_72h_v1"
RUN_IDS = {"moelora_tr32_e16_k8": "moelora", "hydralora_r2": "hydralora"}
OUT = RUN_ROOT / "moe_study_final_macro3_200_raw_benchmark_report_20260724.xlsx"


def main() -> None:
    benchmark_summary_path = RUN_ROOT / "benchmarks/benchmark_summary.json"
    benchmark_summary = safe_json_load(benchmark_summary_path)
    coverage = {str(row.get("run_id", "")): row for row in benchmark_summary.get("runs", []) if isinstance(row, dict)}
    for run_id in RUN_IDS:
        row = coverage.get(run_id, {})
        if int(row.get("num_tasks_seen", 0)) != 306 or row.get("missing_tasks"):
            raise ValueError(f"Incomplete benchmark for {run_id}: {row}")

    summary_rows = []
    for run_id, kind in RUN_IDS.items():
        train = safe_json_load(RUN_ROOT / "runs" / run_id / "train_results.json")
        summary_rows.append({
            "run_id": run_id,
            "kind": kind,
            "train_loss": train.get("train_loss", ""),
            "train_runtime": train.get("train_runtime", ""),
            "eval_loss": "",  # Explicitly withheld until the queued final-loss stage completes.
        })

    eval_rows = [row for row in iter_eval_rows(RUN_ROOT / "benchmarks") if row.get("run_id") in RUN_IDS]
    if {row.get("run_id") for row in eval_rows} != set(RUN_IDS):
        raise ValueError("Raw benchmark rows are missing for a Macro3 run")

    model = "meta-llama/Llama-3.1-8B"
    baseline_json = resolve_baseline_lm_eval_path(ROOT / "outputs/baselines/llama31_8b_full_eval")
    baseline_rows = _merge_baseline_rows_with_precedence(
        (0, load_baseline_rows(ROOT / "outputs/baselines", model)),
        (1, load_baseline_rows_from_lm_eval_json(baseline_json, model) if baseline_json else []),
        (2, [
            row
            for path in [
                ROOT / "configs/baselines/llama31_8b_full_eval/l-3-1-8b_full_baseline_metrics_summary.csv",
                ROOT / "configs/baselines/llama31_8b_sib_taxi_eval/l-3-1-8b_sib_taxi_baseline_metrics_summary.csv",
            ]
            for row in load_baseline_rows_from_csv(path)
        ]),
    )

    profile = safe_json_load(RUN_ROOT / "state/resolved_profile.json")
    profile["report_status"] = "INTERIM: raw benchmarks complete; final-loss validation and final pipeline report pending"
    profile["included_runs"] = sorted(RUN_IDS)
    profile["excluded_run"] = "mtllora_r8_nup16 (diverged; no benchmark score is reported)"

    build_workbook(
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        baseline_rows=baseline_rows,
        resource_map=load_resource_map(ROOT / "result_analysis/l2v_nested_samples/fineweb2-language-distribution.csv"),
        profile_payload=profile,
        runs_root=RUN_ROOT / "runs",
        sample_manifest_path=RUN_ROOT / "sampled/manifest.json",
        tokenized_dir=RUN_ROOT / "tokenized",
        benchmark_summary_path=benchmark_summary_path,
        repo_root=ROOT,
        profile_name="moe_study_final_macro3_200_raw_benchmark_report_20260724",
        logs_root=RUN_ROOT / "logs",
        out_path=OUT,
    )
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
