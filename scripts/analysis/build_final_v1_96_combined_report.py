#!/usr/bin/env python3
"""Assemble the finalized 96-language v1 reports into one pipeline workbook."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from pipeline_tools.reporting.artifacts import build_workbook
from pipeline_tools.reporting.sources import (
    iter_eval_rows,
    load_baseline_rows,
    load_baseline_rows_from_csv,
    load_baseline_rows_from_lm_eval_json,
    load_resource_map,
    resolve_baseline_lm_eval_path,
)
from pipeline_tools.reporting.workbook import _merge_baseline_rows_with_precedence


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pipeline_runs/moe_study_final_v1_96_combined_report_20260724"
SOURCES = {
    "macro3": ROOT / "outputs/pipeline_runs/moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1",
    "macro45": ROOT / "outputs/pipeline_runs/moe_study_final_macro45_noctua2_a100_4gpu_96_langs_34p5h_v1",
    "mtllora_lr5e5": ROOT / "outputs/pipeline_runs/moe_study_final_mtllora_lr5e5_noctua2_a100_4gpu_96_langs_34p5h_v1",
}
EXPECTED_RUNS = {
    "macro3": {"moelora_tr32_e16_k8", "hydralora_r2"},
    "macro45": {"moelora_tr64_e32_k8", "mola_tr16_e8_k6"},
    "mtllora_lr5e5": {"mtllora_lr5e5_r8_nup16"},
}


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "runs").mkdir(parents=True)
    (OUT / "benchmarks").mkdir()

    summary_rows: list[dict[str, str]] = []
    benchmark_runs: list[dict] = []
    source_metadata: list[dict[str, str]] = []

    for group, root in SOURCES.items():
        rows = load_summary(root / "summary.csv")
        actual_runs = {row["run_id"] for row in rows}
        if actual_runs != EXPECTED_RUNS[group]:
            raise ValueError(f"{group}: expected {EXPECTED_RUNS[group]}, got {actual_runs}")
        summary_rows.extend(rows)

        summary = json.loads((root / "benchmarks/benchmark_summary.json").read_text())
        expected = summary.get("expected_task_breakdown", {})
        for run in summary.get("runs", []):
            run_id = str(run.get("run_id", ""))
            if run_id not in EXPECTED_RUNS[group]:
                continue
            if run.get("missing_tasks") or int(run.get("num_tasks_seen", 0)) != int(run.get("num_tasks_expected", 0)):
                raise ValueError(f"Incomplete v1 benchmark: {group}/{run_id}")
            link(root / "runs" / run_id, OUT / "runs" / run_id)
            link(root / "benchmarks" / run_id, OUT / "benchmarks" / run_id)
            benchmark_runs.append(run)
        source_metadata.append({"group": group, "root": str(root), "expected_task_breakdown": expected})

    if len(summary_rows) != 5 or {row["run_id"] for row in summary_rows} != set().union(*EXPECTED_RUNS.values()):
        raise ValueError("Combined report must contain exactly the five valid v1 runs")

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    benchmark_summary = {
        "report_type": "combined_v1_96",
        "source_reports": source_metadata,
        "runs": benchmark_runs,
        "num_selected_tasks": 183,
        "expected_task_breakdown": {"Belebele": 34, "FLORES": 92, "SIB200": 34, "Taxi1500": 23},
        "strict_completeness": True,
    }
    (OUT / "benchmarks/benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2) + "\n")

    profile = {
        "report_type": "combined_v1_96",
        "sources": source_metadata,
        "benchmark": benchmark_summary["expected_task_breakdown"],
        "training": {"runs": [{"id": row["run_id"], "kind": row["kind"]} for row in summary_rows]},
    }
    (OUT / "combined_profile.json").write_text(json.dumps(profile, indent=2) + "\n")

    eval_rows = iter_eval_rows(OUT / "benchmarks")
    observed = {row["run_id"] for row in eval_rows}
    expected_runs = {row["run_id"] for row in summary_rows}
    if observed != expected_runs:
        raise ValueError(f"Raw benchmark rows missing for: {sorted(expected_runs - observed)}")

    baseline_root = ROOT / "outputs/baselines"
    model = "meta-llama/Llama-3.1-8B"
    baseline_json = resolve_baseline_lm_eval_path(ROOT / "outputs/baselines/llama31_8b_full_eval")
    baseline_csvs = [
        ROOT / "configs/baselines/llama31_8b_full_eval/l-3-1-8b_full_baseline_metrics_summary.csv",
        ROOT / "configs/baselines/llama31_8b_sib_taxi_eval/l-3-1-8b_sib_taxi_baseline_metrics_summary.csv",
    ]
    baseline_rows = _merge_baseline_rows_with_precedence(
        (0, load_baseline_rows(baseline_root, model)),
        (1, load_baseline_rows_from_lm_eval_json(baseline_json, model) if baseline_json else []),
        (2, [row for path in baseline_csvs for row in load_baseline_rows_from_csv(path)]),
    )

    build_workbook(
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        baseline_rows=baseline_rows,
        resource_map=load_resource_map(ROOT / "result_analysis/l2v_nested_samples/fineweb2-language-distribution.csv"),
        profile_payload=profile,
        runs_root=OUT / "runs",
        sample_manifest_path=SOURCES["macro3"] / "sampled/manifest.json",
        tokenized_dir=SOURCES["macro3"] / "tokenized",
        benchmark_summary_path=OUT / "benchmarks/benchmark_summary.json",
        repo_root=ROOT,
        profile_name="moe_study_final_v1_96_combined_report_20260724",
        logs_root=None,
        out_path=OUT / "moe_study_final_v1_96_combined_report_20260724.xlsx",
    )
    print(f"[OK] wrote {OUT / 'moe_study_final_v1_96_combined_report_20260724.xlsx'}")


if __name__ == "__main__":
    main()
