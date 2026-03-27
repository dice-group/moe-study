#!/usr/bin/env python3
"""Workbook assembly and CLI entrypoint for pipeline report export."""

from __future__ import annotations

import argparse
from pathlib import Path

from openpyxl import Workbook

from .artifacts import (
    build_data_overview,
    collect_benchmark_raw_rows,
    collect_experiment_meta,
    collect_run_configs,
    write_benchmark_raw_sheet,
    write_configs_sheet,
    write_data_overview_sheet,
    write_experiment_meta_sheet,
    write_profile_sheet,
)
from .common import safe_json_load
from .compact import write_comparison_sheet
from .sources import (
    iter_eval_rows,
    load_baseline_rows,
    load_baseline_rows_from_csv,
    load_baseline_rows_from_lm_eval_json,
    load_resource_map,
    read_summary_csv,
    resolve_baseline_lm_eval_path,
)


def build_workbook(
    *,
    summary_rows,
    eval_rows,
    baseline_rows,
    resource_map,
    profile_payload,
    runs_root: Path,
    sample_manifest_path: Path,
    tokenized_dir: Path,
    benchmark_summary_path: Path,
    repo_root: Path,
    profile_name: str,
    logs_root: Path | None,
    out_path: Path,
) -> None:
    wb = Workbook()

    ws_compact = wb.active
    ws_compact.title = "benchmark_compact"
    write_comparison_sheet(
        ws_compact,
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        baseline_rows=baseline_rows,
        resource_map=resource_map,
    )

    ws_data = wb.create_sheet("data_overview")
    write_data_overview_sheet(
        ws_data,
        build_data_overview(
            sample_manifest_path=sample_manifest_path,
            tokenized_dir=tokenized_dir,
            profile_payload=profile_payload,
            benchmark_summary_path=benchmark_summary_path,
        ),
    )

    ws_configs = wb.create_sheet("run_configs")
    write_configs_sheet(ws_configs, collect_run_configs(summary_rows, runs_root, profile_payload, logs_root))

    ws_profile = wb.create_sheet("pipeline_profile")
    write_profile_sheet(ws_profile, profile_payload)

    ws_raw = wb.create_sheet("benchmark_raw")
    write_benchmark_raw_sheet(ws_raw, collect_benchmark_raw_rows(eval_rows, baseline_rows, summary_rows, resource_map))

    ws_meta = wb.create_sheet("experiment_meta")
    write_experiment_meta_sheet(
        ws_meta,
        collect_experiment_meta(
            repo_root=repo_root,
            profile_name=profile_name,
            profile_payload=profile_payload,
            summary_rows=summary_rows,
            benchmark_summary_path=benchmark_summary_path,
            output_xlsx=out_path,
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export proof run results to Excel.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--benchmark-summary-json", default="", help="Legacy/no-op: kept for CLI compatibility.")
    parser.add_argument("--baseline-cache-root", default="")
    parser.add_argument("--baseline-lm-eval-json", default="outputs/baselines/llama31_8b_full_eval")
    parser.add_argument("--baseline-metrics-csv", default="configs/baselines/llama31_8b_full_eval/l-3-1-8b_full_baseline_metrics_summary.csv")
    parser.add_argument("--resource-availability-csv", default="result_analysis/l2v_nested_samples/fineweb2-language-distribution.csv")
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--sample-manifest", default="")
    parser.add_argument("--tokenized-dir", default="")
    parser.add_argument("--profile-json", default="")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--profile-name", default="")
    parser.add_argument("--logs-root", default="")
    parser.add_argument("--output-xlsx", required=True)
    args = parser.parse_args()

    summary_rows = read_summary_csv(Path(args.summary_csv))
    eval_rows = iter_eval_rows(Path(args.benchmark_dir))

    baseline_rows = []
    if args.baseline_cache_root.strip() and args.model_name_or_path.strip():
        baseline_rows = load_baseline_rows(Path(args.baseline_cache_root), args.model_name_or_path)
    baseline_json_rows = []
    if args.baseline_lm_eval_json.strip():
        baseline_json_path = resolve_baseline_lm_eval_path(Path(args.baseline_lm_eval_json))
        if baseline_json_path is not None:
            baseline_json_rows = load_baseline_rows_from_lm_eval_json(baseline_json_path, args.model_name_or_path)
    baseline_csv_rows = load_baseline_rows_from_csv(Path(args.baseline_metrics_csv))
    if baseline_rows or baseline_json_rows or baseline_csv_rows:
        merged = {}
        for row in [*baseline_rows, *baseline_json_rows, *baseline_csv_rows]:
            merged[(str(row["run_id"]), str(row["task"]), str(row["metric"]))] = row
        baseline_rows = list(merged.values())

    resource_map = load_resource_map(Path(args.resource_availability_csv))
    profile_payload = safe_json_load(Path(args.profile_json)) if str(args.profile_json).strip() else {}
    build_workbook(
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        baseline_rows=baseline_rows,
        resource_map=resource_map,
        profile_payload=profile_payload,
        runs_root=Path(args.summary_csv).resolve().parent / "runs",
        sample_manifest_path=Path(args.sample_manifest) if str(args.sample_manifest).strip() else Path(args.summary_csv).resolve().parent / "sampled" / "manifest.json",
        tokenized_dir=Path(args.tokenized_dir) if str(args.tokenized_dir).strip() else Path(args.summary_csv).resolve().parent / "tokenized",
        benchmark_summary_path=Path(args.benchmark_summary_json) if str(args.benchmark_summary_json).strip() else Path(args.summary_csv).resolve().parent / "benchmarks" / "benchmark_summary.json",
        repo_root=Path(args.repo_root).resolve(),
        profile_name=str(args.profile_name).strip(),
        logs_root=Path(args.logs_root).resolve() if str(args.logs_root).strip() else None,
        out_path=Path(args.output_xlsx),
    )
    print(f"[OK] wrote {args.output_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
