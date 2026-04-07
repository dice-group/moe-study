from __future__ import annotations

from .runtime import (
    _abs,
    _dump_json,
    _load_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..context import PipelineContext

DEFAULT_BASELINE_METRICS_CSVS = [
    "configs/baselines/llama31_8b_full_eval/l-3-1-8b_full_baseline_metrics_summary.csv",
    "configs/baselines/llama31_8b_sib_taxi_eval/l-3-1-8b_sib_taxi_baseline_metrics_summary.csv",
]


def stage_report(ctx: PipelineContext) -> None:
    if "report" in ctx.active_stages and bool(ctx.report_cfg.get("enabled", False)):
        missing = _missing_modules(["openpyxl"])
        if missing:
            raise SystemExit(f"Missing Python modules for report stage: {missing}")
        benchmark_summary_path = ctx.benchmark_root / "benchmark_summary.json"
        signature = _stage_signature(
            {
                "stage": "report",
                "cfg": ctx.report_cfg,
                "summary_csv": str(ctx.summary_path),
                "benchmark_summary": str(benchmark_summary_path),
            }
        )
        expected = [ctx.report_xlsx]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="report",
            signature=signature,
            expected_artifacts=expected,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] report stage (cache hit)")
        else:
            resolved_profile_path = ctx.output_root / "state" / "resolved_profile.json"
            _dump_json(resolved_profile_path, ctx.profile)
            cmd = [
                "python3",
                str(ctx.repo_root / "pipeline_tools" / "export_proof_results_to_excel.py"),
                "--summary-csv",
                str(ctx.summary_path),
                "--benchmark-dir",
                str(ctx.benchmark_root),
                "--benchmark-summary-json",
                str(benchmark_summary_path),
                "--baseline-cache-root",
                str(_abs(ctx.repo_root, str(ctx.report_cfg.get("baseline_cache_root", str(ctx.baseline_cache_root))))),
                "--model-name-or-path",
                str(
                    _load_json(ctx.embedding_init_manifest_path).get(
                        "initialized_model_dir", ctx.train_cfg.get("model_name_or_path", "")
                    )
                    if ctx.embedding_init_manifest_path.exists()
                    else ctx.train_cfg.get("model_name_or_path", "")
                ),
                "--sample-manifest",
                str(ctx.sampled_manifest),
                "--tokenized-dir",
                str(ctx.tokenized_dir),
                "--profile-json",
                str(resolved_profile_path),
                "--repo-root",
                str(ctx.repo_root),
                "--profile-name",
                str(ctx.args.profile),
                "--logs-root",
                str(ctx.logs_root),
                "--output-xlsx",
                str(ctx.report_xlsx),
            ]
            baseline_metrics_csv_values: list[str] = []
            baseline_metrics_csv = str(ctx.report_cfg.get("baseline_metrics_csv", "")).strip()
            if baseline_metrics_csv:
                baseline_metrics_csv_values.append(baseline_metrics_csv)
            baseline_metrics_csvs = ctx.report_cfg.get("baseline_metrics_csvs", [])
            if isinstance(baseline_metrics_csvs, list):
                for csv_path in baseline_metrics_csvs:
                    value = str(csv_path).strip()
                    if value:
                        baseline_metrics_csv_values.append(value)
            if not baseline_metrics_csv_values:
                baseline_metrics_csv_values = list(DEFAULT_BASELINE_METRICS_CSVS)
            for csv_path in baseline_metrics_csv_values:
                cmd.extend(["--baseline-metrics-csv", str(_abs(ctx.repo_root, csv_path))])
            _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="report", executor=ctx.executor)
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "report", signature, [str(ctx.report_xlsx)])
            ctx.logger.info("[OK] report_xlsx=%s", ctx.report_xlsx)
