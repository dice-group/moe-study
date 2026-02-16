from __future__ import annotations

from .runtime import (
    _abs,
    _load_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..context import PipelineContext


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
                "--output-xlsx",
                str(ctx.report_xlsx),
            ]
            _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="report", executor=ctx.executor)
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "report", signature, [str(ctx.report_xlsx)])
            ctx.logger.info("[OK] report_xlsx=%s", ctx.report_xlsx)
