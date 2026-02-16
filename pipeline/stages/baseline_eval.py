from __future__ import annotations

from .common import ensure_lm_eval_layout
from .runtime import _abs, _run_stage_cmd, _save_stage_state, _should_skip_stage, _stage_signature
from ..context import PipelineContext


def stage_baseline_eval(ctx: PipelineContext) -> None:
    if "baseline_eval" in ctx.active_stages and bool(ctx.baseline_eval_cfg.get("enabled", False)):
        signature = _stage_signature({"stage": "baseline_eval", "cfg": ctx.baseline_eval_cfg})
        baseline_manifest = ctx.baseline_cache_root / "manifest.json"
        expected = [baseline_manifest]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="baseline_eval",
            signature=signature,
            expected_artifacts=expected,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] baseline_eval stage (cache hit)")
        else:
            ensure_lm_eval_layout(ctx.repo_root, ctx.logger)
            models = ctx.baseline_eval_cfg.get(
                "models", ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"]
            )
            cmd = [
                "python3",
                str(ctx.repo_root / "pipeline_tools" / "run_base_model_eval_cache.py"),
                "--models",
                ",".join((str(x) for x in models)),
                "--tasks-file",
                str(_abs(ctx.repo_root, str(ctx.baseline_eval_cfg.get("tasks_file", "configs/lm_eval_tasks/200_langs_with_flores.txt")))),
                "--output-root",
                str(ctx.baseline_cache_root),
                "--batch-size",
                str(ctx.baseline_eval_cfg.get("batch_size", "auto")),
                "--lm-eval-bin",
                str(ctx.baseline_eval_cfg.get("lm_eval_bin", "lm_eval")),
                "--seed",
                str(int(ctx.baseline_eval_cfg.get("seed", 42))),
                "--extra-args",
                str(ctx.baseline_eval_cfg.get("extra_args", "")),
            ]
            if bool(ctx.baseline_eval_cfg.get("force", False)):
                cmd.append("--force")
            _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="baseline_eval", executor=ctx.executor)
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "baseline_eval", signature, [str(baseline_manifest)])
            ctx.logger.info("[OK] baseline_manifest=%s", baseline_manifest)
