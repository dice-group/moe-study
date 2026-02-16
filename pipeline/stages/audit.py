from __future__ import annotations

from .runtime import (
    _dump_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
    _validate_lpr_contract,
)
from ..context import PipelineContext


def stage_audit(ctx: PipelineContext) -> None:
    if "audit" in ctx.active_stages and bool(ctx.audit_cfg.get("enabled", True)):
        missing = _missing_modules(["datasets"])
        if missing:
            raise SystemExit(f"Missing Python modules for audit stage: {missing}")
        signature = _stage_signature(
            {
                "stage": "audit",
                "cfg": ctx.audit_cfg,
                "tokenized_dir": str(ctx.tokenized_dir),
                "train_cfg": ctx.train_cfg,
                "language_map_path": str(ctx.language_map_path),
                "sample_languages": ctx.sample_cfg.get("languages", []),
            }
        )
        artifacts = [ctx.tokenized_dir, ctx.lpr_contract_path]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="audit",
            signature=signature,
            expected_artifacts=artifacts,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] audit stage (cache hit)")
        else:
            cmd = [
                "python3",
                str(ctx.repo_root / "pipeline_tools" / "audit_pad_label_mask.py"),
                "--dataset-path",
                str(ctx.tokenized_dir),
                "--split",
                str(ctx.audit_cfg.get("split", "all")),
                "--pad-label-id",
                str(int(ctx.audit_cfg.get("pad_label_id", -100))),
                "--max-rows",
                str(int(ctx.audit_cfg.get("max_rows", 1024))),
            ]
            if bool(ctx.audit_cfg.get("strict", True)):
                cmd.append("--strict")
            _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="audit", executor=ctx.executor)
            lpr_contract = _validate_lpr_contract(
                train_cfg=ctx.train_cfg,
                sample_cfg=ctx.sample_cfg,
                language_map_path=ctx.language_map_path,
            )
            if not ctx.args.dry_run:
                _dump_json(ctx.lpr_contract_path, lpr_contract)
                ctx.logger.info("[OK] lpr_contract=%s", ctx.lpr_contract_path)
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "audit", signature, [str(ctx.tokenized_dir), str(ctx.lpr_contract_path)])
