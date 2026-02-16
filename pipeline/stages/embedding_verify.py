from __future__ import annotations

from .runtime import (
    _dump_json,
    _load_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..context import PipelineContext


def stage_embedding_verify(ctx: PipelineContext) -> None:
    if "embedding_verify" in ctx.active_stages:
        verify_enabled = bool(ctx.embedding_training_cfg.get("enabled", False)) and bool(
            ctx.embedding_training_cfg.get("verify_after_train", True)
        )
        if not verify_enabled:
            ctx.logger.info("[SKIP] embedding_verify stage (disabled)")
        else:
            missing = _missing_modules(["transformers", "torch", "safetensors"])
            if missing:
                raise SystemExit(f"Missing Python modules for embedding_verify stage: {missing}")
            if not ctx.embedding_init_manifest_path.exists():
                raise SystemExit(
                    f"Embedding verification requires embedding init manifest at {ctx.embedding_init_manifest_path}."
                )
            if not ctx.extension_manifest_path.exists():
                raise SystemExit(
                    f"Embedding verification requires tokenizer extension manifest at {ctx.extension_manifest_path}."
                )
            signature = _stage_signature(
                {
                    "stage": "embedding_verify",
                    "cfg": ctx.embedding_training_cfg,
                    "embedding_init_manifest": str(ctx.embedding_init_manifest_path),
                    "extension_manifest": str(ctx.extension_manifest_path),
                    "runs": [run.get("id") for run in ctx.train_cfg.get("runs", [])],
                }
            )
            expected = [ctx.embedding_verify_summary_path]
            if _should_skip_stage(
                state_path=ctx.state_path,
                stage="embedding_verify",
                signature=signature,
                expected_artifacts=expected,
                force_stages=ctx.force_stages,
                dry_run=ctx.args.dry_run,
            ):
                ctx.logger.info("[SKIP] embedding_verify stage (cache hit)")
            else:
                ctx.embedding_verify_root.mkdir(parents=True, exist_ok=True)
                ext_manifest = _load_json(ctx.extension_manifest_path)
                init_manifest = _load_json(ctx.embedding_init_manifest_path)
                run_results: list[dict[str, object]] = []
                all_passed = True
                for run in ctx.train_cfg.get("runs", []):
                    run_id = str(run["id"])
                    run_dir = ctx.train_root / run_id
                    run_manifest = ctx.embedding_verify_root / f"{run_id}.json"
                    cmd = [
                        "python3",
                        str(ctx.repo_root / "pipeline_tools" / "verify_embedding_integration.py"),
                        "--run-dir",
                        str(run_dir),
                        "--initialized-model-dir",
                        str(init_manifest["initialized_model_dir"]),
                        "--base-tokenizer",
                        str(ext_manifest["base_tokenizer"]),
                        "--extended-tokenizer",
                        str(ext_manifest["extended_tokenizer_dir"]),
                        "--manifest-path",
                        str(run_manifest),
                    ]
                    if bool(ctx.embedding_training_cfg.get("require_weight_change", True)):
                        cmd.append("--require-weight-change")
                    _run_stage_cmd(
                        cmd=cmd,
                        env=ctx.env,
                        dry_run=ctx.args.dry_run,
                        logs_root=ctx.logs_root,
                        stage="embedding_verify",
                        suffix=run_id,
                        executor=ctx.executor,
                    )
                    if not ctx.args.dry_run:
                        run_payload = _load_json(run_manifest)
                        passed = bool(run_payload.get("pass", False))
                        all_passed = all_passed and passed
                        run_results.append(
                            {
                                "run_id": run_id,
                                "manifest": str(run_manifest.resolve()),
                                "pass": passed,
                                "weights_changed": bool(run_payload.get("weights_changed", False)),
                            }
                        )
                if not ctx.args.dry_run:
                    _dump_json(
                        ctx.embedding_verify_summary_path,
                        {
                            "pass": all_passed,
                            "runs": run_results,
                            "require_weight_change": bool(ctx.embedding_training_cfg.get("require_weight_change", True)),
                        },
                    )
                    if not all_passed:
                        raise SystemExit(f"Embedding verification failed: {ctx.embedding_verify_summary_path}")
                    _save_stage_state(ctx.state_path, "embedding_verify", signature, [str(ctx.embedding_verify_summary_path)])
                    ctx.logger.info("[OK] embedding_verify_summary=%s", ctx.embedding_verify_summary_path)
