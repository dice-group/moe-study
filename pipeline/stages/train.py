from __future__ import annotations

from pathlib import Path

from .runtime import (
    _build_run_label,
    _dump_json,
    _load_json,
    _missing_modules,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..checks import check_train
from ..context import PipelineContext
from ..training import build_training_cmd, run_training_variant


def stage_train(ctx: PipelineContext) -> None:
    if "train" in ctx.active_stages and bool(ctx.train_cfg.get("enabled", True)):
        missing = _missing_modules(["datasets", "transformers", "torch"])
        if missing:
            raise SystemExit(f"Missing Python modules for train stage: {missing}")
        train_cfg_local = dict(ctx.train_cfg)
        if bool(ctx.wandb_cfg.get("enabled", False)):
            train_cfg_local["report_to"] = "wandb"
            train_cfg_local["train_run_prefix"] = str(ctx.wandb_cfg.get("train_prefix", "train"))
        train_cfg_for_stage = dict(train_cfg_local)
        resolved_model_path = str(train_cfg_for_stage.get("model_name_or_path", ""))
        if bool(ctx.tokenizer_extension_cfg.get("enabled", False)) and bool(ctx.embedding_init_cfg.get("enabled", False)):
            if not ctx.embedding_init_manifest_path.exists():
                raise SystemExit(
                    f"Training with embedding init requires manifest at {ctx.embedding_init_manifest_path}. "
                    "Run embedding_init stage first."
                )
            embedding_manifest = _load_json(ctx.embedding_init_manifest_path)
            resolved_model_path = str(embedding_manifest.get("initialized_model_dir", resolved_model_path))
        train_cfg_for_stage["model_name_or_path"] = resolved_model_path
        runs = train_cfg_local.get("runs", [])
        if not runs:
            raise SystemExit("No training runs configured in profile.training.runs")
        ctx.train_root.mkdir(parents=True, exist_ok=True)
        signature = _stage_signature(
            {
                "stage": "train",
                "cfg": train_cfg_for_stage,
                "embedding_training": ctx.embedding_training_cfg,
                "tokenized_dir": str(ctx.tokenized_dir),
            }
        )
        expected = [ctx.train_root / run["id"] / "train_results.json" for run in runs]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="train",
            signature=signature,
            expected_artifacts=expected,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] train stage (cache hit)")
        else:
            # Collect pending runs (skip already-complete ones).
            pending: list[
                tuple[dict, str, str, Path, Path]
            ] = []  # (run_cfg, run_id, run_name_label, output_dir, log_path)
            for run in runs:
                run_id = str(run["id"])
                run_name_label = _build_run_label(
                    base_run_label=ctx.base_run_label,
                    run=run,
                    default_num_experts=int(train_cfg_local.get("num_experts", 0)),
                )
                output_dir = ctx.train_root / run_id
                train_results = output_dir / "train_results.json"
                trainer_state = output_dir / "trainer_state.json"
                done = output_dir / ".training_complete"
                if (
                    "train" not in ctx.force_stages
                    and (not ctx.args.dry_run)
                    and train_results.exists()
                    and trainer_state.exists()
                    and done.exists()
                ):
                    ctx.logger.info("[SKIP] train run %s (already complete)", run_id)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                run_log_dir = ctx.logs_root / "train_runs" / run_id
                run_log_path = run_log_dir / f"{run_name_label}.train.log"
                pending.append((run, run_id, run_name_label, output_dir, run_log_path))

            if pending and hasattr(ctx.executor, "run_batch") and len(pending) > 1:
                # Parallel path: build all commands, submit as a batch.
                jobs: list[tuple[list[str], dict[str, str], Path]] = []
                for run, _run_id, run_name_label, output_dir, log_path in pending:
                    cmd, run_env = build_training_cmd(
                        repo_root=ctx.repo_root,
                        env=ctx.env,
                        output_dir=output_dir,
                        run_cfg=run,
                        common=train_cfg_for_stage,
                        language_map_path=ctx.language_map_path,
                        tokenized_path=ctx.tokenized_dir,
                        embedding_training_cfg=ctx.embedding_training_cfg,
                        run_name=run_name_label,
                    )
                    jobs.append((cmd, run_env, log_path))
                ctx.logger.info(
                    "[BATCH] Submitting %d training runs via %s",
                    len(jobs),
                    type(ctx.executor).__name__,
                )
                ctx.executor.run_batch(jobs, dry_run=ctx.args.dry_run)
                # After batch completes, mark each output dir as done.
                if not ctx.args.dry_run:
                    for _run, _run_id, _label, output_dir, _log_path in pending:
                        (output_dir / ".training_complete").write_text("ok\n", encoding="utf-8")
            else:
                # Sequential path: one run at a time (single run or LocalExecutor).
                for run, _run_id, run_name_label, output_dir, log_path in pending:
                    run_training_variant(
                        repo_root=ctx.repo_root,
                        env=ctx.env,
                        output_dir=output_dir,
                        run_cfg=run,
                        common=train_cfg_for_stage,
                        language_map_path=ctx.language_map_path,
                        tokenized_path=ctx.tokenized_dir,
                        embedding_training_cfg=ctx.embedding_training_cfg,
                        run_name=run_name_label,
                        log_path=log_path,
                        dry_run=ctx.args.dry_run,
                        executor=ctx.executor,
                    )

            # Write lineage manifests for all completed runs.
            if not ctx.args.dry_run:
                for run, run_id, _label, output_dir, _log_path in pending:
                    _dump_json(
                        output_dir / "model_tokenizer_lineage.json",
                        {
                            "run_id": run_id,
                            "model_name_or_path": resolved_model_path,
                            "tokenized_path": str(ctx.tokenized_dir.resolve()),
                            "tokenizer_extension_manifest": (
                                str(ctx.extension_manifest_path.resolve()) if ctx.extension_manifest_path.exists() else ""
                            ),
                            "embedding_init_manifest": (
                                str(ctx.embedding_init_manifest_path.resolve())
                                if ctx.embedding_init_manifest_path.exists()
                                else ""
                            ),
                            "embedding_training_enabled": bool(ctx.embedding_training_cfg.get("enabled", False)),
                            "additional_target": str(ctx.embedding_training_cfg.get("additional_target", "")),
                        },
                    )
                _save_stage_state(ctx.state_path, "train", signature, [str(path) for path in expected])
                check_train(ctx.train_root, runs, dry_run=ctx.args.dry_run)
