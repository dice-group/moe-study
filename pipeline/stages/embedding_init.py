from __future__ import annotations

from .runtime import (
    _load_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..checks import check_embedding_init
from ..context import PipelineContext


def stage_embedding_init(ctx: PipelineContext) -> None:
    if "embedding_init" in ctx.active_stages:
        init_enabled = bool(ctx.embedding_init_cfg.get("enabled", False))
        if not init_enabled:
            ctx.logger.info("[SKIP] embedding_init stage (disabled)")
        else:
            missing = _missing_modules(["transformers", "torch"])
            if missing:
                raise SystemExit(f"Missing Python modules for embedding_init stage: {missing}")
            signature = _stage_signature(
                {
                    "stage": "embedding_init",
                    "cfg": ctx.embedding_init_cfg,
                    "extension_manifest": str(ctx.extension_manifest_path),
                }
            )
            expected = [ctx.embedding_init_output_dir / "config.json", ctx.embedding_init_manifest_path]
            if _should_skip_stage(
                state_path=ctx.state_path,
                stage="embedding_init",
                signature=signature,
                expected_artifacts=expected,
                force_stages=ctx.force_stages,
                dry_run=ctx.args.dry_run,
            ):
                ctx.logger.info("[SKIP] embedding_init stage (cache hit)")
            else:
                if not ctx.args.dry_run and (not ctx.extension_manifest_path.exists()):
                    raise SystemExit(
                        f"Embedding init requires tokenizer extension manifest at {ctx.extension_manifest_path}. "
                        "Run tokenizer_extend stage first."
                    )
                if ctx.extension_manifest_path.exists():
                    extension_manifest = _load_json(ctx.extension_manifest_path)
                    base_tokenizer = str(extension_manifest["base_tokenizer"])
                    extended_tokenizer = str(extension_manifest["extended_tokenizer_dir"])
                else:
                    base_tokenizer = str(
                        ctx.tokenizer_extension_cfg.get(
                            "base_tokenizer",
                            ctx.tokenize_cfg.get("model_name", ctx.train_cfg.get("model_name_or_path", "")),
                        )
                    )
                    extended_tokenizer = str(ctx.extended_tokenizer_dir)
                model_path = str(ctx.embedding_init_cfg.get("model_path", "")).strip()
                if not model_path:
                    model_path = str(ctx.train_cfg.get("model_name_or_path"))
                cmd = [
                    "python3",
                    str(ctx.repo_root / "pipeline_tools" / "init_extended_embeddings.py"),
                    "--base-tokenizer",
                    base_tokenizer,
                    "--extended-tokenizer",
                    extended_tokenizer,
                    "--model-path",
                    model_path,
                    "--output-model-dir",
                    str(ctx.embedding_init_output_dir),
                    "--manifest-path",
                    str(ctx.embedding_init_manifest_path),
                ]
                _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="embedding_init", executor=ctx.executor)
                if not ctx.args.dry_run:
                    _save_stage_state(ctx.state_path, "embedding_init", signature, [str(path) for path in expected])
                    expected_vocab = extension_manifest.get("new_vocab_size") if ctx.extension_manifest_path.exists() else None
                    check_embedding_init(ctx.embedding_init_output_dir, ctx.embedding_init_manifest_path, expected_vocab, dry_run=ctx.args.dry_run)
                    ctx.logger.info("[OK] embedding_init_manifest=%s", ctx.embedding_init_manifest_path)
