from __future__ import annotations

from .runtime import (
    _load_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..checks import check_tokenize
from ..context import PipelineContext


def stage_tokenize(ctx: PipelineContext) -> None:
    if "tokenize" in ctx.active_stages and bool(ctx.tokenize_cfg.get("enabled", True)):
        missing = _missing_modules(["datasets", "transformers"])
        if missing:
            raise SystemExit(f"Missing Python modules for tokenize stage: {missing}")
        tokenize_model_name = str(ctx.tokenize_cfg["model_name"])
        if bool(ctx.tokenizer_extension_cfg.get("enabled", False)) and ctx.extension_manifest_path.exists():
            extension_manifest = _load_json(ctx.extension_manifest_path)
            tokenize_model_name = str(extension_manifest.get("extended_tokenizer_dir", tokenize_model_name))
        signature = _stage_signature(
            {
                "stage": "tokenize",
                "cfg": ctx.tokenize_cfg,
                "sample_manifest": str(ctx.sampled_manifest),
                "tokenizer_model_name": tokenize_model_name,
            }
        )
        artifacts = [ctx.tokenized_dir / "dataset_info.json", ctx.tokenized_dir / "state.json"]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="tokenize",
            signature=signature,
            expected_artifacts=artifacts,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] tokenize stage (cache hit)")
        else:
            cmd = [
                "python3",
                str(ctx.repo_root / "data_prep" / "merlin_data_prep" / "distributed_data_processor" / "src" / "tokenize_slurm.py"),
                "--shard_dir",
                str(ctx.sample_shard_dir),
                "--save_tokenized_data_dir",
                str(ctx.tokenized_dir),
                "--model_name",
                tokenize_model_name,
                "--num_proc",
                str(int(ctx.tokenize_cfg.get("num_proc", 1))),
                "--eval_fraction",
                str(float(ctx.tokenize_cfg.get("eval_fraction", 0.02))),
                "--eval_seed",
                str(int(ctx.tokenize_cfg.get("eval_seed", 42))),
                "--sample_fraction",
                str(float(ctx.tokenize_cfg.get("sample_fraction", 1.0))),
                "--sample_seed",
                str(int(ctx.tokenize_cfg.get("sample_seed", 42))),
                "--max_length",
                str(int(ctx.tokenize_cfg.get("max_length", 1024))),
                "--language-map",
                str(ctx.language_map_path),
                "--mask_pad_labels",
                "true" if bool(ctx.tokenize_cfg.get("mask_pad_labels", True)) else "false",
            ]
            max_samples = int(ctx.tokenize_cfg.get("max_samples", 0))
            if max_samples > 0:
                cmd.extend(["--max_samples", str(max_samples)])
            _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="tokenize", executor=ctx.executor)
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "tokenize", signature, [str(ctx.tokenized_dir)])
            check_tokenize(ctx.tokenized_dir, dry_run=ctx.args.dry_run)
