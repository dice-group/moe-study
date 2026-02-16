from __future__ import annotations

import json
from pathlib import Path

from .runtime import (
    _load_json,
    _missing_modules,
    _resolve_profile_languages,
    _run_stage_cmd,
    _sanitize_slug,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
    _dump_json,
)
from ..checks import check_tokenizer_extend
from ..context import PipelineContext


def stage_tokenizer_extend(ctx: PipelineContext) -> None:
    if "tokenizer_extend" in ctx.active_stages:
        extension_enabled = bool(ctx.tokenizer_extension_cfg.get("enabled", False))
        if not extension_enabled:
            ctx.logger.info("[SKIP] tokenizer_extend stage (disabled)")
        else:
            missing = _missing_modules(["transformers", "pandas", "torch"])
            if missing:
                raise SystemExit(f"Missing Python modules for tokenizer_extend stage: {missing}")
            signature = _stage_signature(
                {
                    "stage": "tokenizer_extend",
                    "cfg": ctx.tokenizer_extension_cfg,
                    "sample_manifest": str(ctx.sampled_manifest),
                    "sample_shard_dir": str(ctx.sample_shard_dir),
                }
            )
            expected = [ctx.extended_tokenizer_dir / "tokenizer.json", ctx.extension_manifest_path]
            if _should_skip_stage(
                state_path=ctx.state_path,
                stage="tokenizer_extend",
                signature=signature,
                expected_artifacts=expected,
                force_stages=ctx.force_stages,
                dry_run=ctx.args.dry_run,
            ):
                ctx.logger.info("[SKIP] tokenizer_extend stage (cache hit)")
            else:
                ctx.extension_root.mkdir(parents=True, exist_ok=True)
                resolved_langs = _resolve_profile_languages(
                    sample_cfg=ctx.sample_cfg,
                    sample_manifest_path=ctx.sampled_manifest,
                    sampled_dir=ctx.sampled_dir,
                )
                prepare_cmd = [
                    "python3",
                    str(ctx.repo_root / "pipeline_tools" / "prepare_tokenizer_extension_inputs.py"),
                    "--shard-dir",
                    str(ctx.sample_shard_dir),
                    "--output-dir",
                    str(ctx.extension_prepared_dir),
                    "--languages",
                    ",".join(resolved_langs),
                    "--manifest-path",
                    str(ctx.extension_prepared_manifest),
                ]
                _run_stage_cmd(
                    cmd=prepare_cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="tokenizer_extend",
                    suffix="prepare_inputs",
                    executor=ctx.executor,
                )
                base_tokenizer = str(ctx.tokenizer_extension_cfg.get("base_tokenizer", "")).strip()
                if not base_tokenizer:
                    base_tokenizer = str(
                        ctx.tokenize_cfg.get("model_name", ctx.train_cfg.get("model_name_or_path", ""))
                    ).strip()
                multilingual_tokenizer = str(ctx.tokenizer_extension_cfg.get("multilingual_tokenizer_path", "")).strip()
                if not multilingual_tokenizer:
                    multilingual_tokenizer = base_tokenizer
                local_base_tokenizer = base_tokenizer
                local_multilingual_tokenizer = multilingual_tokenizer
                base_vocab_size = None
                base_model_vocab_size = None
                if not ctx.args.dry_run:
                    from transformers import AutoTokenizer

                    materialized_dir = ctx.extension_root / "materialized_tokenizers"
                    materialized_dir.mkdir(parents=True, exist_ok=True)
                    if not Path(base_tokenizer).exists():
                        base_slug = _sanitize_slug(base_tokenizer)
                        local_base_path = materialized_dir / f"base-{base_slug}"
                        if not local_base_path.exists():
                            AutoTokenizer.from_pretrained(base_tokenizer, use_fast=True).save_pretrained(local_base_path)
                        local_base_tokenizer = str(local_base_path)
                    if not Path(multilingual_tokenizer).exists():
                        multi_slug = _sanitize_slug(multilingual_tokenizer)
                        local_multi_path = materialized_dir / f"multi-{multi_slug}"
                        if not local_multi_path.exists():
                            AutoTokenizer.from_pretrained(multilingual_tokenizer, use_fast=True).save_pretrained(local_multi_path)
                        local_multilingual_tokenizer = str(local_multi_path)
                    base_vocab_size = len(AutoTokenizer.from_pretrained(local_base_tokenizer, use_fast=True))
                    tokenizer_json = Path(local_base_tokenizer) / "tokenizer.json"
                    if tokenizer_json.exists():
                        with tokenizer_json.open("r", encoding="utf-8") as handle:
                            tokenizer_data = json.load(handle)
                        base_model_vocab_size = len(tokenizer_data.get("model", {}).get("vocab", {}))
                stage_args = ["base_coverage", "allocation", "extension"]
                train_multilingual = bool(ctx.tokenizer_extension_cfg.get("train_multilingual", False))
                trained_tokenizer_dir = ctx.extension_pipeline_root / "trained_tokenizer"
                if train_multilingual:
                    stage_args = ["base_coverage", "allocation", "training", "extension"]
                if bool(ctx.tokenizer_extension_cfg.get("compute_extended_coverage", False)):
                    stage_args.extend(["extended_coverage", "comparison"])
                total_new_tokens = int(ctx.tokenizer_extension_cfg.get("total_new_tokens", 128000))
                raw_vocab_cap = ctx.tokenizer_extension_cfg.get("extension_vocab_cap", 256000)
                auto_vocab_cap = raw_vocab_cap in (None, "", "auto", "auto_total_new_tokens", "base_plus_total_new_tokens")
                if auto_vocab_cap:
                    if base_model_vocab_size is None:
                        extension_vocab_cap = 256000
                        ctx.logger.info(
                            "[WARN] tokenizer_extend auto vocab cap requested but tokenizer.json vocab size unavailable. "
                            "Falling back to extension_vocab_cap=%s",
                            extension_vocab_cap,
                        )
                    else:
                        extension_vocab_cap = int(base_model_vocab_size + total_new_tokens)
                        ctx.logger.info(
                            "[INFO] tokenizer_extend auto vocab cap -> base_model_vocab=%s base_tokenizer_len=%s total_new_tokens=%s cap=%s",
                            base_model_vocab_size,
                            base_vocab_size,
                            total_new_tokens,
                            extension_vocab_cap,
                        )
                else:
                    extension_vocab_cap = int(raw_vocab_cap)
                ext_cmd = [
                    "python3",
                    "-m",
                    "data_prep.merlin_data_prep.tokenizer_extension.src.pipeline_slurm",
                    "--output-dir",
                    str(ctx.extension_pipeline_root),
                    "--stage",
                    *stage_args,
                    "--base-data-dir",
                    str(ctx.extension_prepared_dir),
                    "--base-tokenizer",
                    local_base_tokenizer,
                    "--base-max-lines",
                    str(int(ctx.tokenizer_extension_cfg.get("base_max_lines", 10000))),
                    "--total-new-tokens",
                    str(total_new_tokens),
                    "--weight-ppw",
                    str(float(ctx.tokenizer_extension_cfg.get("weight_ppw", 0.5))),
                    "--weight-cpt",
                    str(float(ctx.tokenizer_extension_cfg.get("weight_cpt", 0.5))),
                    "--weight-unk",
                    str(float(ctx.tokenizer_extension_cfg.get("weight_unk", 0.0))),
                    "--gamma",
                    str(float(ctx.tokenizer_extension_cfg.get("gamma", 2.0))),
                    "--train-multilingual" if train_multilingual else "",
                    "--train-data-dir" if train_multilingual else "",
                    str(ctx.extension_prepared_dir) if train_multilingual else "",
                    "--train-output-dir" if train_multilingual else "",
                    str(trained_tokenizer_dir) if train_multilingual else "",
                    "--train-vocab-size" if train_multilingual else "",
                    str(int(ctx.tokenizer_extension_cfg.get("train_vocab_size", 32000))) if train_multilingual else "",
                    "--train-base-model" if train_multilingual else "",
                    local_base_tokenizer if train_multilingual else "",
                    "--train-text-key" if train_multilingual else "",
                    str(ctx.tokenizer_extension_cfg.get("train_text_key", "text")) if train_multilingual else "",
                    "--extend",
                    "--extension-data-dir",
                    str(ctx.extension_prepared_dir),
                    "--extension-base-path",
                    local_base_tokenizer,
                    "--multilingual-tokenizer-path",
                    str(trained_tokenizer_dir if train_multilingual else local_multilingual_tokenizer),
                    "--extension-output-dir",
                    str(ctx.extended_tokenizer_dir),
                    "--extension-sample-docs",
                    str(int(ctx.tokenizer_extension_cfg.get("extension_sample_docs", 50))),
                    "--extension-text-key",
                    str(ctx.tokenizer_extension_cfg.get("extension_text_key", "text")),
                    "--extension-vocab-cap",
                    str(extension_vocab_cap),
                ]
                extension_num_workers = ctx.tokenizer_extension_cfg.get("extension_num_workers")
                if extension_num_workers is not None:
                    ext_cmd.extend(["--extension-num-workers", str(int(extension_num_workers))])
                ext_cmd = [item for item in ext_cmd if item != ""]
                extended_max_lines = ctx.tokenizer_extension_cfg.get("extended_max_lines")
                if bool(ctx.tokenizer_extension_cfg.get("compute_extended_coverage", False)) and extended_max_lines is not None:
                    ext_cmd.extend(["--extended-max-lines", str(int(extended_max_lines))])
                _run_stage_cmd(
                    cmd=ext_cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="tokenizer_extend",
                    suffix="pipeline",
                    executor=ctx.executor,
                )
                if not ctx.args.dry_run:
                    from transformers import AutoTokenizer

                    used_multilingual_tokenizer = str(trained_tokenizer_dir if train_multilingual else local_multilingual_tokenizer)
                    base_tok = AutoTokenizer.from_pretrained(local_base_tokenizer, use_fast=True)
                    ext_tok = AutoTokenizer.from_pretrained(str(ctx.extended_tokenizer_dir), use_fast=True)
                    original_vocab = len(base_tok)
                    new_vocab = len(ext_tok)
                    vocab_growth = new_vocab - original_vocab
                    min_vocab_growth = int(ctx.tokenizer_extension_cfg.get("min_vocab_growth", 1))
                    if vocab_growth < min_vocab_growth:
                        fallback_tokens = ctx.tokenizer_extension_cfg.get("fallback_added_tokens", [])
                        fallback_values = [str(item) for item in fallback_tokens if str(item).strip()]
                        if fallback_values:
                            added = ext_tok.add_tokens(fallback_values)
                            if added > 0:
                                ext_tok.save_pretrained(ctx.extended_tokenizer_dir)
                                new_vocab = len(ext_tok)
                                vocab_growth = new_vocab - original_vocab
                    if vocab_growth < min_vocab_growth:
                        raise SystemExit(
                            f"Tokenizer extension vocab growth check failed: growth={vocab_growth}, required>={min_vocab_growth}"
                        )
                    expected_vocab_growth = ctx.tokenizer_extension_cfg.get("expected_vocab_growth")
                    if expected_vocab_growth is not None and vocab_growth != int(expected_vocab_growth):
                        raise SystemExit(
                            "Tokenizer extension exact vocab growth check failed: "
                            f"growth={vocab_growth}, expected={int(expected_vocab_growth)}"
                        )
                    extension_manifest = {
                        "base_tokenizer": local_base_tokenizer,
                        "multilingual_tokenizer_path": used_multilingual_tokenizer,
                        "prepared_manifest": str(ctx.extension_prepared_manifest.resolve()),
                        "pipeline_output_dir": str(ctx.extension_pipeline_root.resolve()),
                        "extended_tokenizer_dir": str(ctx.extended_tokenizer_dir.resolve()),
                        "original_vocab_size": original_vocab,
                        "new_vocab_size": new_vocab,
                        "vocab_growth": vocab_growth,
                        "allocation_csv": str((ctx.extension_pipeline_root / "allocation.csv").resolve()),
                        "base_metrics_csv": str((ctx.extension_pipeline_root / "base_metrics.csv").resolve()),
                        "extended_metrics_csv": str((ctx.extension_pipeline_root / "extended_metrics.csv").resolve()),
                        "comparison_csv": str((ctx.extension_pipeline_root / "comparison.csv").resolve()),
                    }
                    _dump_json(ctx.extension_manifest_path, extension_manifest)
                    _save_stage_state(ctx.state_path, "tokenizer_extend", signature, [str(path) for path in expected])
                    check_tokenizer_extend(ctx.extension_manifest_path, ctx.extended_tokenizer_dir, dry_run=ctx.args.dry_run)
                    ctx.logger.info("[OK] extension_manifest=%s", ctx.extension_manifest_path)
