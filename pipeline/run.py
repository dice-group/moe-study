#!/usr/bin/env python3
"""Resumable multilingual proof pipeline entrypoint."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .runtime import (
    _abs,
    _compute_pipeline_code_signature,
    _load_json,
    _load_pipeline_config,
    _model_slug,
    _resolve_profile,
    _resolve_profile_languages,
    _sanitize_slug,
    build_executor,
)
from .context import PipelineContext
from .data_stages import DATA_STAGE_HANDLERS
from .model_eval_stages import MODEL_STAGE_HANDLERS


STAGES = [
    "sample",
    "tokenizer_extend",
    "embedding_init",
    "tokenize",
    "audit",
    "train",
    "embedding_verify",
    "benchmark",
    "baseline_eval",
    "summary",
    "report",
]


def _configure_pipeline_logger(logs_root: Path, run_timestamp: str) -> logging.Logger:
    logger = logging.getLogger("proof_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    run_log_path = logs_root / f"pipeline_run_{run_timestamp}.log"
    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multilingual proof pipeline.")
    parser.add_argument("--profile", default="local_mini", help="Profile key in proof config.")
    parser.add_argument(
        "--config",
        default="configs/pipeline",
        help="Pipeline config JSON, config-bundle pointer JSON, or config bundle directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit.")
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=STAGES,
        help="Force rerun of a stage even if cached.",
    )
    parser.add_argument("--start-at", choices=STAGES, default="sample")
    parser.add_argument("--stop-after", choices=STAGES, default="summary")
    return parser.parse_args()


def _build_context(args: argparse.Namespace) -> PipelineContext:
    repo_root = Path(__file__).resolve().parents[1]
    config = _load_pipeline_config(_abs(repo_root, args.config))
    profile = _resolve_profile(config, args.profile)
    force_stages = set(args.force_stage)

    stage_start_idx = STAGES.index(args.start_at)
    stage_stop_idx = STAGES.index(args.stop_after)
    if stage_stop_idx < stage_start_idx:
        raise SystemExit("--stop-after must come after --start-at")
    active_stages = STAGES[stage_start_idx : stage_stop_idx + 1]

    output_root = _abs(repo_root, profile.get("output_root", f"outputs/proof_runs/{args.profile}"))
    state_path = output_root / "state" / "pipeline_state.json"
    output_root.mkdir(parents=True, exist_ok=True)

    language_map_path = _abs(repo_root, profile["language_map_path"])
    sample_cfg = profile.get("sample", {})
    tokenize_cfg = profile.get("tokenization", {})
    audit_cfg = profile.get("audit", {})
    train_cfg = profile.get("training", {})
    tokenizer_extension_cfg = profile.get("tokenizer_extension", {})
    embedding_init_cfg = profile.get("embedding_init", {})
    embedding_training_cfg = profile.get("embedding_training", {})
    summary_cfg = profile.get("summary", {})
    benchmark_cfg = profile.get("benchmark", {})
    baseline_eval_cfg = profile.get("baseline_eval", {})
    report_cfg = profile.get("report", {})
    wandb_cfg = profile.get("wandb", {})
    run_naming_cfg = profile.get("run_naming", {})
    run_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'LLaMA-Factory' / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PROOF_PIPELINE_RUN_TIMESTAMP"] = run_timestamp
    env["PROOF_PIPELINE_CODE_SIG"] = _compute_pipeline_code_signature(repo_root)
    os.environ["PROOF_PIPELINE_CODE_SIG"] = env["PROOF_PIPELINE_CODE_SIG"]
    env.setdefault("WANDB_MODE", str(profile.get("wandb_mode", "disabled")))
    if bool(wandb_cfg.get("enabled", False)):
        project = str(wandb_cfg.get("project", "")).strip()
        if project:
            env["WANDB_PROJECT"] = project
        entity = str(wandb_cfg.get("entity", "")).strip()
        if entity:
            env["WANDB_ENTITY"] = entity
        env["WANDB_MODE"] = str(wandb_cfg.get("mode", env.get("WANDB_MODE", "online")))

    sampled_dir = _abs(repo_root, sample_cfg.get("output_dir", str(output_root / "sampled")))
    sampled_manifest = sampled_dir / "manifest.json"
    shard_subdir = str(sample_cfg.get("shard_subdir", "")).strip().strip("/")
    sample_shard_dir = sampled_dir if not shard_subdir else (sampled_dir / shard_subdir)
    tokenized_dir = _abs(repo_root, tokenize_cfg.get("output_dir", str(output_root / "tokenized")))
    train_root = _abs(repo_root, train_cfg.get("output_dir", str(output_root / "runs")))
    summary_path = _abs(repo_root, summary_cfg.get("summary_csv", str(output_root / "summary.csv")))
    gate_path = _abs(repo_root, summary_cfg.get("gate_json", str(output_root / "gate.json")))
    benchmark_root = _abs(repo_root, benchmark_cfg.get("output_dir", str(output_root / "benchmarks")))
    baseline_cache_root = _abs(repo_root, baseline_eval_cfg.get("output_root", "outputs/baselines"))
    report_xlsx = _abs(repo_root, report_cfg.get("output_xlsx", str(output_root / "report.xlsx")))
    lpr_contract_path = output_root / "audit" / "lpr_contract.json"

    extension_root = _abs(repo_root, tokenizer_extension_cfg.get("output_dir", str(output_root / "tokenizer_extension")))
    extension_pipeline_root = _abs(
        repo_root,
        tokenizer_extension_cfg.get("pipeline_output_dir", str(extension_root / "pipeline")),
    )
    extension_prepared_dir = _abs(
        repo_root,
        tokenizer_extension_cfg.get("prepared_data_dir", str(extension_root / "prepared_shards")),
    )
    extension_prepared_manifest = extension_root / "prepared_manifest.json"
    extension_manifest_path = extension_root / "extension_manifest.json"
    extended_tokenizer_dir = _abs(
        repo_root,
        tokenizer_extension_cfg.get("extended_tokenizer_dir", str(extension_pipeline_root / "extended_tokenizer")),
    )

    embedding_init_output_dir = _abs(
        repo_root,
        embedding_init_cfg.get("output_dir", str(extension_root / "initialized_model")),
    )
    embedding_init_manifest_path = extension_root / "embedding_init_manifest.json"
    embedding_verify_root = output_root / "audit" / "embedding_verify"
    embedding_verify_summary_path = output_root / "audit" / "embedding_verify_summary.json"
    lineage_manifest_path = output_root / "lineage" / "tokenizer_embedding_lineage.json"

    logs_root = _abs(repo_root, profile.get("logs_dir", str(output_root / "logs")))
    logs_root.mkdir(parents=True, exist_ok=True)

    naming_languages = _resolve_profile_languages(
        sample_cfg=sample_cfg,
        sample_manifest_path=sampled_manifest,
        sampled_dir=sampled_dir,
    )
    lang_count = (
        len(naming_languages)
        if naming_languages
        else int(run_naming_cfg.get("language_count_hint", len(sample_cfg.get("languages", []))))
    )
    model_slug = _model_slug(str(train_cfg.get("model_name_or_path", "model")))
    exp_tag = _sanitize_slug(str(run_naming_cfg.get("experiment_tag", "proof")))
    base_run_label = f"{model_slug}-{lang_count}langs-{exp_tag}-{run_timestamp}"
    logger = _configure_pipeline_logger(logs_root, run_timestamp)

    return PipelineContext(
        args=args,
        repo_root=repo_root,
        profile=profile,
        force_stages=force_stages,
        active_stages=active_stages,
        output_root=output_root,
        state_path=state_path,
        language_map_path=language_map_path,
        sample_cfg=sample_cfg,
        tokenize_cfg=tokenize_cfg,
        audit_cfg=audit_cfg,
        train_cfg=train_cfg,
        tokenizer_extension_cfg=tokenizer_extension_cfg,
        embedding_init_cfg=embedding_init_cfg,
        embedding_training_cfg=embedding_training_cfg,
        summary_cfg=summary_cfg,
        benchmark_cfg=benchmark_cfg,
        baseline_eval_cfg=baseline_eval_cfg,
        report_cfg=report_cfg,
        wandb_cfg=wandb_cfg,
        env=env,
        logger=logger,
        sampled_dir=sampled_dir,
        sampled_manifest=sampled_manifest,
        shard_subdir=shard_subdir,
        sample_shard_dir=sample_shard_dir,
        tokenized_dir=tokenized_dir,
        train_root=train_root,
        summary_path=summary_path,
        gate_path=gate_path,
        benchmark_root=benchmark_root,
        baseline_cache_root=baseline_cache_root,
        report_xlsx=report_xlsx,
        lpr_contract_path=lpr_contract_path,
        extension_root=extension_root,
        extension_pipeline_root=extension_pipeline_root,
        extension_prepared_dir=extension_prepared_dir,
        extension_prepared_manifest=extension_prepared_manifest,
        extension_manifest_path=extension_manifest_path,
        extended_tokenizer_dir=extended_tokenizer_dir,
        embedding_init_output_dir=embedding_init_output_dir,
        embedding_init_manifest_path=embedding_init_manifest_path,
        embedding_verify_root=embedding_verify_root,
        embedding_verify_summary_path=embedding_verify_summary_path,
        executor=build_executor(profile),
        lineage_manifest_path=lineage_manifest_path,
        logs_root=logs_root,
        run_naming_cfg=run_naming_cfg,
        run_timestamp=run_timestamp,
        naming_languages=naming_languages,
        lang_count=lang_count,
        model_slug=model_slug,
        exp_tag=exp_tag,
        base_run_label=base_run_label,
    )


def main() -> int:
    args = _parse_args()
    if args.list_profiles:
        repo_root = Path(__file__).resolve().parents[1]
        config = _load_pipeline_config(_abs(repo_root, args.config))
        profiles = sorted(config.get("profiles", {}).keys())
        print(f"Available profiles ({len(profiles)}):")
        for name in profiles:
            raw = config["profiles"][name]
            extends = raw.get("extends", "")
            suffix = f"  (extends: {extends})" if extends else ""
            print(f"  {name}{suffix}")
        return 0
    ctx = _build_context(args)
    stage_handlers = {**DATA_STAGE_HANDLERS, **MODEL_STAGE_HANDLERS}
    run_started_perf = time.perf_counter()
    run_started_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_events_path = ctx.logs_root / f"pipeline_run_{ctx.run_timestamp}.jsonl"
    run_summary_path = ctx.logs_root / f"pipeline_run_{ctx.run_timestamp}_summary.json"
    stage_results: list[dict[str, Any]] = []

    def _emit(event: str, *, stage: str | None = None, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "profile": ctx.args.profile,
            "event": event,
        }
        if stage is not None:
            payload["stage"] = stage
        payload.update(fields)
        run_events_path.parent.mkdir(parents=True, exist_ok=True)
        with run_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    _emit(
        "run_start",
        dry_run=bool(ctx.args.dry_run),
        start_at=ctx.args.start_at,
        stop_after=ctx.args.stop_after,
        active_stages=ctx.active_stages,
    )

    for stage in ctx.active_stages:
        handler = stage_handlers.get(stage)
        if handler is None:
            raise SystemExit(f"No stage handler configured for: {stage}")
        stage_started_perf = time.perf_counter()
        ctx.logger.info("[STAGE_START] %s", stage)
        _emit("stage_start", stage=stage)
        try:
            handler(ctx)
        except Exception as exc:
            duration_s = time.perf_counter() - stage_started_perf
            stage_results.append({"stage": stage, "status": "failed", "duration_seconds": round(duration_s, 3)})
            ctx.logger.error(
                "[STAGE_FAIL] %s duration_seconds=%.3f error=%s: %s",
                stage,
                duration_s,
                type(exc).__name__,
                exc,
            )
            _emit(
                "stage_fail",
                stage=stage,
                duration_seconds=round(duration_s, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        duration_s = time.perf_counter() - stage_started_perf
        stage_results.append({"stage": stage, "status": "ok", "duration_seconds": round(duration_s, 3)})
        ctx.logger.info("[STAGE_DONE] %s duration_seconds=%.3f", stage, duration_s)
        _emit("stage_done", stage=stage, duration_seconds=round(duration_s, 3))

    total_duration_s = time.perf_counter() - run_started_perf
    run_summary: dict[str, Any] = {
        "profile": ctx.args.profile,
        "run_started_utc": run_started_utc,
        "run_finished_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dry_run": bool(ctx.args.dry_run),
        "start_at": ctx.args.start_at,
        "stop_after": ctx.args.stop_after,
        "total_duration_seconds": round(total_duration_s, 3),
        "stages": stage_results,
        "events_jsonl": str(run_events_path.resolve()),
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")
    _emit("run_done", total_duration_seconds=round(total_duration_s, 3), summary_json=str(run_summary_path.resolve()))
    ctx.logger.info("[RUN_SUMMARY] %s", run_summary_path)
    ctx.logger.info("[DONE] profile=%s", ctx.args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
