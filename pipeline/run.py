#!/usr/bin/env python3
"""Resumable multilingual proof pipeline entrypoint."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
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
from .executor import SlurmExecutor
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


def _parse_run_ids(raw_values: list[str]) -> list[str]:
    """ This enables to run different approaches form a bigger configu individually by using its run id (e.g. hmora)"""
    run_ids: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for item in str(raw).split(","):
            run_id = item.strip()
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            run_ids.append(run_id)
    return run_ids


def _stage_profile_cfg(profile: dict[str, Any], stage: str) -> dict[str, Any]:
    """Map pipeline stage names to their profile config blocks."""
    stage_to_profile_key = {
        "sample": "sample",
        "tokenizer_extend": "tokenizer_extension",
        "embedding_init": "embedding_init",
        "tokenize": "tokenization",
        "audit": "audit",
        "train": "training",
        "embedding_verify": "embedding_training",
        "benchmark": "benchmark",
        "baseline_eval": "baseline_eval",
        "summary": "summary",
        "report": "report",
    }
    key = stage_to_profile_key.get(stage, "")
    cfg = profile.get(key, {})
    return cfg if isinstance(cfg, dict) else {}


def _resolve_stage_cfg(profile: dict[str, Any], stage: str) -> dict[str, Any]:
    cfg = _stage_profile_cfg(profile, stage)
    stage_executors = profile.get("stage_executors", {})
    if isinstance(stage_executors, dict):
        stage_exec = stage_executors.get(stage)
        if isinstance(stage_exec, dict):
            merged = dict(cfg) if isinstance(cfg, dict) else {}
            exec_override = dict(stage_exec)
            exec_override.update(merged.get("executor", {}) if isinstance(merged.get("executor"), dict) else {})
            merged["executor"] = exec_override
            return merged
    return cfg


def _build_language_map_from_csv(
    *,
    csv_path: Path,
    output_path: Path,
    subset_column: str = "subset",
    family_column: str = "family",
    split_column: str = "split",
    split_value: str = "train",
    extra_languages: dict[str, str] | None = None,
) -> Path:
    rows: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"language_map_csv is missing a header: {csv_path}")
        if subset_column not in reader.fieldnames:
            raise SystemExit(f"language_map_csv missing column '{subset_column}'. Found: {reader.fieldnames}")
        if family_column not in reader.fieldnames:
            raise SystemExit(f"language_map_csv missing column '{family_column}'. Found: {reader.fieldnames}")
        if split_value and split_column not in reader.fieldnames:
            raise SystemExit(f"language_map_csv missing split column '{split_column}'. Found: {reader.fieldnames}")

        for row in reader:
            if split_value and str(row.get(split_column, "")).strip() != split_value:
                continue
            subset = str(row.get(subset_column, "")).strip()
            if not subset:
                continue
            family = str(row.get(family_column, "")).strip() or "unknown"
            rows.setdefault(subset, family)

    if extra_languages:
        for lang, family in extra_languages.items():
            if not str(lang).strip():
                continue
            rows.setdefault(str(lang).strip(), str(family).strip() or "unknown")

    if not rows:
        raise SystemExit(f"No languages resolved from language_map_csv: {csv_path}")

    grouped: dict[str, list[str]] = {}
    for lang, family in rows.items():
        grouped.setdefault(family, []).append(lang)

    payload: dict[str, dict[str, Any]] = {}
    for idx, family in enumerate(sorted(grouped.keys())):
        payload[str(idx)] = {
            "group": family,
            "languages": sorted(grouped[family]),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


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
    parser.add_argument("--config", default="configs/pipeline", help="Pipeline config JSON, config-bundle pointer JSON, or config bundle directory.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit.")
    parser.add_argument("--force-stage", action="append", default=[], choices=STAGES, help="Force rerun of a stage even if cached (e.g tokenization)")
    parser.add_argument("--start-at", choices=STAGES, default="sample")
    parser.add_argument("--stop-after", choices=STAGES, default="summary")
    parser.add_argument("--run-id", action="append", default=[], help="Run only selected training run ids (e.g hmora, cola)")
    parser.add_argument("--submit-chain", action="store_true", help="Submit stage-wise Slurm jobs with dependencies.")
    parser.add_argument("--executor-type", choices=["local", "slurm"], default="", help="Override executor type from profile.")
    parser.add_argument("--cuda-visible-devices", default="", help="Override CUDA_VISIBLE_DEVICES for pipeline subprocesses.",)
    parser.add_argument("--pycharm-debug", action="store_true", help="Attach train subprocesses to a PyCharm debug server.",)
    parser.add_argument("--pycharm-debug-host", default="127.0.0.1")
    parser.add_argument("--pycharm-debug-port", type=int, default=12345)
    parser.add_argument("--pycharm-debug-suspend", action="store_true", help="Suspend training subprocess at debugger attach point.")
    return parser.parse_args()


def _build_context(args: argparse.Namespace) -> PipelineContext:
    # main paths / configs
    repo_root = Path(__file__).resolve().parents[1]
    config = _load_pipeline_config(_abs(repo_root, args.config))
    profile = dict(_resolve_profile(config, args.profile))
    run_ids = _parse_run_ids(args.run_id)
    args.run_id = run_ids
    force_stages = set(args.force_stage)

    executor_cfg_raw = profile.get("executor", {})
    executor_cfg = dict(executor_cfg_raw) if isinstance(executor_cfg_raw, dict) else {}
    executor_overridden = False
    if args.executor_type:
        executor_cfg["type"] = args.executor_type
        executor_overridden = True
    if str(args.cuda_visible_devices).strip():
        executor_cfg["cuda_visible_devices"] = str(args.cuda_visible_devices).strip()
        executor_overridden = True
    if executor_overridden:
        profile["executor"] = executor_cfg

    stage_start_idx = STAGES.index(args.start_at)
    stage_stop_idx = STAGES.index(args.stop_after)
    if stage_stop_idx < stage_start_idx:
        raise SystemExit("--stop-after must come after --start-at")
    active_stages = STAGES[stage_start_idx: stage_stop_idx + 1]

    output_root = _abs(repo_root, profile.get("output_root", f"outputs/proof_runs/{args.profile}"))
    state_path = output_root / "state" / "pipeline_state.json"
    output_root.mkdir(parents=True, exist_ok=True)

    # stage wise config resolution
    language_map_csv = profile.get("language_map_csv")
    if isinstance(language_map_csv, str) and language_map_csv.strip():
        csv_path = _abs(repo_root, language_map_csv)
        extra_raw = profile.get("language_map_extra_languages", {})
        extra: dict[str, str] = {}
        if isinstance(extra_raw, dict):
            extra = {str(k): str(v) for k, v in extra_raw.items() if str(k).strip()}
        elif isinstance(extra_raw, list):
            fallback_family = str(profile.get("language_map_extra_family", "unknown")).strip() or "unknown"
            for item in extra_raw:
                if str(item).strip():
                    extra[str(item).strip()] = fallback_family
        output_map_path = output_root / "language_map_from_csv.json"
        language_map_path = _build_language_map_from_csv(
            csv_path=csv_path,
            output_path=output_map_path,
            subset_column=str(profile.get("language_map_csv_subset_column", "subset")).strip() or "subset",
            family_column=str(profile.get("language_map_csv_family_column", "family")).strip() or "family",
            split_column=str(profile.get("language_map_csv_split_column", "split")).strip() or "split",
            split_value=str(profile.get("language_map_csv_split", "train")).strip(),
            extra_languages=extra,
        )
    else:
        language_map_path = _abs(repo_root, profile["language_map_path"])
    sample_cfg = profile.get("sample", {})
    tokenize_cfg = profile.get("tokenization", {})
    audit_cfg = profile.get("audit", {})
    train_cfg = profile.get("training", {})
    if run_ids:
        runs = train_cfg.get("runs", []) if isinstance(train_cfg, dict) else []
        if not isinstance(runs, list):
            raise SystemExit("profile.training.runs must be a list to use --run-id")
        selected_run_ids = set(run_ids)
        selected = [run for run in runs if str(run.get("id", "")) in selected_run_ids]
        selected_ids = {str(run.get("id", "")) for run in selected}
        missing = [run_id for run_id in run_ids if run_id not in selected_ids]
        if missing:
            raise SystemExit(f"Unknown --run-id values for profile '{args.profile}': {missing}")
        train_cfg = dict(train_cfg)
        train_cfg["runs"] = selected
        profile["training"] = train_cfg
    if args.pycharm_debug:
        train_cfg = dict(train_cfg)
        debugger_cfg_raw = train_cfg.get("debugger", {})
        debugger_cfg = dict(debugger_cfg_raw) if isinstance(debugger_cfg_raw, dict) else {}
        debugger_cfg["enabled"] = True
        debugger_cfg["kind"] = "pycharm"
        debugger_cfg["host"] = str(args.pycharm_debug_host).strip() or "127.0.0.1"
        debugger_cfg["port"] = int(args.pycharm_debug_port)
        debugger_cfg["suspend"] = bool(args.pycharm_debug_suspend)
        train_cfg["debugger"] = debugger_cfg
        profile["training"] = train_cfg
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
    # Preserve virtualenv interpreter paths; resolving symlinks here can strip
    # the venv context and cause stage subprocesses to miss installed packages.
    python_exec = Path(sys.executable)
    env["PYTHON_EXECUTABLE"] = str(python_exec)
    env["PATH"] = f"{python_exec.parent}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'LLaMA-Factory' / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PROOF_PIPELINE_RUN_TIMESTAMP"] = run_timestamp
    env["PROOF_PIPELINE_CODE_SIG"] = _compute_pipeline_code_signature(repo_root)
    os.environ["PROOF_PIPELINE_CODE_SIG"] = env["PROOF_PIPELINE_CODE_SIG"]
    executor_cfg = profile.get("executor", {})
    if isinstance(executor_cfg, dict):
        cuda_visible_devices = str(executor_cfg.get("cuda_visible_devices", "")).strip()
        if cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if run_ids:
        env["PROOF_PIPELINE_SELECTED_RUN_IDS"] = ",".join(run_ids)
    if args.pycharm_debug:
        env["PROOF_PIPELINE_PYCHARM_DEBUG"] = "1"
    env.setdefault("WANDB_MODE", str(profile.get("wandb_mode", "disabled")))
    if bool(wandb_cfg.get("enabled", False)):
        project = str(wandb_cfg.get("project", "")).strip()
        if project:
            env["WANDB_PROJECT"] = project
        entity = str(wandb_cfg.get("entity", "")).strip()
        if entity:
            env["WANDB_ENTITY"] = entity
        env["WANDB_MODE"] = str(wandb_cfg.get("mode", env.get("WANDB_MODE", "online")))

    # paths and dirs
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
    extension_pipeline_root = _abs(repo_root, tokenizer_extension_cfg.get("pipeline_output_dir", str(extension_root / "pipeline")))
    extension_prepared_dir = _abs(repo_root, tokenizer_extension_cfg.get("prepared_data_dir", str(extension_root / "prepared_shards")))
    extension_prepared_manifest = extension_root / "prepared_manifest.json"
    extension_manifest_path = extension_root / "extension_manifest.json"
    extended_tokenizer_dir = _abs(repo_root, tokenizer_extension_cfg.get("extended_tokenizer_dir", str(extension_pipeline_root / "extended_tokenizer")))

    embedding_init_output_dir = _abs(repo_root, embedding_init_cfg.get("output_dir", str(extension_root / "initialized_model")),)
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


def _submit_chain(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config = _load_pipeline_config(_abs(repo_root, args.config))
    profile = dict(_resolve_profile(config, args.profile))
    run_ids = _parse_run_ids(args.run_id)

    executor_cfg_raw = profile.get("executor", {})
    executor_cfg = dict(executor_cfg_raw) if isinstance(executor_cfg_raw, dict) else {}
    executor_overridden = False
    if args.executor_type:
        executor_cfg["type"] = args.executor_type
        executor_overridden = True
    if str(args.cuda_visible_devices).strip():
        executor_cfg["cuda_visible_devices"] = str(args.cuda_visible_devices).strip()
        executor_overridden = True
    if executor_overridden:
        profile["executor"] = executor_cfg

    stage_start_idx = STAGES.index(args.start_at)
    stage_stop_idx = STAGES.index(args.stop_after)
    if stage_stop_idx < stage_start_idx:
        raise SystemExit("--stop-after must come after --start-at")
    active_stages = STAGES[stage_start_idx: stage_stop_idx + 1]

    output_root = _abs(repo_root, profile.get("output_root", f"outputs/proof_runs/{args.profile}"))
    logs_root = _abs(repo_root, profile.get("logs_dir", str(output_root / "logs")))
    logs_root.mkdir(parents=True, exist_ok=True)
    chain_root = logs_root / "chain"
    chain_logs_root = chain_root / "logs"
    chain_sbatch_root = chain_root / "sbatch"
    chain_slurm_root = chain_root / "slurm"
    chain_logs_root.mkdir(parents=True, exist_ok=True)
    chain_sbatch_root.mkdir(parents=True, exist_ok=True)
    chain_slurm_root.mkdir(parents=True, exist_ok=True)

    sample_cfg = profile.get("sample", {}) if isinstance(profile.get("sample", {}), dict) else {}
    sampled_dir = _abs(repo_root, sample_cfg.get("output_dir", str(output_root / "sampled")))
    sampled_manifest = sampled_dir / "sampling_manifest.json"
    naming_languages = _resolve_profile_languages(
        sample_cfg=sample_cfg,
        sample_manifest_path=sampled_manifest,
        sampled_dir=sampled_dir,
    )
    run_naming_cfg = profile.get("run_naming", {}) if isinstance(profile.get("run_naming", {}), dict) else {}
    lang_count = (
        len(naming_languages)
        if naming_languages
        else int(run_naming_cfg.get("language_count_hint", len(sample_cfg.get("languages", [])) or 0) or 0)
    )
    lang_tag = f"{max(1, lang_count)}l" if lang_count else "xl"

    stage_short = {
        "sample": "samp",
        "tokenizer_extend": "tokx",
        "embedding_init": "embi",
        "tokenize": "tok",
        "audit": "audit",
        "embedding_verify": "embv",
        "summary": "sum",
        "report": "rpt",
        "train": "train",
        "benchmark": "bench",
        "baseline_eval": "beval",
    }

    chain_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dependency_ids: list[str] = []
    submitted: list[str] = []
    last_stage_job_ids_by_run: dict[str, str] = {}

    for stage in active_stages:
        stage_cfg = _resolve_stage_cfg(profile, stage)
        if isinstance(stage_cfg, dict) and stage_cfg.get("enabled") is False:
            continue
        executor = build_executor(profile, stage_cfg)
        if not isinstance(executor, SlurmExecutor):
            raise SystemExit(f"submit-chain requires Slurm executor for stage '{stage}'")
        stage_tag = stage_short.get(stage, stage)
        dependency = f"afterok:{':'.join(dependency_ids)}" if dependency_ids else None
        stage_job_ids: list[str] = []

        if stage in {"train", "benchmark"} and len(run_ids) > 1:
            for run_id in run_ids:
                cmd = [
                    sys.executable,
                    "-m",
                    "pipeline.run",
                    "--config",
                    args.config,
                    "--profile",
                    args.profile,
                    "--start-at",
                    stage,
                    "--stop-after",
                    stage,
                    "--force-stage",
                    stage,
                    "--executor-type",
                    "local",
                    "--run-id",
                    run_id,
                ]
                if args.dry_run:
                    cmd.append("--dry-run")
                log_path = chain_logs_root / f"chain_{stage}.{run_id}.{chain_timestamp}.log"
                slurm_output = chain_slurm_root / f"chain_{stage}.{run_id}.{chain_timestamp}.slurm.log"
                job_name = f"{stage_tag}{lang_tag}_{run_id}"
                job_dependency = dependency
                if stage == "benchmark":
                    prior_job = last_stage_job_ids_by_run.get(run_id)
                    if prior_job:
                        job_dependency = f"afterok:{prior_job}"
                job_id = executor.submit(
                    cmd,
                    os.environ.copy(),
                    args.dry_run,
                    log_path,
                    dependency=job_dependency,
                    job_name=job_name,
                    script_dir=chain_sbatch_root,
                    slurm_output=slurm_output,
                )
                if job_id:
                    stage_job_ids.append(job_id)
                    submitted.append(job_id)
                    last_stage_job_ids_by_run[run_id] = job_id
        else:
            cmd = [
                sys.executable,
                "-m",
                "pipeline.run",
                "--config",
                args.config,
                "--profile",
                args.profile,
                "--start-at",
                stage,
                "--stop-after",
                stage,
                "--force-stage",
                stage,
                "--executor-type",
                "local",
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            for run_id in run_ids:
                cmd.extend(["--run-id", run_id])
            log_path = chain_logs_root / f"chain_{stage}.{chain_timestamp}.log"
            slurm_output = chain_slurm_root / f"chain_{stage}.{chain_timestamp}.slurm.log"
            if stage in {"train", "benchmark", "baseline_eval"}:
                if len(run_ids) == 1:
                    approach = run_ids[0]
                elif run_ids:
                    approach = "multi"
                else:
                    approach = "all"
            else:
                approach = "all"
            job_name = f"{stage_tag}{lang_tag}_{approach}"
            job_id = executor.submit(
                cmd,
                os.environ.copy(),
                args.dry_run,
                log_path,
                dependency=dependency,
                job_name=job_name,
                script_dir=chain_sbatch_root,
                slurm_output=slurm_output,
            )
            if job_id:
                stage_job_ids.append(job_id)
                submitted.append(job_id)

        dependency_ids = stage_job_ids if stage_job_ids else dependency_ids

    if submitted:
        print(f"Submitted {len(submitted)} job(s): {' -> '.join(submitted)}")
    else:
        print("No stages submitted (all disabled or dry-run).")
    return 0


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
    if args.submit_chain:
        return _submit_chain(args)
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
        run_ids=ctx.args.run_id,
    )

    for stage in ctx.active_stages:
        handler = stage_handlers.get(stage)
        if handler is None:
            raise SystemExit(f"No stage handler configured for: {stage}")
        stage_cfg = _resolve_stage_cfg(ctx.profile, stage)
        ctx.executor = build_executor(ctx.profile, stage_cfg)
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
