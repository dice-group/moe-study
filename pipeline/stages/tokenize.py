from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .runtime import (
    _load_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_log_path,
    _stage_signature,
)
from ..checks import check_tokenize
from ..context import PipelineContext
from ..executor import SlurmExecutor, build_executor


def _resolve_languages_for_tokenize(ctx: PipelineContext) -> list[str]:
    token_cfg = ctx.tokenize_cfg
    english_key = str(token_cfg.get("language_rates_english_key", "english")).strip() or "english"
    include_english = bool(token_cfg.get("include_english_key", True))
    explicit = token_cfg.get("languages")
    if isinstance(explicit, list) and explicit:
        base = [str(item).strip() for item in explicit if str(item).strip()]
        extras = _resolve_language_extras(token_cfg)        
        if include_english:
            return sorted(set(base + extras + [english_key]))  # preserve legacy behavior by default
        return sorted(set(base + extras))

    csv_path = str(token_cfg.get("languages_csv", "")).strip()
    if csv_path:
        column = str(token_cfg.get("languages_csv_column", "subset")).strip() or "subset"
        split_value = str(token_cfg.get("languages_csv_split", "train")).strip()
        split_column = str(token_cfg.get("languages_csv_split_column", "split")).strip() or "split"
        languages: set[str] = set()
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV is missing a header: {csv_path}")
            if column not in reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV missing column '{column}'. Found: {reader.fieldnames}")
            if split_value and split_column not in reader.fieldnames:
                raise SystemExit(f"Tokenize languages CSV missing split column '{split_column}'. Found: {reader.fieldnames}")
            for row in reader:
                if split_value:
                    if str(row.get(split_column, "")).strip() != split_value:
                        continue
                value = str(row.get(column, "")).strip()
                if value:
                    languages.add(value)
        if not languages:
            raise SystemExit(f"No languages resolved from {csv_path} (column={column}, split={split_value})")
        extras = _resolve_language_extras(token_cfg)
        resolved = set(languages).union(extras)
        if include_english:
            resolved.add(english_key)
        return sorted(resolved)

    explicit = ctx.sample_cfg.get("languages")
    if isinstance(explicit, list) and explicit:
        base = [str(item).strip() for item in explicit if str(item).strip()]
        extras = _resolve_language_extras(token_cfg)
        if include_english:
            return sorted(set(base + extras + [english_key]))
        return sorted(set(base + extras))

    return []


def _resolve_language_extras(token_cfg: dict) -> list[str]:
    extras = token_cfg.get("languages_extra")
    if extras is None:
        return []
    if isinstance(extras, str):
        return [item.strip() for item in extras.split(",") if item.strip()]
    if isinstance(extras, list):
        return [str(item).strip() for item in extras if str(item).strip()]
    return []


def _plan_complete(plan_dir: Path, world_size: int) -> bool:
    return all((plan_dir / f"rank_{rank:05d}.jsonl").exists() for rank in range(world_size))


def _default_shard_doc_counts_tsv(ctx: PipelineContext) -> Path:
    return ctx.repo_root / "data_prep" / "merlin_data_prep" / "distributed_data_processor" / "configs" / "shard_doc_counts.tsv"

def _resolve_shard_doc_counts_tsv(ctx: PipelineContext, *, languages: list[str]) -> Path:
    token_cfg = ctx.tokenize_cfg

    if bool(token_cfg.get("shard_counts_from_sampled", False)):
        output_tsv = ctx.sampled_dir / "sampled_shard_doc_counts.tsv"
        cmd = [
            "python3",
            str(ctx.repo_root / "pipeline_tools" / "build_sampled_shard_doc_counts_tsv.py"),
            "--shard_dir",
            str(ctx.sample_shard_dir),
            "--output_tsv",
            str(output_tsv),
            "--languages",
            ",".join(languages),
        ]
        _run_stage_cmd(
            cmd=cmd,
            env=ctx.env,
            dry_run=ctx.args.dry_run,
            logs_root=ctx.logs_root,
            stage="tokenize_shard_counts",
            executor=ctx.executor,
        )
        return output_tsv

    override = token_cfg.get("shard_counts_tsv")
    if isinstance(override, str) and override.strip():
        return (ctx.repo_root / override.strip()).resolve()

    return _default_shard_doc_counts_tsv(ctx)


def _build_shard_plan_cmd(
    ctx: PipelineContext,
    *,
    shard_counts_tsv: Path,
    languages: list[str],
    world_size: int,
    rates_path: Path | None,
) -> list[str]:
    cmd = [
        "python3",
        str(ctx.repo_root / "data_prep" / "merlin_data_prep" / "distributed_data_processor" / "src" / "plan_tokenization_shards.py"),
        "--shard_counts_tsv",
        str(shard_counts_tsv),
        "--output_dir",
        str(ctx.tokenized_dir.parent / f"{ctx.tokenized_dir.name}_shard_plan"),
        "--languages",
        *languages,
        "--world_size",
        str(world_size),
        "--sample_fraction",
        str(float(ctx.tokenize_cfg.get("sample_fraction", 1.0))),
        "--eval_fraction",
        str(float(ctx.tokenize_cfg.get("eval_fraction", 0.02))),
    ]
    if rates_path is not None:
        cmd.extend(["--language_rates_json", str(rates_path)])
    return cmd


def _build_language_rates_for_tokenize(ctx: PipelineContext, languages: list[str], *, counts_tsv: Path) -> Path | None:
    token_cfg = ctx.tokenize_cfg
    english_key = str(token_cfg.get("language_rates_english_key", "english")).strip() or "english"
    if english_key not in languages:
        return None
    if not counts_tsv.exists():
        if ctx.args.dry_run:
            return None
        raise SystemExit(f"Missing shard doc counts TSV: {counts_tsv}")

    totals = {language: 0 for language in languages}
    with counts_tsv.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        if header != ["language", "shard", "doc_count"]:
            raise SystemExit(f"Unexpected shard counts header in {counts_tsv}: {header}")
        for line_no, line in enumerate(handle, start=2):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 3:
                raise SystemExit(f"Invalid shard counts row in {counts_tsv}:{line_no}")
            language, _shard, doc_count = parts
            if language not in totals:
                continue
            totals[language] += int(doc_count)

    missing = [language for language, total in totals.items() if total <= 0]
    if missing:
        raise SystemExit(f"Missing shard counts for languages: {missing}")

    base_rate = float(token_cfg.get("sample_fraction", 1.0))
    english_docs = float(totals[english_key])
    if english_docs <= 0:
        raise SystemExit(f"Non-positive shard count for english language '{english_key}': {counts_tsv}")

    other_targets: list[int] = []
    for language in languages:
        if language == english_key:
            continue
        total_docs = int(totals[language])
        target = int(round(total_docs * base_rate))
        if total_docs > 0 and base_rate > 0.0:
            target = max(1, target)
        other_targets.append(min(total_docs, target))

    if not other_targets:
        return None
    english_target = int(round(statistics.median(other_targets)))
    english_rate = english_target / english_docs
    english_rate = max(0.0, min(1.0, english_rate))

    rates = {english_key: english_rate}
    output_path = ctx.output_root / "tokenization_language_rates.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rates, indent=2) + "\n", encoding="utf-8")
    return output_path


def _submit_tokenize_slurm_array(
    ctx: PipelineContext,
    *,
    languages: list[str],
    rates_path: Path | None,
    tokenize_model_name: str,
    shard_counts_tsv: Path,
) -> None:
    token_cfg = ctx.tokenize_cfg
    executor = ctx.executor
    if not isinstance(executor, SlurmExecutor):
        stage_cfg = dict(token_cfg)
        stage_executors = ctx.profile.get("stage_executors", {})
        if isinstance(stage_executors, dict):
            stage_exec = stage_executors.get("tokenize")
            if isinstance(stage_exec, dict):
                merged_exec = dict(stage_exec)
                if isinstance(stage_cfg.get("executor"), dict):
                    merged_exec.update(stage_cfg["executor"])
                stage_cfg["executor"] = merged_exec
        profile_copy = dict(ctx.profile)
        profile_copy["executor"] = dict(ctx.profile.get("executor", {}))
        profile_copy["executor"]["type"] = "slurm"
        executor = build_executor(profile_copy, stage_cfg)
    if not isinstance(executor, SlurmExecutor):
        raise SystemExit("tokenization.slurm_array_enabled requires a Slurm executor")

    array_ranks = int(token_cfg.get("slurm_array_ranks", 0) or 0)
    if array_ranks <= 0:
        array_ranks = len(languages)
    if array_ranks <= 0:
        raise SystemExit("tokenization.slurm_array_ranks must be set when no languages list is available")

    rank_output_dir = ctx.tokenized_dir.parent / f"{ctx.tokenized_dir.name}_ranks"
    shard_plan_dir = ctx.tokenized_dir.parent / f"{ctx.tokenized_dir.name}_shard_plan"
    if not languages:
        raise SystemExit("Tokenization requires resolved languages for shard-plan creation.")
    if not shard_counts_tsv.exists():
        if ctx.args.dry_run:
            return
        raise SystemExit(f"Missing shard doc counts TSV: {shard_counts_tsv}")
    max_samples = int(token_cfg.get("max_samples", 0) or 0)
    cmd = [
        "python3",
        str(ctx.repo_root / "data_prep" / "merlin_data_prep" / "distributed_data_processor" / "src" / "tokenize_slurm.py"),
        "--shard_dir",
        str(ctx.sample_shard_dir),
        "--save_tokenized_data_dir",
        str(rank_output_dir),
        "--model_name",
        tokenize_model_name,
        "--num_proc",
        str(int(token_cfg.get("num_proc", 1))),
        "--eval_seed",
        str(int(token_cfg.get("eval_seed", 42))),
        "--sample_seed",
        str(int(token_cfg.get("sample_seed", 42))),
        "--max_length",
        str(int(token_cfg.get("max_length", 1024))),
        "--language-map",
        str(ctx.language_map_path),
        "--mask_pad_labels",
        "true" if bool(token_cfg.get("mask_pad_labels", True)) else "false",
        "--shard-plan-dir",
        str(shard_plan_dir),
    ]
    if max_samples > 0:
        cmd.extend(["--max_samples", str(max_samples)])

    lang_count = max(1, int(getattr(ctx, "lang_count", len(languages) or 0) or 0))
    array_spec = f"0-{array_ranks - 1}"
    shard_plan_logs_root = ctx.logs_root / "tokenize" / "shard_plan"
    array_logs_root = ctx.logs_root / "tokenize" / "array"
    merge_logs_root = ctx.logs_root / "tokenize" / "merge"
    shard_plan_logs_root.mkdir(parents=True, exist_ok=True)
    array_logs_root.mkdir(parents=True, exist_ok=True)
    merge_logs_root.mkdir(parents=True, exist_ok=True)

    shard_plan_job_id = None
    if not _plan_complete(shard_plan_dir, array_ranks):
        shard_plan_cmd = _build_shard_plan_cmd(
            ctx,
            shard_counts_tsv=shard_counts_tsv,
            languages=languages,
            world_size=array_ranks,
            rates_path=rates_path,
        )
        shard_plan_log = _stage_log_path(shard_plan_logs_root, ctx.env, "tokenize", "shard_plan")
        shard_plan_job_id = executor.submit(
            shard_plan_cmd,
            ctx.env,
            ctx.args.dry_run,
            shard_plan_log,
            job_name=f"toks{lang_count}l_all",
        )

    stage_log = _stage_log_path(array_logs_root, ctx.env, "tokenize", "array")
    slurm_output = array_logs_root / "tokenize_array_%A_%a.log"
    job_name = f"tok{lang_count}l_all"
    array_job_id = executor.submit_array(
        cmd,
        ctx.env,
        ctx.args.dry_run,
        stage_log,
        array=array_spec,
        dependency=f"afterok:{shard_plan_job_id}" if shard_plan_job_id else None,
        slurm_output=str(slurm_output),
        job_name=job_name,
    )
    if ctx.args.dry_run or not array_job_id:
        return

    merge_cmd = [
        "python3",
        str(ctx.repo_root / "data_prep" / "merlin_data_prep" / "distributed_data_processor" / "src" / "merge_tokenized_ranks.py"),
        "--tokenized_root",
        str(rank_output_dir),
        "--output_path",
        str(ctx.tokenized_dir),
        "--overwrite",
    ]
    merge_log = _stage_log_path(merge_logs_root, ctx.env, "tokenize", "merge")
    merge_job_name = f"tokm{lang_count}l_all"
    merge_job_id = executor.submit(
        merge_cmd,
        ctx.env,
        ctx.args.dry_run,
        merge_log,
        dependency=f"afterok:{array_job_id}",
        job_name=merge_job_name,
    )
    if merge_job_id:
        executor.wait_for_jobs([(merge_job_id, merge_log)], ctx.args.dry_run)


def _submit_tokenize_parallel_check(ctx: PipelineContext, *, languages: list[str]) -> bool:
    token_cfg = ctx.tokenize_cfg
    if not bool(token_cfg.get("check_parallel_enabled", False)):
        return False
    if ctx.args.dry_run:
        return False

    rank_dirs = sorted(
        p for p in ctx.tokenized_dir.iterdir() if p.is_dir() and p.name.startswith("rank_")
    )
    if not rank_dirs:
        return False

    executor = ctx.executor
    if not isinstance(executor, SlurmExecutor):
        stage_cfg = dict(token_cfg)
        stage_executors = ctx.profile.get("stage_executors", {})
        if isinstance(stage_executors, dict):
            stage_exec = stage_executors.get("tokenize_check") or stage_executors.get("tokenize")
            if isinstance(stage_exec, dict):
                merged_exec = dict(stage_exec)
                if isinstance(stage_cfg.get("executor"), dict):
                    merged_exec.update(stage_cfg["executor"])
                stage_cfg["executor"] = merged_exec
        profile_copy = dict(ctx.profile)
        profile_copy["executor"] = dict(ctx.profile.get("executor", {}))
        profile_copy["executor"]["type"] = "slurm"
        executor = build_executor(profile_copy, stage_cfg)
    if not isinstance(executor, SlurmExecutor):
        return False

    check_ranks = int(token_cfg.get("check_parallel_ranks", 0) or 0)
    if check_ranks <= 0:
        check_ranks = int(token_cfg.get("slurm_array_ranks", 0) or 0)
    if check_ranks <= 0:
        check_ranks = len(rank_dirs)
    check_ranks = max(1, min(check_ranks, len(rank_dirs)))

    parts_dir = ctx.tokenized_dir / "_check_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for part_file in parts_dir.glob("part_*.json"):
        part_file.unlink()

    array_logs_root = ctx.logs_root / "tokenize" / "check" / "array"
    merge_logs_root = ctx.logs_root / "tokenize" / "check" / "merge"
    array_logs_root.mkdir(parents=True, exist_ok=True)
    merge_logs_root.mkdir(parents=True, exist_ok=True)
    array_sbatch_root = array_logs_root / "sbatch"
    merge_sbatch_root = merge_logs_root / "sbatch"
    array_slurm_root = array_logs_root / "slurm"
    merge_slurm_root = merge_logs_root / "slurm"
    array_sbatch_root.mkdir(parents=True, exist_ok=True)
    merge_sbatch_root.mkdir(parents=True, exist_ok=True)
    array_slurm_root.mkdir(parents=True, exist_ok=True)
    merge_slurm_root.mkdir(parents=True, exist_ok=True)

    stage_log = _stage_log_path(array_logs_root, ctx.env, "tokenize_check", "array")
    array_spec = f"0-{check_ranks - 1}"
    lang_count = max(1, int(getattr(ctx, "lang_count", len(languages) or 0) or 0))
    job_name = f"tokc{lang_count}l"
    array_cmd = [
        "python3",
        str(ctx.repo_root / "pipeline_tools" / "check_tokenized_parallel.py"),
        "--tokenized-dir",
        str(ctx.tokenized_dir),
        "--output-dir",
        str(parts_dir),
        "--mode",
        "worker",
    ]
    array_job_id = executor.submit_array(
        array_cmd,
        ctx.env,
        ctx.args.dry_run,
        stage_log,
        array=array_spec,
        slurm_output=str(array_slurm_root / "tokenize_check_array_%A_%a.log"),
        job_name=job_name,
        script_dir=array_sbatch_root,
    )
    if ctx.args.dry_run or not array_job_id:
        return True

    merge_log = _stage_log_path(merge_logs_root, ctx.env, "tokenize_check", "merge")
    merge_job_name = f"tokcm{lang_count}l"
    merge_cmd = [
        "python3",
        str(ctx.repo_root / "pipeline_tools" / "check_tokenized_parallel.py"),
        "--tokenized-dir",
        str(ctx.tokenized_dir),
        "--output-dir",
        str(parts_dir),
        "--mode",
        "merge",
    ]
    if languages:
        merge_cmd.extend(["--expected-languages", ",".join(languages)])
    english_key = str(token_cfg.get("language_rates_english_key", "english")).strip() or "english"
    merge_cmd.extend(["--english-key", english_key])
    merge_cmd.extend(["--eval-fraction", str(float(token_cfg.get("eval_fraction", 0.02)))])
    if bool(token_cfg.get("english_median_check", False)):
        merge_cmd.append("--english-median-check")
        merge_cmd.extend(
            [
                "--english-median-tolerance",
                str(float(token_cfg.get("english_median_tolerance", 0.05))),
            ]
        )
    merge_job_id = executor.submit(
        merge_cmd,
        ctx.env,
        ctx.args.dry_run,
        merge_log,
        dependency=f"afterok:{array_job_id}",
        job_name=merge_job_name,
        script_dir=merge_sbatch_root,
        slurm_output=str(merge_slurm_root / "tokenize_check_merge_%A.log"),
    )
    if merge_job_id:
        executor.wait_for_jobs([(merge_job_id, merge_log)], ctx.args.dry_run)
    return True


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
            languages = _resolve_languages_for_tokenize(ctx)
            shard_counts_tsv = _resolve_shard_doc_counts_tsv(ctx, languages=languages)
            rates_path = _build_language_rates_for_tokenize(ctx, languages, counts_tsv=shard_counts_tsv)
            skip_if_exists = bool(ctx.tokenize_cfg.get("skip_if_output_exists", False))
            existing_artifacts = [ctx.tokenized_dir / "dataset_info.json", ctx.tokenized_dir / "state.json"]
            has_existing = all(p.exists() for p in existing_artifacts)
            if skip_if_exists and has_existing:
                ctx.logger.info("[SKIP] tokenize stage (output exists)")
            elif bool(ctx.tokenize_cfg.get("slurm_array_enabled", False)):
                _submit_tokenize_slurm_array(
                    ctx,
                    languages=languages,
                    rates_path=rates_path,
                    tokenize_model_name=tokenize_model_name,
                    shard_counts_tsv=shard_counts_tsv,
                )
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
                    "--eval_seed",
                    str(int(ctx.tokenize_cfg.get("eval_seed", 42))),
                    "--sample_seed",
                    str(int(ctx.tokenize_cfg.get("sample_seed", 42))),
                    "--max_length",
                    str(int(ctx.tokenize_cfg.get("max_length", 1024))),
                    "--language-map",
                    str(ctx.language_map_path),
                    "--mask_pad_labels",
                    "true" if bool(ctx.tokenize_cfg.get("mask_pad_labels", True)) else "false",
                ]
                shard_plan_dir = ctx.tokenized_dir.parent / f"{ctx.tokenized_dir.name}_shard_plan"
                if not languages:
                    raise SystemExit("Tokenization requires resolved languages for shard-plan creation.")
                if not shard_counts_tsv.exists():
                    if ctx.args.dry_run:
                        return
                    raise SystemExit(f"Missing shard doc counts TSV: {shard_counts_tsv}")
                force_replan = bool(ctx.tokenize_cfg.get("shard_counts_from_sampled", False))
                if force_replan or (not _plan_complete(shard_plan_dir, 1)):
                    plan_cmd = _build_shard_plan_cmd(
                        ctx,
                        shard_counts_tsv=shard_counts_tsv,
                        languages=languages,
                        world_size=1,
                        rates_path=rates_path,
                    )
                    _run_stage_cmd(
                        cmd=plan_cmd,
                        env=ctx.env,
                        dry_run=ctx.args.dry_run,
                        logs_root=ctx.logs_root,
                        stage="tokenize_shard_plan",
                        executor=ctx.executor,
                    )
                cmd.extend(["--shard-plan-dir", str(shard_plan_dir)])
                max_samples = int(ctx.tokenize_cfg.get("max_samples", 0) or 0)
                if max_samples > 0:
                    cmd.extend(["--max_samples", str(max_samples)])
                _run_stage_cmd(cmd=cmd, env=ctx.env, dry_run=ctx.args.dry_run, logs_root=ctx.logs_root, stage="tokenize", executor=ctx.executor)
            check_done = _submit_tokenize_parallel_check(ctx, languages=languages)
            if not check_done:
                check_tokenize(
                    ctx.tokenized_dir,
                    dry_run=ctx.args.dry_run,
                    expected_languages=languages,
                    english_median_check=bool(ctx.tokenize_cfg.get("english_median_check", False)),
                    english_key=str(ctx.tokenize_cfg.get("language_rates_english_key", "english")).strip() or "english",
                    english_median_tolerance=float(ctx.tokenize_cfg.get("english_median_tolerance", 0.05)),
                    eval_fraction=float(ctx.tokenize_cfg.get("eval_fraction", 0.02)),
                )
            if not ctx.args.dry_run:
                _save_stage_state(ctx.state_path, "tokenize", signature, [str(ctx.tokenized_dir)])
