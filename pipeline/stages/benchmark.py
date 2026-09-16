from __future__ import annotations

import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .common import ensure_lm_eval_layout
from .runtime import (
    _abs,
    _build_run_label,
    _load_json,
    _resolve_profile_languages,
    _run_stage_cmd,
    _save_stage_state,
    _select_tasks_for_languages,
    _should_skip_stage,
    _stage_signature,
)
from ..checks import check_benchmark
from ..context import PipelineContext
from ..runtime import _deep_merge
from ..training import adapter_checkpoint_provenance, resolve_adapter_eval_path
from .per_run_summary import write_per_run_summary


def _read_selected_tasks_file(path) -> list[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _extract_tasks_from_payload(payload: object) -> set[str]:
    tasks: set[str] = set()
    if not isinstance(payload, dict):
        return tasks
    task_name = payload.get("task")
    if isinstance(task_name, str):
        tasks.add(task_name)
    results = payload.get("results")
    if isinstance(results, dict):
        tasks.update(task for task in results if isinstance(task, str) and not task.startswith("_"))
    return tasks


def _load_results_payloads_from_dir(path: Path) -> list[dict[str, object]]:
    candidates = sorted(path.rglob("results_*.json"))
    payloads: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _collect_tasks_from_output_dir(path: Path) -> tuple[list[Path], set[str]]:
    files = sorted(path.glob("*_lm_eval.jsonl"))
    files.extend(sorted(path.glob("no_language_ids.json")))
    files.extend(sorted(path.glob("with_language_ids_*.json")))
    task_seen: set[str] = set()
    for file in files:
        if file.is_dir():
            for payload in _load_results_payloads_from_dir(file):
                task_seen.update(_extract_tasks_from_payload(payload))
            continue
        if file.suffix == ".jsonl":
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_seen.update(_extract_tasks_from_payload(payload))
            continue
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        task_seen.update(_extract_tasks_from_payload(payload))
    return files, task_seen


def _task_family(task: str) -> str:
    if task.startswith("belebele_"):
        return "Belebele"
    if task.startswith("flores_en_perplexity_"):
        return "FLORES"
    if task.startswith("sib200_"):
        return "SIB200"
    if task.startswith("taxi1500_"):
        return "Taxi1500"
    return "Other"


def _task_family_counts(tasks: list[str] | set[str]) -> dict[str, int]:
    counts = {"Belebele": 0, "FLORES": 0, "SIB200": 0, "Taxi1500": 0}
    for task in tasks:
        family = _task_family(task)
        if family in counts:
            counts[family] += 1
    return counts


def _validate_strict_task_selection(selected_tasks: list[str], cfg: dict[str, object]) -> dict[str, int]:
    """Validate the selected task panel before spending GPU time on evaluation."""
    counts = _task_family_counts(selected_tasks)
    if len(selected_tasks) != len(set(selected_tasks)):
        raise SystemExit("Benchmark strict completeness: selected task list contains duplicates")
    expected_count_raw = cfg.get("expected_task_count")
    if expected_count_raw is not None:
        try:
            expected_count = int(expected_count_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit("benchmark.expected_task_count must be an integer") from exc
        if len(selected_tasks) != expected_count:
            raise SystemExit(
                f"Benchmark strict completeness: selected {len(selected_tasks)} tasks, expected {expected_count}"
            )

    expected_breakdown_raw = cfg.get("expected_task_breakdown")
    if expected_breakdown_raw is not None:
        if not isinstance(expected_breakdown_raw, dict):
            raise SystemExit("benchmark.expected_task_breakdown must be an object")
        try:
            expected_breakdown = {
                family: int(expected_breakdown_raw.get(family, 0))
                for family in counts
            }
        except (TypeError, ValueError) as exc:
            raise SystemExit("benchmark.expected_task_breakdown values must be integers") from exc
        if counts != expected_breakdown:
            raise SystemExit(
                f"Benchmark strict completeness: selected task breakdown {counts}, expected {expected_breakdown}"
            )
    if not selected_tasks:
        raise SystemExit("Benchmark strict completeness: selected task list is empty")
    return counts


def _resolve_eval_script(ctx: PipelineContext) -> str:
    raw = str(ctx.benchmark_cfg.get("eval_script", "")).strip()
    if raw:
        return _abs(ctx.repo_root, raw)
    return _abs(ctx.repo_root, "scripts/eval/lm_eval_checkpoint_adapter_only.sh")


def _extract_test_data_file(config_path: Path) -> str | None:
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("test:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _resolve_lm_eval_relative_path(include_path: Path, rel_path: Path) -> Path | None:
    if not rel_path.parts or rel_path.parts[0] != "lm_eval":
        return None
    current = include_path
    while current.name != "lm_eval" and current.parent != current:
        current = current.parent
    if current.name != "lm_eval":
        return None
    return current.parent / rel_path


def _validate_flores_task_data_paths(repo_root: Path, include_path: Path, selected_tasks: list[str]) -> None:
    for task in selected_tasks:
        if not task.startswith("flores_en_perplexity_"):
            continue
        task_yaml = include_path / f"{task}.yaml"
        if not task_yaml.exists():
            raise SystemExit(f"Missing FLORES task config: {task_yaml}")
        data_file = _extract_test_data_file(task_yaml)
        if not data_file:
            raise SystemExit(f"FLORES task config missing dataset test path: {task_yaml}")
        data_path = Path(data_file)
        if not data_path.is_absolute():
            resolved = _resolve_lm_eval_relative_path(include_path, data_path)
            data_path = resolved if resolved is not None else (repo_root / data_path)
        if not data_path.exists():
            raise SystemExit(
                f"FLORES dataset path not found for task '{task}': {data_path}. "
                "Regenerate/repair flores_en_perplexity YAML configs."
            )


def _validate_task_yaml_presence(include_path: Path, selected_tasks: list[str], prefix: str, label: str) -> None:
    for task in selected_tasks:
        if not task.startswith(prefix):
            continue
        task_yaml = include_path / f"{task}.yaml"
        if not task_yaml.exists():
            raise SystemExit(f"Missing {label} task config: {task_yaml}")


def _selected_external_task_configs_present(repo_root: Path, selected_tasks: list[str]) -> bool:
    task_root = repo_root / "scripts" / "eval" / "latest_checkpoint" / "lm-evaluation-harness" / "lm_eval" / "tasks"
    required: list[Path] = []
    required.extend(task_root / "sib200" / "configs" / f"{task}.yaml" for task in selected_tasks if task.startswith("sib200_"))
    required.extend(task_root / "taxi1500" / "configs" / f"{task}.yaml" for task in selected_tasks if task.startswith("taxi1500_"))
    return bool(required) and all(path.exists() for path in required)


def _ensure_external_task_configs(repo_root: Path, selected_tasks: list[str]) -> None:
    if _selected_external_task_configs_present(repo_root, selected_tasks):
        return
    generator = (
        repo_root
        / "scripts"
        / "eval"
        / "latest_checkpoint"
        / "lm-evaluation-harness"
        / "lm_eval"
        / "tasks"
        / "generate_local_task_configs.py"
    )
    if not generator.exists():
        raise SystemExit(f"Missing external task generator: {generator}")
    subprocess.run([sys.executable, str(generator)], check=True, cwd=repo_root)


def _default_lm_eval_include_dirs(repo_root: Path, selected_tasks: list[str]) -> list[Path]:
    task_root = repo_root / "scripts" / "eval" / "latest_checkpoint" / "lm-evaluation-harness" / "lm_eval" / "tasks"
    include_dirs: list[Path] = []
    if any(task.startswith("flores_en_perplexity_") for task in selected_tasks):
        include_dirs.append(task_root / "flores_en_perplexity" / "configs")
    if any(task.startswith("sib200_") for task in selected_tasks):
        include_dirs.append(task_root / "sib200" / "configs")
    if any(task.startswith("taxi1500_") for task in selected_tasks):
        include_dirs.append(task_root / "taxi1500" / "configs")
    return include_dirs


def _resolve_benchmark_languages(ctx: PipelineContext) -> list[str]:
    langs = _resolve_profile_languages(
        sample_cfg=ctx.sample_cfg,
        sample_manifest_path=ctx.sampled_manifest,
        sampled_dir=ctx.sampled_dir,
    )
    if langs:
        return langs

    check_path = ctx.tokenized_dir / "_check.json"
    if check_path.exists():
        try:
            payload = json.loads(check_path.read_text(encoding="utf-8"))
            samples = payload.get("samples_per_language", {})
            if isinstance(samples, dict) and samples:
                return sorted({str(k) for k in samples.keys() if str(k).strip()})
        except Exception:
            pass

    token_cfg = ctx.tokenize_cfg
    csv_path = str(token_cfg.get("languages_csv", "")).strip()
    if csv_path:
        column = str(token_cfg.get("languages_csv_column", "subset")).strip() or "subset"
        split_value = str(token_cfg.get("languages_csv_split", "train")).strip()
        split_column = str(token_cfg.get("languages_csv_split_column", "split")).strip() or "split"
        languages: set[str] = set()
        try:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and column in reader.fieldnames:
                    for row in reader:
                        if split_value:
                            if str(row.get(split_column, "")).strip() != split_value:
                                continue
                        value = str(row.get(column, "")).strip()
                        if value:
                            languages.add(value)
        except Exception:
            languages = set()
        extras = token_cfg.get("languages_extra", [])
        if isinstance(extras, list):
            for item in extras:
                if isinstance(item, str) and item.strip():
                    languages.add(item.strip())
        if languages:
            return sorted(languages)

    return []


def _expected_max_steps(ctx: PipelineContext, run: dict[str, object]) -> int | None:
    train_cfg = _deep_merge(ctx.train_cfg.get("train", {}), run.get("train", {}))
    raw = train_cfg.get("max_steps") if isinstance(train_cfg, dict) else None
    if raw is None:
        return None
    try:
        steps = int(raw)
    except (TypeError, ValueError):
        return None
    return steps if steps > 0 else None


def _trainer_state_global_step(run_dir: Path, *, dry_run: bool) -> int | None:
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        if dry_run:
            return None
        raise SystemExit(
            f"benchmark: checkpoint_step_source='trainer_state' but trainer_state.json is missing: {state_path}"
        )
    payload = _load_json(state_path)
    raw = payload.get("global_step") if isinstance(payload, dict) else None
    try:
        step = int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"benchmark: trainer_state.json has invalid global_step for {run_dir}: {raw!r}") from exc
    if step <= 0:
        raise SystemExit(f"benchmark: trainer_state.json global_step must be positive for {run_dir}: {step}")
    return step


def _expected_checkpoint_step(ctx: PipelineContext, run: dict[str, object], run_dir: Path) -> int | None:
    source = str(ctx.benchmark_cfg.get("checkpoint_step_source", "max_steps")).strip().lower()
    if source in {"", "max_steps", "config", "configured"}:
        return _expected_max_steps(ctx, run)
    if source in {"trainer_state", "state", "global_step"}:
        return _trainer_state_global_step(run_dir, dry_run=ctx.args.dry_run)
    raise SystemExit(
        "benchmark.checkpoint_step_source must be one of "
        f"'max_steps' or 'trainer_state' (got {source!r})"
    )


def stage_benchmark(ctx: PipelineContext) -> None:
    if "benchmark" in ctx.active_stages and bool(ctx.benchmark_cfg.get("enabled", False)):
        resolved_eval_tokenizer = str(ctx.benchmark_cfg.get("tokenizer", "")).strip()
        if not resolved_eval_tokenizer:
            if bool(ctx.tokenizer_extension_cfg.get("enabled", False)) and bool(ctx.embedding_init_cfg.get("enabled", False)):
                if ctx.embedding_init_manifest_path.exists():
                    resolved_eval_tokenizer = str(_load_json(ctx.embedding_init_manifest_path).get("initialized_model_dir", ""))
            if not resolved_eval_tokenizer:
                resolved_eval_tokenizer = str(ctx.train_cfg.get("model_name_or_path"))
        signature = _stage_signature(
            {
                "stage": "benchmark",
                "cfg": ctx.benchmark_cfg,
                "resolved_eval_tokenizer": resolved_eval_tokenizer,
                "sample_manifest": str(ctx.sampled_manifest),
                "train_runs": [run.get("id") for run in ctx.train_cfg.get("runs", [])],
            }
        )
        selected_tasks_file = ctx.benchmark_root / "selected_tasks.txt"
        selected_tasks_summary = ctx.benchmark_root / "selected_tasks_summary.json"
        if ctx.args.run_id and len(ctx.args.run_id) == 1:
            run_id = str(ctx.args.run_id[0])
            selected_tasks_file = ctx.benchmark_root / f"selected_tasks.{run_id}.txt"
            selected_tasks_summary = ctx.benchmark_root / f"selected_tasks_summary.{run_id}.json"
        benchmark_summary_path = ctx.benchmark_root / "benchmark_summary.json"
        per_run_summary_root = ctx.benchmark_root / "run_summaries"
        all_runs = ctx.train_cfg.get("runs", [])
        if ctx.args.run_id:
            run_id_set = {str(rid) for rid in ctx.args.run_id}
            runs = [run for run in all_runs if str(run.get("id")) in run_id_set]
        else:
            runs = all_runs
        state_stage = (
            f"benchmark:{runs[0]['id']}"
            if len(runs) == 1 and ctx.args.run_id
            else "benchmark"
        )
        force_stages = set(ctx.force_stages)
        if "benchmark" in force_stages:
            force_stages.add(state_stage)
        expected = [selected_tasks_file, selected_tasks_summary, benchmark_summary_path]
        expected.extend(per_run_summary_root / f"{run['id']}.json" for run in runs)
        expected.extend([ctx.benchmark_root / str(run["id"]) for run in runs])
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage=state_stage,
            signature=signature,
            expected_artifacts=expected,
            force_stages=force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] benchmark stage (cache hit)")
        else:
            ensure_lm_eval_layout(ctx.repo_root, ctx.logger)
            tasks_file = _abs(ctx.repo_root, str(ctx.benchmark_cfg["tasks_file"]))
            langs = _resolve_benchmark_languages(ctx)
            print(f"[INFO] Resolved benchmark languages, Eval is only performed in those languages: {langs}")
            if not langs:
                raise SystemExit("Benchmark stage could not resolve trained language list.")
            ctx.benchmark_root.mkdir(parents=True, exist_ok=True)
            select_cmd = [
                "python3",
                str(ctx.repo_root / "pipeline_tools" / "select_multilingual_eval_tasks.py"),
                "--tasks-file",
                str(tasks_file),
                "--languages",
                ",".join(langs),
                "--output-file",
                str(selected_tasks_file),
                "--summary-json",
                str(selected_tasks_summary),
            ]
            if bool(ctx.benchmark_cfg.get("include_belebele", True)):
                select_cmd.append("--include-belebele")
            if bool(ctx.benchmark_cfg.get("include_sib200", True)):
                select_cmd.append("--include-sib200")
            if bool(ctx.benchmark_cfg.get("include_taxi1500", True)):
                select_cmd.append("--include-taxi1500")
            if bool(ctx.benchmark_cfg.get("include_flores", True)):
                select_cmd.append("--include-flores")
                select_cmd.extend(["--flores-mode", str(ctx.benchmark_cfg.get("flores_mode", "both"))])
            _run_stage_cmd(
                cmd=select_cmd,
                env=ctx.env,
                dry_run=ctx.args.dry_run,
                logs_root=ctx.logs_root,
                stage="benchmark",
                suffix=(
                    f"task_select_{ctx.args.run_id[0]}"
                    if len(ctx.args.run_id) == 1
                    else "task_select"
                ),
                executor=ctx.executor,
            )
            if ctx.args.dry_run:
                selected_tasks = _select_tasks_for_languages(
                    tasks_file=tasks_file,
                    languages=langs,
                    include_belebele=bool(ctx.benchmark_cfg.get("include_belebele", True)),
                    include_sib200=bool(ctx.benchmark_cfg.get("include_sib200", True)),
                    include_taxi1500=bool(ctx.benchmark_cfg.get("include_taxi1500", True)),
                    include_flores=bool(ctx.benchmark_cfg.get("include_flores", True)),
                    flores_mode=str(ctx.benchmark_cfg.get("flores_mode", "both")),
                )
            else:
                selected_tasks = _read_selected_tasks_file(selected_tasks_file)
            strict_completeness = bool(ctx.benchmark_cfg.get("strict_completeness", False))
            if strict_completeness and bool(ctx.benchmark_cfg.get("allow_missing_lm_eval", False)):
                raise SystemExit("Benchmark strict completeness requires allow_missing_lm_eval=false")
            expected_task_breakdown = (
                _validate_strict_task_selection(selected_tasks, ctx.benchmark_cfg)
                if strict_completeness
                else _task_family_counts(selected_tasks)
            )
            if not ctx.args.dry_run and (not selected_tasks) and bool(ctx.benchmark_cfg.get("fail_if_no_tasks", True)):
                raise SystemExit("Benchmark stage selected zero tasks.")
            tasks_csv = ",".join(selected_tasks) if selected_tasks else "belebele_eng_Latn"
            lm_eval_bin = str(ctx.benchmark_cfg.get("lm_eval_bin", "lm_eval"))
            if not ctx.args.dry_run and (not bool(ctx.benchmark_cfg.get("allow_missing_lm_eval", False))):
                if shutil.which(lm_eval_bin) is None:
                    raise SystemExit(f"Benchmark stage requires '{lm_eval_bin}' in PATH.")
            eval_script = _resolve_eval_script(ctx)
            if not eval_script.exists():
                raise SystemExit(f"Benchmark eval script not found: {eval_script}")
            tokenizer = resolved_eval_tokenizer
            batch_size = str(ctx.benchmark_cfg.get("batch_size", "auto"))
            lang_mode = str(ctx.benchmark_cfg.get("lang_mode", "")).strip()
            extra_args = str(ctx.benchmark_cfg.get("extra_args", "")).strip()
            if any(task.startswith(("sib200_", "taxi1500_")) for task in selected_tasks):
                _ensure_external_task_configs(ctx.repo_root, selected_tasks)
            selected_flores_tasks = [task for task in selected_tasks if task.startswith("flores_en_perplexity_")]
            include_dirs = _default_lm_eval_include_dirs(ctx.repo_root, selected_tasks)
            for include_dir in include_dirs:
                if include_dir.parent.name == "flores_en_perplexity" and selected_flores_tasks:
                    _validate_flores_task_data_paths(ctx.repo_root, include_dir, selected_flores_tasks)
                elif include_dir.parent.name == "sib200":
                    _validate_task_yaml_presence(include_dir, selected_tasks, "sib200_", "SIB200")
                elif include_dir.parent.name == "taxi1500":
                    _validate_task_yaml_presence(include_dir, selected_tasks, "taxi1500_", "Taxi1500")

            parsed_extra_args = shlex.split(extra_args) if extra_args else []
            if "--include-path" in parsed_extra_args:
                idx = parsed_extra_args.index("--include-path")
                parsed_extra_args[idx] = "--include_path"
            if include_dirs and "--include_path" not in parsed_extra_args:
                parsed_extra_args.extend(["--include_path", ":".join(str(path) for path in include_dirs)])
            extra_args = shlex.join(parsed_extra_args) if parsed_extra_args else ""
            default_eval_project = str(ctx.wandb_cfg.get("project", "proof_eval")) if bool(ctx.wandb_cfg.get("enabled", False)) else "proof_eval"
            wandb_project = str(ctx.benchmark_cfg.get("wandb_project", default_eval_project))
            wandb_prefix_root = str(
                ctx.benchmark_cfg.get(
                    "wandb_prefix",
                    str(ctx.wandb_cfg.get("eval_prefix", "eval/benchmark")) if bool(ctx.wandb_cfg.get("enabled", False)) else "eval/benchmark",
                )
            )
            wandb_mode = str(
                ctx.benchmark_cfg.get(
                    "wandb_mode",
                    str(ctx.wandb_cfg.get("mode", "disabled")) if bool(ctx.wandb_cfg.get("enabled", False)) else "disabled",
                )
            )
            run_name_by_id: dict[str, str] = {
                str(run["id"]): _build_run_label(
                    base_run_label=ctx.base_run_label,
                    run=run,
                    default_num_experts=int(ctx.train_cfg.get("num_experts", 0)),
                )
                for run in runs
            }
            for run in runs:
                run_id = str(run["id"])
                run_name_label = run_name_by_id.get(run_id, run_id)
                checkpoint_dir = ctx.train_root / run_id
                if not checkpoint_dir.exists():
                    raise SystemExit(f"Benchmark checkpoint directory missing: {checkpoint_dir}")
                expected_step = _expected_checkpoint_step(ctx, run, checkpoint_dir)
                strict_final_checkpoint = bool(ctx.benchmark_cfg.get("strict_final_checkpoint", False))
                require_direct_save_marker = ctx.benchmark_cfg.get("require_direct_save_marker")
                if require_direct_save_marker is None:
                    require_direct_save_marker = strict_final_checkpoint
                if ctx.args.dry_run:
                    try:
                        adapter_dir = resolve_adapter_eval_path(
                            checkpoint_dir,
                            expected_step=expected_step,
                            strict_final_checkpoint=strict_final_checkpoint,
                            require_direct_save_marker=bool(require_direct_save_marker),
                        )
                    except SystemExit:
                        adapter_dir = checkpoint_dir
                else:
                    adapter_dir = resolve_adapter_eval_path(
                        checkpoint_dir,
                        expected_step=expected_step,
                        strict_final_checkpoint=strict_final_checkpoint,
                        require_direct_save_marker=bool(require_direct_save_marker),
                    )
                out_dir = ctx.benchmark_root / run_id
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "bash",
                    str(eval_script),
                    "--checkpoint",
                    str(adapter_dir),
                    "--tokenizer",
                    tokenizer,
                    "--tasks",
                    tasks_csv,
                    "--batch-size",
                    batch_size,
                    "--output-dir",
                    str(out_dir),
                    "--wandb-project",
                    wandb_project,
                    "--wandb-prefix",
                    f"{wandb_prefix_root}/{run_name_label}",
                    "--wandb-mode",
                    wandb_mode,
                ]
                if lang_mode:
                    cmd.extend(["--lang-mode", lang_mode])
                if extra_args:
                    cmd.extend(["--extra-args", extra_args])
                _run_stage_cmd(
                    cmd=cmd,
                    env=ctx.env,
                    dry_run=ctx.args.dry_run,
                    logs_root=ctx.logs_root,
                    stage="benchmark",
                    suffix=run_name_label,
                    executor=ctx.executor,
                )
                if not ctx.args.dry_run:
                    files, task_seen = _collect_tasks_from_output_dir(out_dir)
                    selected_task_set = set(selected_tasks)
                    matched_tasks = selected_task_set & task_seen
                    benchmark_row = {
                        "run_id": run_id,
                        "evaluated_checkpoint": str(adapter_dir.resolve()),
                        "adapter_provenance": adapter_checkpoint_provenance(
                            adapter_dir,
                            expected_step=expected_step,
                            require_direct_save_marker=bool(require_direct_save_marker),
                        ),
                        "evaluated_tokenizer": tokenizer,
                        "output_dir": str(out_dir.resolve()),
                        "num_output_files": len(files),
                        "num_tasks_expected": len(selected_tasks),
                        "num_tasks_seen": len(matched_tasks),
                        "expected_task_breakdown": expected_task_breakdown,
                        "seen_task_breakdown": _task_family_counts(matched_tasks),
                        "missing_tasks": sorted(selected_task_set - task_seen),
                        "extra_tasks_seen": sorted(task_seen - selected_task_set),
                        "strict_completeness": strict_completeness,
                    }
                    write_per_run_summary(
                        per_run_path=per_run_summary_root / f"{run_id}.json",
                        aggregate_path=benchmark_summary_path,
                        metadata={
                            "tasks_file": str(tasks_file.resolve()),
                            "selected_tasks_file": str(selected_tasks_file.resolve()),
                            "selected_tasks_summary": str(selected_tasks_summary.resolve()),
                            "resolved_eval_tokenizer": tokenizer,
                            "lm_eval_limit": str(ctx.env.get("LM_EVAL_LIMIT", "")).strip(),
                            "embedding_init_manifest": (
                                str(ctx.embedding_init_manifest_path.resolve())
                                if ctx.embedding_init_manifest_path.exists()
                                else ""
                            ),
                            "num_selected_tasks": len(selected_tasks),
                            "expected_task_breakdown": expected_task_breakdown,
                            "strict_completeness": strict_completeness,
                        },
                        run=benchmark_row,
                    )
            if not ctx.args.dry_run:
                check_benchmark(
                    ctx.benchmark_root,
                    benchmark_summary_path,
                    dry_run=ctx.args.dry_run,
                    check_path=(ctx.benchmark_root / f"_check.{runs[0]['id']}.json") if len(runs) == 1 else None,
                    run_ids={str(run["id"]) for run in runs},
                )
                _save_stage_state(ctx.state_path, state_stage, signature, [str(path) for path in expected])
