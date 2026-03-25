from __future__ import annotations

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
    _dump_json,
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
            data_path = repo_root / data_path
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


def _ensure_external_task_configs(repo_root: Path) -> None:
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
        benchmark_summary_path = ctx.benchmark_root / "benchmark_summary.json"
        runs = ctx.train_cfg.get("runs", [])
        expected = [selected_tasks_file, selected_tasks_summary, benchmark_summary_path]
        expected.extend([ctx.benchmark_root / str(run["id"]) for run in runs])
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="benchmark",
            signature=signature,
            expected_artifacts=expected,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] benchmark stage (cache hit)")
        else:
            ensure_lm_eval_layout(ctx.repo_root, ctx.logger)
            tasks_file = _abs(ctx.repo_root, str(ctx.benchmark_cfg["tasks_file"]))
            langs = _resolve_profile_languages(
                sample_cfg=ctx.sample_cfg,
                sample_manifest_path=ctx.sampled_manifest,
                sampled_dir=ctx.sampled_dir,
            )
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
                suffix="task_select",
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
            extra_args = str(ctx.benchmark_cfg.get("extra_args", "")).strip()
            if any(task.startswith(("sib200_", "taxi1500_")) for task in selected_tasks):
                _ensure_external_task_configs(ctx.repo_root)
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
            benchmark_rows: list[dict[str, object]] = []
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
                out_dir = ctx.benchmark_root / run_id
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "bash",
                    str(eval_script),
                    "--checkpoint",
                    str(checkpoint_dir),
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
                    benchmark_rows.append(
                        {
                            "run_id": run_id,
                            "evaluated_checkpoint": str(checkpoint_dir.resolve()),
                            "evaluated_tokenizer": tokenizer,
                            "output_dir": str(out_dir.resolve()),
                            "num_output_files": len(files),
                            "num_tasks_expected": len(selected_tasks),
                            "num_tasks_seen": len(task_seen),
                            "missing_tasks": sorted(set(selected_tasks) - task_seen),
                        }
                    )
            if not ctx.args.dry_run:
                _dump_json(
                    benchmark_summary_path,
                    {
                        "tasks_file": str(tasks_file.resolve()),
                        "selected_tasks_file": str(selected_tasks_file.resolve()),
                        "selected_tasks_summary": str(selected_tasks_summary.resolve()),
                        "resolved_eval_tokenizer": tokenizer,
                        "embedding_init_manifest": (
                            str(ctx.embedding_init_manifest_path.resolve()) if ctx.embedding_init_manifest_path.exists() else ""
                        ),
                        "num_selected_tasks": len(selected_tasks),
                        "runs": benchmark_rows,
                    },
                )
                _save_stage_state(ctx.state_path, "benchmark", signature, [str(path) for path in expected])
                check_benchmark(ctx.benchmark_root, benchmark_summary_path, dry_run=ctx.args.dry_run)
