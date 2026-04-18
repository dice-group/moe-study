from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openpyxl.utils import get_column_letter

from .common import TRAINABLE_RE, flatten_payload, safe_float, safe_json_dumps, safe_json_load
from .compact import metric_direction, parse_task_identity
from .excel import SECTION_FILL, SECTION_FONT, style_header


def load_readme_hparams(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    out: dict[str, Any] = {}
    patterns = {
        "learning_rate": r"- learning_rate:\s+([^\n]+)",
        "train_batch_size": r"- train_batch_size:\s+([^\n]+)",
        "eval_batch_size": r"- eval_batch_size:\s+([^\n]+)",
        "lr_scheduler_type": r"- lr_scheduler_type:\s+([^\n]+)",
        "max_steps": r"- training_steps:\s+([^\n]+)",
        "seed": r"- seed:\s+([^\n]+)",
    }
    text = path.read_text(encoding="utf-8")
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = match.group(1).strip()
    return out


def load_trainer_state_summary(path: Path) -> dict[str, Any]:
    payload = safe_json_load(path)
    if not payload:
        return {}
    return {
        "eval_steps": payload.get("eval_steps", ""),
        "save_steps": payload.get("save_steps", ""),
        "max_steps": payload.get("max_steps", ""),
        "num_train_epochs": payload.get("num_train_epochs", ""),
        "train_batch_size": payload.get("train_batch_size", ""),
    }


def load_trainable_stats_from_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    last_match = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = TRAINABLE_RE.search(line)
        if match:
            last_match = match
    if last_match is None:
        return {}
    return {"trainable_params": int(last_match.group(1).replace(",", "")), "all_params": int(last_match.group(2).replace(",", "")), "trainable_pct": float(last_match.group(3))}


def resolve_trainable_stats(run_id: str, run_dir: Path, logs_root: Path | None) -> dict[str, Any]:
    payload = safe_json_load(run_dir / "trainable_params.json")
    if payload:
        return payload
    if logs_root is None:
        return {}

    candidate_sets = [
        sorted((logs_root / "train_runs" / run_id).glob("*.train.log"), key=lambda path: path.stat().st_mtime, reverse=True),
        sorted((logs_root / "chain" / "slurm").glob(f"chain_train.{run_id}.*.slurm.log"), key=lambda path: path.stat().st_mtime, reverse=True),
    ]
    for candidates in candidate_sets:
        for candidate in candidates:
            payload = load_trainable_stats_from_log(candidate)
            if payload:
                payload["source_log"] = str(candidate)
                return payload
    return {}


def load_shard_plan_summary(tokenized_dir: Path) -> dict[str, Any]:
    plan_dir = tokenized_dir.parent / f"{tokenized_dir.name}_shard_plan"
    summary = safe_json_load(plan_dir / "summary.json")
    if not summary:
        return {}
    selected_targets = summary.get("selected_targets", {}) if isinstance(summary.get("selected_targets"), dict) else {}
    validation_targets = summary.get("validation_targets", {}) if isinstance(summary.get("validation_targets"), dict) else {}
    shard_counts: Counter[str] = Counter()
    for path in sorted(plan_dir.glob("rank_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            language = str(row.get("language", "")).strip()
            if language:
                shard_counts[language] += 1
    train_targets = {
        str(language): int(selected_targets.get(language, 0)) - int(validation_targets.get(language, 0))
        for language in selected_targets.keys()
    }
    return {
        "selected_targets": {str(k): int(v) for k, v in selected_targets.items()},
        "validation_targets": {str(k): int(v) for k, v in validation_targets.items()},
        "train_targets": train_targets,
        "selected_shards": dict(shard_counts),
    }


def build_data_overview(*, sample_manifest_path: Path, tokenized_dir: Path, profile_payload: dict[str, Any], benchmark_summary_path: Path) -> dict[str, Any]:
    sample_manifest = safe_json_load(sample_manifest_path)
    sample_check = safe_json_load(sample_manifest_path.parent / "_check.json")
    tokenized_check = safe_json_load(tokenized_dir / "_check.json")
    tokenized_info = safe_json_load(tokenized_dir / "dataset_info.json")
    benchmark_summary = safe_json_load(benchmark_summary_path)
    selected_tasks_summary_path = Path(str(benchmark_summary.get("selected_tasks_summary", "")).strip())
    selected_tasks_summary = safe_json_load(selected_tasks_summary_path) if str(selected_tasks_summary_path) else {}
    sample_cfg = profile_payload.get("sample", {}) if isinstance(profile_payload, dict) else {}
    tokenization_cfg = profile_payload.get("tokenization", {}) if isinstance(profile_payload, dict) else {}
    benchmark_cfg = profile_payload.get("benchmark", {}) if isinstance(profile_payload, dict) else {}
    shard_plan_summary = load_shard_plan_summary(tokenized_dir)
    source = sample_manifest.get("source", {}) if isinstance(sample_manifest.get("source"), dict) else {}
    selection = sample_manifest.get("selection", {}) if isinstance(sample_manifest.get("selection"), dict) else {}
    sample_languages = sample_check.get("languages", {}) if isinstance(sample_check.get("languages"), dict) else {}
    tokenized_samples = tokenized_check.get("samples_per_language", {}) if isinstance(tokenized_check.get("samples_per_language"), dict) else {}
    tokenized_tokens = tokenized_check.get("tokens_per_language", {}) if isinstance(tokenized_check.get("tokens_per_language"), dict) else {}
    tokenized_samples_by_split = tokenized_check.get("samples_per_language_by_split", {}) if isinstance(tokenized_check.get("samples_per_language_by_split"), dict) else {}
    tokenized_tokens_by_split = tokenized_check.get("tokens_per_language_by_split", {}) if isinstance(tokenized_check.get("tokens_per_language_by_split"), dict) else {}
    splits = tokenized_check.get("splits", {}) if isinstance(tokenized_check.get("splits"), dict) else {}
    dataset_splits = tokenized_info.get("splits", {}) if isinstance(tokenized_info.get("splits"), dict) else {}
    if (not tokenized_samples_by_split or not tokenized_tokens_by_split) and tokenized_dir.exists():
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(tokenized_dir))
            split_counts: dict[str, Counter[str]] = defaultdict(Counter)
            split_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
            for split_name, language, ids in zip(ds["split"], ds["language"], ds["input_ids"]):
                split_counts[str(split_name)][str(language)] += 1
                split_token_counts[str(split_name)][str(language)] += len(ids)
            tokenized_samples_by_split = {split: dict(counts) for split, counts in split_counts.items()}
            tokenized_tokens_by_split = {split: dict(counts) for split, counts in split_token_counts.items()}
        except Exception:
            pass
    languages = selection.get("languages", sample_cfg.get("languages", []))
    if not isinstance(languages, list):
        languages = []
    planned_selected_docs = shard_plan_summary.get("selected_targets", {}) if isinstance(shard_plan_summary.get("selected_targets"), dict) else {}
    planned_validation_docs = shard_plan_summary.get("validation_targets", {}) if isinstance(shard_plan_summary.get("validation_targets"), dict) else {}
    planned_train_docs = shard_plan_summary.get("train_targets", {}) if isinstance(shard_plan_summary.get("train_targets"), dict) else {}
    planned_selected_shards = shard_plan_summary.get("selected_shards", {}) if isinstance(shard_plan_summary.get("selected_shards"), dict) else {}
    language_rows: list[dict[str, Any]] = []
    for language in sorted(set([*sample_languages.keys(), *tokenized_samples.keys(), *tokenized_tokens.keys(), *planned_selected_docs.keys(), *[str(x) for x in languages]])):
        sampled = sample_languages.get(language, {}) if isinstance(sample_languages.get(language), dict) else {}
        language_rows.append({
            "language": language,
            "selected_docs": sampled.get("docs", planned_selected_docs.get(language, "")),
            "selected_shards": sampled.get("files", planned_selected_shards.get(language, "")),
            "selected_bytes": sampled.get("bytes", ""),
            "train_docs": planned_train_docs.get(language, ""),
            "validation_docs": planned_validation_docs.get(language, ""),
            "tokenized_examples": tokenized_samples.get(language, ""),
            "tokenized_train_examples": tokenized_samples_by_split.get("train", {}).get(language, "") if isinstance(tokenized_samples_by_split.get("train", {}), dict) else "",
            "tokenized_validation_examples": tokenized_samples_by_split.get("validation", {}).get(language, "") if isinstance(tokenized_samples_by_split.get("validation", {}), dict) else "",
            "tokenized_tokens": tokenized_tokens.get(language, ""),
            "tokenized_train_tokens": tokenized_tokens_by_split.get("train", {}).get(language, "") if isinstance(tokenized_tokens_by_split.get("train", {}), dict) else "",
            "tokenized_validation_tokens": tokenized_tokens_by_split.get("validation", {}).get(language, "") if isinstance(tokenized_tokens_by_split.get("validation", {}), dict) else "",
        })
    metadata_rows = [
        ("source_type", source.get("type", sample_cfg.get("source_type", ""))),
        ("source_data_files", source.get("data_files", sample_cfg.get("data_files", ""))),
        ("source_dataset_name", source.get("dataset_name", sample_cfg.get("dataset_name", ""))),
        ("source_dataset_config", source.get("dataset_config", sample_cfg.get("dataset_config", ""))),
        ("source_dataset_split", source.get("dataset_split", sample_cfg.get("dataset_split", ""))),
        ("languages", ",".join(str(x) for x in languages)),
        ("num_languages", sample_check.get("num_languages", tokenized_check.get("num_languages", len(languages)))),
        ("sampling_mode", selection.get("sampling_mode", "")),
        ("per_language_target", sample_cfg.get("per_language_target", "")),
        ("sample_seen_total", sample_manifest.get("seen_total", "")),
        ("sample_total_docs", sample_check.get("total_docs", "")),
        ("sample_total_bytes", sample_check.get("total_bytes", "")),
        ("tokenized_num_rows", tokenized_check.get("num_rows", "")),
        ("tokenized_train_rows", splits.get("train", dataset_splits.get("train", {}).get("num_examples", ""))),
        ("tokenized_validation_rows", splits.get("validation", dataset_splits.get("validation", {}).get("num_examples", ""))),
        ("tokenized_total_tokens", tokenized_check.get("total_tokens", "")),
        ("avg_tokens_per_example", (float(tokenized_check.get("total_tokens")) / float(tokenized_check.get("num_rows")) if safe_float(tokenized_check.get("total_tokens")) not in (None, 0.0) and safe_float(tokenized_check.get("num_rows")) not in (None, 0.0) else "")),
        ("tokenization_model_name", tokenization_cfg.get("model_name", "")),
        ("tokenization_max_length", tokenization_cfg.get("max_length", "")),
        ("tokenization_eval_fraction", tokenization_cfg.get("eval_fraction", "")),
        ("selected_docs_total", sum(int(v) for v in planned_selected_docs.values()) if planned_selected_docs else ""),
        ("train_docs_total", sum(int(v) for v in planned_train_docs.values()) if planned_train_docs else ""),
        ("validation_docs_total", sum(int(v) for v in planned_validation_docs.values()) if planned_validation_docs else ""),
        ("benchmark_enabled", benchmark_cfg.get("enabled", "")),
        ("benchmark_tasks_file", benchmark_summary.get("tasks_file", benchmark_cfg.get("tasks_file", ""))),
        ("benchmark_selected_tasks_file", benchmark_summary.get("selected_tasks_file", "")),
        ("benchmark_num_selected_tasks", benchmark_summary.get("num_selected_tasks", selected_tasks_summary.get("num_tasks", ""))),
        ("benchmark_lm_eval_limit", benchmark_summary.get("lm_eval_limit", "")),
        ("benchmark_include_belebele", selected_tasks_summary.get("include_belebele", benchmark_cfg.get("include_belebele", ""))),
        ("benchmark_include_flores", selected_tasks_summary.get("include_flores", benchmark_cfg.get("include_flores", ""))),
        ("benchmark_include_sib200", selected_tasks_summary.get("include_sib200", benchmark_cfg.get("include_sib200", ""))),
        ("benchmark_include_taxi1500", selected_tasks_summary.get("include_taxi1500", benchmark_cfg.get("include_taxi1500", ""))),
        ("benchmark_num_belebele", selected_tasks_summary.get("num_belebele", "")),
        ("benchmark_num_flores", selected_tasks_summary.get("num_flores", "")),
        ("benchmark_num_sib200", selected_tasks_summary.get("num_sib200", "")),
        ("benchmark_num_taxi1500", selected_tasks_summary.get("num_taxi1500", "")),
    ]
    benchmark_run_rows = [{"run_id": run.get("run_id", ""), "evaluated_checkpoint": run.get("evaluated_checkpoint", ""), "num_tasks_expected": run.get("num_tasks_expected", ""), "num_tasks_seen": run.get("num_tasks_seen", ""), "missing_tasks": ",".join(str(x) for x in run.get("missing_tasks", [])) if isinstance(run.get("missing_tasks"), list) else ""} for run in benchmark_summary.get("runs", []) if isinstance(run, dict)]
    return {"metadata_rows": metadata_rows, "language_rows": language_rows, "benchmark_run_rows": benchmark_run_rows}


def collect_run_configs(summary_rows: list[dict[str, str]], runs_root: Path, profile_payload: dict[str, Any], logs_root: Path | None) -> list[dict[str, Any]]:
    training_cfg = profile_payload.get("training", {}) if isinstance(profile_payload, dict) else {}
    train_cfg = training_cfg.get("train", {}) if isinstance(training_cfg, dict) else {}
    eval_cfg = training_cfg.get("eval", {}) if isinstance(training_cfg, dict) else {}
    lora_cfg = training_cfg.get("lora", {}) if isinstance(training_cfg, dict) else {}
    profile_runs = training_cfg.get("runs", []) if isinstance(training_cfg, dict) else []
    profile_run_map = {str(run.get("id", "")).strip(): run for run in profile_runs if isinstance(run, dict) and str(run.get("id", "")).strip()}
    rows: list[dict[str, Any]] = []
    for summary_row in summary_rows:
        run_id = str(summary_row.get("run_id", "")).strip()
        if not run_id:
            continue
        run_dir = runs_root / run_id
        profile_run = profile_run_map.get(run_id, {})
        adapter_cfg = safe_json_load(run_dir / "adapter_config.json")
        lineage = safe_json_load(run_dir / "model_tokenizer_lineage.json")
        train_results = safe_json_load(run_dir / "train_results.json")
        eval_results = safe_json_load(run_dir / "eval_results.json")
        trainable_stats = resolve_trainable_stats(run_id, run_dir, logs_root)
        readme_hparams = load_readme_hparams(run_dir / "README.md")
        trainer_state = load_trainer_state_summary(run_dir / "trainer_state.json")
        target_modules = adapter_cfg.get("target_modules")
        if isinstance(target_modules, list):
            target_modules = ",".join(str(item) for item in target_modules)
        rows.append({"run_id": run_id, "kind": summary_row.get("kind", ""), "peft_type": adapter_cfg.get("peft_type", ""), "base_model": adapter_cfg.get("base_model_name_or_path", lineage.get("model_name_or_path", "")), "tokenized_path": lineage.get("tokenized_path", ""), "run_name": run_id, "train_batch_size": trainer_state.get("train_batch_size", readme_hparams.get("train_batch_size", train_cfg.get("batch_size", ""))), "eval_batch_size": readme_hparams.get("eval_batch_size", eval_cfg.get("batch_size", "")), "grad_accum": train_cfg.get("grad_accum", ""), "learning_rate": readme_hparams.get("learning_rate", train_cfg.get("learning_rate", "")), "max_steps": trainer_state.get("max_steps", readme_hparams.get("max_steps", train_cfg.get("max_steps", ""))), "save_steps": trainer_state.get("save_steps", train_cfg.get("save_steps", "")), "eval_steps": trainer_state.get("eval_steps", eval_cfg.get("eval_steps", "")), "bf16": train_cfg.get("bf16", ""), "fp16": train_cfg.get("fp16", ""), "lr_scheduler_type": readme_hparams.get("lr_scheduler_type", train_cfg.get("lr_scheduler_type", "linear")), "warmup_steps": train_cfg.get("warmup_steps", ""), "seed": readme_hparams.get("seed", train_cfg.get("seed", "")), "lora_rank": adapter_cfg.get("r", lora_cfg.get("rank", "")), "lora_alpha": adapter_cfg.get("lora_alpha", lora_cfg.get("alpha", "")), "lora_dropout": adapter_cfg.get("lora_dropout", lora_cfg.get("dropout", "")), "target_modules": target_modules or "", "num_experts": adapter_cfg.get("num_experts", ""), "max_threshold": adapter_cfg.get("max_threshold", ""), "trainable_params": trainable_stats.get("trainable_params", ""), "all_params": trainable_stats.get("all_params", ""), "trainable_pct": trainable_stats.get("trainable_pct", ""), "profile_run_kind": profile_run.get("kind", ""), "profile_run_params_json": safe_json_dumps(profile_run.get("params", {})), "adapter_config_json": safe_json_dumps(adapter_cfg), "train_loss": train_results.get("train_loss", summary_row.get("train_loss", "")), "eval_loss": summary_row.get("eval_loss", eval_results.get("eval_loss", ""))})
    return rows


def collect_benchmark_raw_rows(eval_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], summary_rows: list[dict[str, str]], resource_map: dict[str, str]) -> list[dict[str, Any]]:
    kind_map = {str(row.get("run_id", "")).strip(): str(row.get("kind", "")).strip() for row in summary_rows if str(row.get("run_id", "")).strip()}
    eval_tasks = {str(row.get("task", "")).strip() for row in eval_rows if str(row.get("task", "")).strip()}
    seen: set[tuple[str, str, str, float]] = set()
    rows: list[dict[str, Any]] = []
    for source_row in [*baseline_rows, *eval_rows]:
        run_id, task, metric, value = str(source_row.get("run_id", "")).strip(), str(source_row.get("task", "")).strip(), str(source_row.get("metric", "")).strip(), source_row.get("value")
        if not run_id or not task or not isinstance(value, (int, float)):
            continue
        if run_id.startswith("baseline::") and eval_tasks and task not in eval_tasks:
            continue
        dedupe_key = (run_id, task, metric, float(value))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        benchmark, language, metric_display = parse_task_identity(task, metric)
        rows.append({"run_id": run_id, "kind": "baseline" if run_id.startswith("baseline::") else kind_map.get(run_id, ""), "benchmark": benchmark, "task": task, "language": language or task, "resource_availability": resource_map.get(language, "unknown") if language else "unknown", "metric": metric_display or metric, "better": metric_direction(task, metric), "value": float(value), "is_baseline": run_id.startswith("baseline::")})
    rows.sort(key=lambda row: (str(row["benchmark"]), str(row["language"]), str(row["metric"]), str(row["run_id"])))
    return rows


def git_value(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True)
    except Exception:
        return ""
    return result.stdout.strip()


def collect_experiment_meta(*, repo_root: Path, profile_name: str, profile_payload: dict[str, Any], summary_rows: list[dict[str, str]], benchmark_summary_path: Path, output_xlsx: Path) -> list[tuple[str, Any]]:
    training_cfg = profile_payload.get("training", {}) if isinstance(profile_payload, dict) else {}
    benchmark_summary = safe_json_load(benchmark_summary_path)
    llama_factory_root = repo_root / "LLaMA-Factory"
    return [
        ("report_generated_utc", dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"),
        ("profile_name", profile_name),
        ("output_root", profile_payload.get("output_root", "")),
        ("report_path", str(output_xlsx)),
        ("model_name_or_path", training_cfg.get("model_name_or_path", "")),
        ("num_runs_in_summary", len(summary_rows)),
        ("root_repo_commit", git_value(repo_root, "rev-parse", "HEAD")),
        ("root_repo_branch", git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")),
        ("root_repo_dirty", "true" if git_value(repo_root, "status", "--porcelain") else "false"),
        ("llama_factory_commit", git_value(llama_factory_root, "rev-parse", "HEAD") if llama_factory_root.exists() else ""),
        ("llama_factory_branch", git_value(llama_factory_root, "rev-parse", "--abbrev-ref", "HEAD") if llama_factory_root.exists() else ""),
        ("llama_factory_dirty", "true" if llama_factory_root.exists() and git_value(llama_factory_root, "status", "--porcelain") else "false"),
        ("benchmark_summary_path", str(benchmark_summary_path)),
        ("benchmark_selected_tasks_file", benchmark_summary.get("selected_tasks_file", "")),
        ("benchmark_lm_eval_limit", benchmark_summary.get("lm_eval_limit", "")),
    ]


def write_configs_sheet(ws, config_rows: list[dict[str, Any]]) -> None:
    headers = ["run_id", "kind", "peft_type", "base_model", "tokenized_path", "run_name", "train_batch_size", "eval_batch_size", "grad_accum", "learning_rate", "max_steps", "save_steps", "eval_steps", "bf16", "fp16", "lr_scheduler_type", "warmup_steps", "seed", "lora_rank", "lora_alpha", "lora_dropout", "target_modules", "num_experts", "max_threshold", "trainable_params", "all_params", "trainable_pct", "profile_run_kind", "profile_run_params_json", "adapter_config_json", "train_loss", "eval_loss"]
    ws.append(headers)
    style_header(ws, header_row=1)
    for row in config_rows:
        ws.append([row.get(header, "") for header in headers])
    widths = {1: 18, 2: 14, 3: 14, 4: 28, 5: 36, 6: 36, 7: 14, 8: 14, 9: 12, 10: 14, 11: 10, 12: 10, 13: 10, 14: 8, 15: 8, 16: 18, 17: 12, 18: 10, 19: 10, 20: 12, 21: 36, 22: 12, 23: 14, 24: 16, 25: 16, 26: 14, 27: 16, 28: 28, 29: 28, 30: 52, 31: 52, 32: 12, 33: 12}
    for col_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.freeze_panes = "A2"


def write_profile_sheet(ws, profile_payload: dict[str, Any]) -> None:
    ws.append(["key", "value"])
    style_header(ws, header_row=1)
    for key, value in flatten_payload(profile_payload):
        ws.append([key, value])
    ws.column_dimensions["A"].width = 48
    ws.column_dimensions["B"].width = 120
    ws.freeze_panes = "A2"


def write_data_overview_sheet(ws, overview: dict[str, Any]) -> None:
    ws.append(["key", "value"])
    style_header(ws, header_row=1)
    for key, value in overview.get("metadata_rows", []):
        ws.append([key, value])
    start_row = ws.max_row + 2
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=6)
    ws.cell(start_row, 1, "Per-Language Data")
    ws.cell(start_row, 1).fill = SECTION_FILL
    ws.cell(start_row, 1).font = SECTION_FONT
    lang_headers = [
        "language",
        "selected_docs",
        "selected_shards",
        "selected_bytes",
        "train_docs",
        "validation_docs",
        "tokenized_examples",
        "tokenized_train_examples",
        "tokenized_validation_examples",
        "tokenized_tokens",
        "tokenized_train_tokens",
        "tokenized_validation_tokens",
    ]
    for idx, header in enumerate(lang_headers, start=1):
        ws.cell(start_row + 1, idx, header)
    style_header(ws, header_row=start_row + 1)
    for row in overview.get("language_rows", []):
        ws.append([row.get(header, "") for header in lang_headers])
    start_row = ws.max_row + 2
    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=5)
    ws.cell(start_row, 1, "Benchmark Coverage By Run")
    ws.cell(start_row, 1).fill = SECTION_FILL
    ws.cell(start_row, 1).font = SECTION_FONT
    run_headers = ["run_id", "evaluated_checkpoint", "num_tasks_expected", "num_tasks_seen", "missing_tasks"]
    for idx, header in enumerate(run_headers, start=1):
        ws.cell(start_row + 1, idx, header)
    style_header(ws, header_row=start_row + 1)
    for row in overview.get("benchmark_run_rows", []):
        ws.append([row.get(header, "") for header in run_headers])
    for col, width in {"A": 28, "B": 16, "C": 16, "D": 16, "E": 18, "F": 20, "G": 18, "H": 18, "I": 18, "J": 18, "K": 18, "L": 18}.items():
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"


def write_benchmark_raw_sheet(ws, rows: list[dict[str, Any]]) -> None:
    headers = ["run_id", "kind", "benchmark", "task", "language", "resource_availability", "metric", "better", "value", "is_baseline"]
    ws.append(headers)
    style_header(ws, header_row=1)
    for row in rows:
        ws.append([row.get(header, "") for header in headers])
    for col, width in {"A": 22, "B": 14, "C": 14, "D": 32, "E": 16, "F": 20, "G": 18, "H": 10, "I": 12, "J": 12}.items():
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"


def write_experiment_meta_sheet(ws, rows: list[tuple[str, Any]]) -> None:
    ws.append(["key", "value"])
    style_header(ws, header_row=1)
    for key, value in rows:
        ws.append([key, value])
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 120
    ws.freeze_panes = "A2"
