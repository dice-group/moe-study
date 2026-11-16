from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .common import is_finite_number, nonfinite_display, safe_float


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _skip_metric(task: str, metric: str) -> bool:
    return False


def _rglob_following_run_symlinks(root: Path, pattern: str) -> list[Path]:
    """Find artifacts, including those below top-level consolidated run symlinks."""
    paths = set(root.rglob(pattern))
    for child in root.iterdir():
        if child.is_symlink() and child.is_dir():
            paths.update(child.rglob(pattern))
    return sorted(paths)


def iter_eval_rows(benchmark_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not benchmark_dir.exists():
        return rows

    def ingest_payload(payload: Any, *, run_id: str) -> None:
        if isinstance(payload, dict) and isinstance(payload.get("task"), str):
            value = payload.get("value")
            if isinstance(value, (int, float)):
                task = payload["task"]
                metric = str(payload.get("metric", ""))
                if _skip_metric(task, metric):
                    return
                if not is_finite_number(value):
                    rows.append(
                        {
                            "run_id": run_id,
                            "task": task,
                            "metric": metric,
                            "value": None,
                            "display_value": nonfinite_display(value),
                            "wandb_project": str(payload.get("wandb_project", "")),
                            "wandb_prefix": str(payload.get("wandb_prefix", "")),
                        }
                    )
                    return
                rows.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "metric": metric,
                        "value": float(value),
                        "wandb_project": str(payload.get("wandb_project", "")),
                        "wandb_prefix": str(payload.get("wandb_prefix", "")),
                    }
                )
            return

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            return
        for task, metrics in results.items():
            if not isinstance(task, str) or not isinstance(metrics, dict):
                continue
            if task.startswith("belebele_"):
                candidates = ("acc_norm,none", "acc,none")
            elif task.startswith("flores_en_perplexity_"):
                candidates = ("byte_perplexity,none", "word_perplexity,none", "bits_per_byte,none")
            elif task.startswith("sib200_") or task.startswith("taxi1500_"):
                candidates = ("f1,none", "acc_norm,none", "acc,none")
            else:
                candidates = ("acc_norm,none", "acc,none", "byte_perplexity,none", "word_perplexity,none")
            for candidate in candidates:
                val = metrics.get(candidate)
                if isinstance(val, (int, float)):
                    if _skip_metric(task, candidate):
                        continue
                    if not is_finite_number(val):
                        rows.append(
                            {
                                "run_id": run_id,
                                "task": task,
                                "metric": candidate,
                                "value": None,
                                "display_value": nonfinite_display(val),
                                "wandb_project": "",
                                "wandb_prefix": "",
                            }
                        )
                        continue
                    rows.append(
                        {
                            "run_id": run_id,
                            "task": task,
                            "metric": candidate,
                            "value": float(val),
                            "wandb_project": "",
                            "wandb_prefix": "",
                        }
                    )
                    if task.startswith("belebele_"):
                        break

    for path in _rglob_following_run_symlinks(benchmark_dir, "*_lm_eval.jsonl"):
        run_id = path.parent.name
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ingest_payload(payload, run_id=run_id)
            continue
        if path.is_dir():
            for result_file in sorted(path.rglob("results_*.json")):
                try:
                    payload = json.loads(result_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                ingest_payload(payload, run_id=run_id)

    for json_file in _rglob_following_run_symlinks(
        benchmark_dir, "with_language_ids_*.json"
    ) + _rglob_following_run_symlinks(benchmark_dir, "no_language_ids.json"):
        if not json_file.is_file():
            continue
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ingest_payload(payload, run_id=json_file.parent.name)
    return rows


def load_resource_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    train_seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            script = str(row.get("script", "")).strip()
            if not code or not script:
                continue
            key = f"{code}_{script}"
            split = str(row.get("split", "")).strip().lower()
            availability = str(row.get("resource_availability", "")).strip().upper() or "unknown"
            if split == "train":
                out[key] = availability
                train_seen.add(key)
                continue
            if key not in train_seen and key not in out:
                out[key] = availability
    return out


def sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "na"


def model_slug(model_name_or_path: str) -> str:
    value = model_name_or_path.strip()
    if "/" in value:
        value = value.split("/")[-1]
    value = value.replace("Llama", "l").replace("llama", "l").replace(".", "-")
    return sanitize_slug(value)


def load_baseline_rows(baseline_cache_root: Path, model_name_or_path: str) -> list[dict[str, Any]]:
    slug = model_slug(model_name_or_path)
    metrics_path = baseline_cache_root / slug / "metrics_long.jsonl"
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        value = payload.get("value")
        task = payload.get("task")
        if not isinstance(task, str):
            continue
        metric = str(payload.get("metric", ""))
        if _skip_metric(task, metric):
            continue
        if not is_finite_number(value):
            rows.append(
                {
                    "run_id": str(payload.get("run_id", f"baseline::{slug}")),
                    "task": task,
                    "metric": metric,
                    "value": None,
                    "display_value": nonfinite_display(value),
                    "wandb_project": str(payload.get("wandb_project", "")),
                    "wandb_prefix": str(payload.get("wandb_prefix", "")),
                }
            )
            continue
        rows.append(
            {
                "run_id": str(payload.get("run_id", f"baseline::{slug}")),
                "task": task,
                "metric": metric,
                "value": float(value),
                "wandb_project": str(payload.get("wandb_project", "")),
                "wandb_prefix": str(payload.get("wandb_prefix", "")),
            }
        )
    return rows


def load_baseline_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = str(row.get("task", "")).strip()
            metric = str(row.get("metric", "")).strip()
            value = safe_float(row.get("value"))
            if not task or not metric or value is None:
                continue
            if _skip_metric(task, metric):
                continue
            model = str(row.get("model_slug", "")).strip() or "baseline"
            rows.append(
                {
                    "run_id": f"baseline::{model}",
                    "task": task,
                    "metric": metric,
                    "value": float(value),
                    "wandb_project": "",
                    "wandb_prefix": "",
                }
            )
    return rows


def resolve_baseline_lm_eval_path(path_or_dir: Path) -> Path | None:
    if not path_or_dir.exists():
        return None
    if path_or_dir.is_file() and path_or_dir.suffix == ".json":
        return path_or_dir
    candidates = sorted(path_or_dir.rglob("*_baseline_lm_eval_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_baseline_rows_from_lm_eval_json(path: Path, model_name_or_path: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    source_model = model_name_or_path.strip()
    if not source_model:
        model_args = payload.get("config", {}).get("model_args", {})
        if isinstance(model_args, dict):
            source_model = str(model_args.get("pretrained", "")).strip()
    slug = model_slug(source_model) if source_model else sanitize_slug(path.stem.split("_baseline_lm_eval_", 1)[0])
    run_id = f"baseline::{slug}"
    results = payload.get("results")
    if not isinstance(results, dict):
        return []

    rows: list[dict[str, Any]] = []
    for task, metrics in results.items():
        if not isinstance(task, str) or not isinstance(metrics, dict):
            continue
        if task.startswith("belebele_"):
            candidates = ("acc_norm,none", "acc,none")
        elif task.startswith("flores_en_perplexity_"):
            candidates = ("byte_perplexity,none", "word_perplexity,none", "bits_per_byte,none")
        else:
            candidates = ("acc_norm,none", "acc,none", "byte_perplexity,none", "word_perplexity,none", "bits_per_byte,none")
        for metric in candidates:
            value = metrics.get(metric)
            if isinstance(value, (int, float)):
                if _skip_metric(task, metric):
                    continue
                if not is_finite_number(value):
                    rows.append(
                        {
                            "run_id": run_id,
                            "task": task,
                            "metric": metric,
                            "value": None,
                            "display_value": nonfinite_display(value),
                            "wandb_project": "",
                            "wandb_prefix": "",
                        }
                    )
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "metric": metric,
                        "value": float(value),
                        "wandb_project": "",
                        "wandb_prefix": "",
                    }
                )
                if task.startswith("belebele_"):
                    break
    return rows
