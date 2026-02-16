#!/usr/bin/env python3
"""Build and cache base-model lm-eval results for reuse in comparisons."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "na"


def _model_slug(model_name_or_path: str) -> str:
    value = model_name_or_path.strip()
    if "/" in value:
        value = value.split("/")[-1]
    value = value.replace("Llama", "l").replace("llama", "l").replace(".", "-")
    return _sanitize_slug(value)


def _read_tasks(path: Path) -> list[str]:
    tasks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        tasks.append(raw)
    return tasks


def _pick_metric(task: str, metrics: dict[str, Any]) -> tuple[str, float] | None:
    if not isinstance(metrics, dict):
        return None
    preferred: list[str]
    if task.startswith("belebele_"):
        preferred = ["acc_norm,none", "acc,none"]
    else:
        preferred = ["word_perplexity,none", "byte_perplexity,none", "perplexity,none"]
    for key in preferred:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return key, float(val)
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            return str(key), float(val)
    return None


def _run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def _load_lm_eval_results_payload(output_path: Path) -> tuple[dict[str, Any], Path]:
    if output_path.is_file():
        payload_path = output_path
    elif output_path.is_dir():
        candidates = sorted(output_path.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No results_*.json files found under {output_path}")
        payload_path = candidates[-1]
    else:
        stem = output_path.stem
        candidates = sorted(output_path.parent.glob(f"{stem}_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"lm-eval output path not found: {output_path}")
        payload_path = candidates[-1]
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"lm-eval payload at {payload_path} is not a JSON object")
    return payload, payload_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache base-model lm-eval results.")
    parser.add_argument("--models", required=True, help="Comma-separated HF model IDs.")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--lm-eval-bin", default="lm_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if not models:
        raise SystemExit("No models provided.")
    tasks_file = Path(args.tasks_file)
    tasks = _read_tasks(tasks_file)
    if not tasks:
        raise SystemExit(f"No tasks found in {tasks_file}")
    tasks_csv = ",".join(tasks)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    manifest_rows: list[dict[str, Any]] = []
    for model in models:
        slug = _model_slug(model)
        model_dir = output_root / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        raw_json = model_dir / "lm_eval_output.json"
        metrics_jsonl = model_dir / "metrics_long.jsonl"

        if metrics_jsonl.exists() and not args.force and not args.dry_run:
            print(f"[SKIP] cached baseline exists for {model} at {metrics_jsonl}")
            manifest_rows.append(
                {
                    "model": model,
                    "model_slug": slug,
                    "raw_output_json": str(raw_json.resolve()),
                    "metrics_long_jsonl": str(metrics_jsonl.resolve()),
                    "num_tasks": None,
                }
            )
            continue

        cmd = [
            args.lm_eval_bin,
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model},tokenizer={model}",
            "--tasks",
            tasks_csv,
            "--batch_size",
            str(args.batch_size),
            "--output_path",
            str(raw_json),
            "--seed",
            str(args.seed),
        ]
        if args.extra_args.strip():
            cmd.extend(shlex.split(args.extra_args))
        _run_cmd(cmd, env=env, dry_run=args.dry_run)

        task_count = None
        if not args.dry_run:
            payload, payload_path = _load_lm_eval_results_payload(raw_json)
            results = payload.get("results", {})
            rows: list[dict[str, Any]] = []
            if isinstance(results, dict):
                for task, metrics in results.items():
                    picked = _pick_metric(str(task), metrics if isinstance(metrics, dict) else {})
                    if picked is None:
                        continue
                    metric_name, metric_value = picked
                    rows.append(
                        {
                            "run_id": f"baseline::{slug}",
                            "task": str(task),
                            "metric": metric_name,
                            "value": metric_value,
                            "wandb_project": "",
                            "wandb_prefix": "",
                            "model": model,
                            "model_slug": slug,
                        }
                    )
            with metrics_jsonl.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            task_count = len(rows)
            raw_output_value = str(payload_path.resolve())
        else:
            raw_output_value = str(raw_json.resolve())

        manifest_rows.append(
            {
                "model": model,
                "model_slug": slug,
                "raw_output_json": raw_output_value,
                "metrics_long_jsonl": str(metrics_jsonl.resolve()),
                "num_tasks": task_count,
            }
        )

    manifest = {
        "tasks_file": str(tasks_file.resolve()),
        "num_tasks": len(tasks),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "rows": manifest_rows,
    }
    manifest_path = output_root / "manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] baseline manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
