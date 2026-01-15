import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from lm_eval.utils import simple_parse_args_string
except Exception:
    def simple_parse_args_string(arg: str) -> dict:
        parsed: dict[str, str] = {}
        for item in arg.split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            key, value = item.split("=", 1)
            parsed[key.strip()] = value.strip()
        return parsed


def _infer_step(checkpoint: str) -> int | None:
    match = re.search(r"checkpoint-(\d+)", checkpoint)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    raw = path.read_text()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    entries = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries[-1] if entries else None


def _flatten_results(data: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results = data.get("results")
    if isinstance(results, dict):
        for task, vals in results.items():
            if isinstance(vals, dict):
                for metric, val in vals.items():
                    if isinstance(val, (int, float)):
                        metrics[f"{task}/{metric}"] = float(val)
            elif isinstance(vals, (int, float)):
                metrics[str(task)] = float(vals)
    return metrics


def _sanitize_series_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "default"


def _load_existing_steps(series_path: Path) -> set[int | None]:
    steps: set[int | None] = set()
    if not series_path.exists():
        return steps
    for line in series_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            steps.add(parsed.get("step"))
    return steps


def _write_series_entry(series_path: Path, entry: dict[str, Any]) -> None:
    series_path.parent.mkdir(parents=True, exist_ok=True)
    with series_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True))
        f.write("\n")


def _maybe_log_wandb(entry: dict[str, Any], series_name: str, wandb_args: str, wandb_config_args: str | None) -> None:
    upload = os.environ.get("WANDB_SUMMARY_UPLOAD", "local").strip().lower()
    if upload not in {"1", "true", "yes", "wandb", "both", "remote"}:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[WARN] wandb import failed: {exc}", file=sys.stderr)
        return

    wandb_args_dict = simple_parse_args_string(wandb_args)
    project = wandb_args_dict.get("project")
    summary_suffix = os.environ.get("WANDB_SUMMARY_SUFFIX", "").strip()
    if project and summary_suffix:
        wandb_args_dict["project"] = f"{project}{summary_suffix}"

    base_name = wandb_args_dict.get("name") or "summary"
    wandb_args_dict["name"] = f"{base_name}_{series_name}"

    run = wandb.init(resume="allow", **wandb_args_dict)
    if wandb_config_args:
        cfg = simple_parse_args_string(wandb_config_args)
        if cfg:
            run.config.update(cfg, allow_val_change=True)
    metrics = entry.get("metrics", {})
    step = entry.get("step")
    if step is None:
        run.log(metrics)
    else:
        run.log(metrics, step=step)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-args", required=True)
    parser.add_argument("--wandb-config-args", default=None)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    if not outdir.exists():
        raise SystemExit(f"Output dir not found: {outdir}")

    step = _infer_step(args.checkpoint)

    result_files = sorted([p for p in outdir.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"}])
    if not result_files:
        print(f"[WARN] No results in {outdir}", file=sys.stderr)
        return 0

    for result_path in result_files:
        data = _load_json(result_path)
        if not data:
            continue
        metrics = _flatten_results(data)
        if not metrics:
            continue
        series_name = _sanitize_series_name(result_path.stem)
        series_path = outdir / f"summary_series_{series_name}.jsonl"
        existing_steps = _load_existing_steps(series_path)
        if step in existing_steps:
            continue
        entry = {"step": step, "metrics": metrics}
        _write_series_entry(series_path, entry)
        _maybe_log_wandb(entry, series_name, args.wandb_args, args.wandb_config_args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
