#!/usr/bin/env python3
"""Export W&B run summary + config to CSV for analysis.
commands:
  - comparison project (summary + config)
  python3 result_analysis/export_wandb_runs_to_csv.py --projects moe-study-comparison
  
  - eval project (metadata + summary only)
  python3 result_analysis/export_wandb_runs_to_csv.py --projects moe-study-comparison_eval --no-config

Outputs one CSV per project with flattened summary/config unless --no-config is set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import wandb

DEFAULT_ENTITY = "dice-nlp"
DEFAULT_PROJECTS = ["moe-study-comparison", "moe-study-comparison_eval"]


def _is_scalar(val: Any) -> bool:
    return isinstance(val, (str, int, float, bool)) or val is None


def _normalize_value(val: Any) -> Any:
    if _is_scalar(val):
        return val
    try:
        return json.dumps(val, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        return str(val)


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in data.items():
        if prefix:
            full_key = f"{prefix}.{key}"
        else:
            full_key = str(key)
        if isinstance(val, dict):
            out.update(_flatten_dict(val, full_key))
        else:
            out[full_key] = _normalize_value(val)
    return out


def _run_to_row(run, project: str, include_config: bool) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "project": project,
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "group": run.group,
        "job_type": run.job_type,
        "created_at": str(getattr(run, "created_at", None)),
        "updated_at": str(getattr(run, "updated_at", None)),
        "tags": ",".join(run.tags or []),
        "url": run.url,
    }

    summary = dict(run.summary)
    config = dict(run.config) if include_config else {}

    row.update({f"summary.{k}": v for k, v in _flatten_dict(summary).items()})
    if include_config:
        row.update({f"config.{k}": v for k, v in _flatten_dict(config).items()})
    return row


def export_project(api: wandb.Api, entity: str, project: str, out_dir: Path, include_config: bool) -> Path:
    runs = api.runs(f"{entity}/{project}")
    rows = [_run_to_row(run, project, include_config) for run in runs]
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{project}_runs.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export W&B run summary/config to CSV.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--projects", nargs="*", default=DEFAULT_PROJECTS)
    parser.add_argument("--no-config", action="store_true", help="Exclude run.config from export.")
    parser.add_argument("--out-dir", default="result_analysis/wandb_exports")
    args = parser.parse_args()

    api = wandb.Api()
    out_dir = Path(args.out_dir)
    for project in args.projects:
        out_path = export_project(api, args.entity, project, out_dir, include_config=not args.no_config)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
