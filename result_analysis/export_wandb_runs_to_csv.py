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


def export_project(
    api: wandb.Api,
    entity: str,
    project: str,
    out_dir: Path,
    include_config: bool,
    run_names: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
) -> Path:
    rows = []
    if run_ids:
        for run_id in run_ids:
            try:
                run = api.run(f"{entity}/{project}/{run_id}")
            except Exception:
                continue
            rows.append(_run_to_row(run, project, include_config))
    else:
        filters = None
        if run_names:
            filters = {"name": {"$in": list(run_names)}}
        runs = api.runs(f"{entity}/{project}", filters=filters)
        rows = [_run_to_row(run, project, include_config) for run in runs]
        if run_names:
            names = set(run_names)
            rows = [row for row in rows if row.get("name") in names]
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
    parser.add_argument("--run-names", help="Comma-separated run names to export (filters server-side when possible).")
    parser.add_argument("--run-ids", help="Comma-separated run IDs to export (fetched directly by ID).")
    args = parser.parse_args()

    api = wandb.Api()
    out_dir = Path(args.out_dir)
    run_names = args.run_names.split(",") if args.run_names else None
    run_ids = args.run_ids.split(",") if args.run_ids else None
    for project in args.projects:
        out_path = export_project(
            api,
            args.entity,
            project,
            out_dir,
            include_config=not args.no_config,
            run_names=run_names,
            run_ids=run_ids,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
