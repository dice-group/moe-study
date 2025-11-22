#!/usr/bin/env python3
"""
Fetches the most recent AdaMoLE / MoLA / MoE-LPR runs from Weights & Biases,
prints summary metrics (first/mid/latest/min/max/mean), and writes everything
to `result_analysis/wandb_snapshot.json`.
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import wandb

PROJECT_PATH = "dice-nlp/moe-study-comparison"
RUN_TAGS = ("adamole", "mola", "moelpr")
REQUIRED_METRICS: Dict[str, List[str]] = {
    "adamole": [
        "train/adamole_aux_loss",
        "train/adamole/threshold_mean",
        "train/adamole/experts_per_token",
    ],
    "mola": [
        "train/mola_aux_loss",
        "train/mola/null_token_pct",
        "train/mola/active_expert_frac",
    ],
    "moelpr": [
        "train/moelpr_aux_loss",
        "train/moelpr/expert_load_cv",
        "train/moelpr/router_entropy",
    ],
}
OPTIONAL_METRICS: Dict[str, List[str]] = {
    "moelpr": ["train/moelpr/original_primary_expert"],
}


@dataclass
class MetricStats:
    latest: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    first: float | None = None
    mid: float | None = None

    @classmethod
    def from_history(cls, values: List[float]) -> "MetricStats":
        if not values:
            return cls()
        mid_idx = len(values) // 2
        return cls(
            latest=values[-1],
            minimum=min(values),
            maximum=max(values),
            mean=statistics.fmean(values),
            first=values[0],
            mid=values[mid_idx],
        )


@dataclass
class RunStatus:
    name: str
    url: str
    state: str
    missing_metrics: List[str]
    summary_keys: List[str]
    metrics: Dict[str, MetricStats] = field(default_factory=dict)


def collect_metric_history(run: wandb.apis.public.Run, metric: str, max_points: int = 200) -> List[float]:
    try:
        history_rows = run.history(keys=[metric], pandas=False)
    except wandb.errors.CommError:
        return []
    values = [
        row.get(metric)
        for row in history_rows[:max_points]
        if metric in row and isinstance(row.get(metric), (int, float))
    ]
    return values


def pick_latest_runs() -> Dict[str, RunStatus]:
    api = wandb.Api()
    runs = api.runs(PROJECT_PATH, order="-created_at")
    latest: Dict[str, RunStatus] = {}

    for run in runs:
        if len(latest) == len(RUN_TAGS):
            break

        name = run.name or run.id
        tags = set(tag.lower() for tag in run.tags or [])
        summary = dict(run.summary)

        for tag in RUN_TAGS:
            if tag in latest:
                continue
            if tag in tags or tag in name.lower():
                missing = [metric for metric in REQUIRED_METRICS[tag] if metric not in summary]
                stats = {}
                for metric in REQUIRED_METRICS[tag]:
                    values = collect_metric_history(run, metric)
                    stats[metric] = MetricStats.from_history(values)
                latest[tag] = RunStatus(
                    name=name,
                    url=run.url,
                    state=run.state,
                    missing_metrics=missing,
                    summary_keys=list(summary.keys()),
                    metrics=stats,
                )
                break

    return latest


def write_snapshot(latest: Dict[str, RunStatus], path: Path) -> None:
    payload = {
        alias: {
            "name": status.name,
            "state": status.state,
            "url": status.url,
            "metrics": {
                metric: status.metrics[metric].__dict__
                for metric in status.metrics
            },
        }
        for alias, status in latest.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    latest = pick_latest_runs()
    write_snapshot(latest, Path("result_analysis/wandb_snapshot.json"))

    for alias in RUN_TAGS:
        if alias not in latest:
            print(f"[WARN] No recent run found for {alias}.")
            continue
        status = latest[alias]
        optional_missing = [metric for metric in OPTIONAL_METRICS.get(alias, []) if metric not in status.summary_keys]
        ok = not status.missing_metrics and status.state == "finished"
        badge = "[OK]" if ok else "[ISSUE]"
        print(f"{badge} {alias.upper()} :: {status.name} ({status.state}) -> {status.url}")
        if status.missing_metrics:
            print(f"    Missing metrics: {', '.join(status.missing_metrics)}")
        else:
            print("    All expected metrics present.")
        for metric, stats in status.metrics.items():
            if stats.latest is None:
                continue
            print(
                f"      - {metric}: first={stats.first:.4g}, mid={stats.mid:.4g}, latest={stats.latest:.4g}, "
                f"min={stats.minimum:.4g}, max={stats.maximum:.4g}, mean={stats.mean:.4g}"
            )
        if optional_missing:
            print(f"    Optional metrics not found: {', '.join(optional_missing)}")


if __name__ == "__main__":
    try:
        main()
    except wandb.errors.CommError as exc:
        print(f"[ERROR] Failed to query Weights & Biases: {exc}", file=sys.stderr)
        sys.exit(1)
