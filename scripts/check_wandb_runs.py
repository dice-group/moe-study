from __future__ import annotations

import sys
from dataclasses import dataclass
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
class RunStatus:
    name: str
    url: str
    state: str
    missing_metrics: List[str]
    summary_keys: List[str]


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
                latest[tag] = RunStatus(
                    name=name,
                    url=run.url,
                    state=run.state,
                    missing_metrics=missing,
                    summary_keys=list(summary.keys()),
                )
                break

    return latest


def main() -> None:
    latest = pick_latest_runs()
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
        if optional_missing:
            print(f"    Optional metrics not found: {', '.join(optional_missing)}")


if __name__ == "__main__":
    try:
        main()
    except wandb.errors.CommError as exc:
        print(f"[ERROR] Failed to query Weights & Biases: {exc}", file=sys.stderr)
        sys.exit(1)
