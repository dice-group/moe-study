#!/usr/bin/env python3
"""Fail-closed consolidation of five disjoint final benchmark shards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.checks import check_benchmark
from pipeline.stages.benchmark import _task_family_counts
from pipeline.stages.per_run_summary import write_per_run_summary
from pipeline.training import adapter_checkpoint_provenance


PANEL_SPECS = {
    "v1_200": {
        "tasks": 306,
        "shards": {
            "belebele": 57,
            "flores_en_x": 78,
            "flores_x_en": 78,
            "sib200": 57,
            "taxi1500": 36,
        },
        "breakdown": {"Belebele": 57, "FLORES": 156, "SIB200": 57, "Taxi1500": 36},
    },
    "v2_200": {
        "tasks": 405,
        "shards": {
            "belebele": 64,
            "flores_en_x": 83,
            "flores_x_en": 83,
            "sib200": 83,
            "taxi1500": 92,
        },
        "breakdown": {"Belebele": 64, "FLORES": 166, "SIB200": 83, "Taxi1500": 92},
    },
    "v2_96_plus_english": {
        "tasks": 249,
        "shards": {
            "belebele": 40,
            "flores_en_x": 50,
            "flores_x_en": 50,
            "sib200": 51,
            "taxi1500": 58,
        },
        "breakdown": {"Belebele": 40, "FLORES": 100, "SIB200": 51, "Taxi1500": 58},
    },
    "v2_200_plus_english": {
        "tasks": 408,
        "shards": {
            "belebele": 65,
            "flores_en_x": 83,
            "flores_x_en": 83,
            "sib200": 84,
            "taxi1500": 93,
        },
        "breakdown": {"Belebele": 65, "FLORES": 166, "SIB200": 84, "Taxi1500": 93},
    },
}
TASK_DICT_KEYS = {
    "configs",
    "group_subtasks",
    "higher_is_better",
    "n-samples",
    "n-shot",
    "results",
    "versions",
}


def task_list(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"Missing task list: {path}")
    tasks = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"Duplicate tasks in {path}")
    return tasks


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"Result payload has no results object: {path}")
    return payload


def result_tasks(payload: dict[str, Any]) -> set[str]:
    return {
        str(task)
        for task in payload["results"]
        if isinstance(task, str) and not task.startswith("_")
    }


def result_file(shard_dir: Path) -> tuple[str, Path, dict[str, Any]]:
    standard = sorted(shard_dir.rglob("results_*.json"))
    mtllora = sorted(shard_dir.rglob("no_language_ids.json"))
    if len(standard) == 1 and not mtllora:
        path = standard[0]
        return "lm_eval", path, load_payload(path)
    if len(mtllora) == 1 and not standard:
        path = mtllora[0]
        return "mtllora", path, load_payload(path)
    raise ValueError(
        f"{shard_dir}: expected exactly one standard or MTL-LoRA result, "
        f"found standard={len(standard)}, mtllora={len(mtllora)}"
    )


def lm_eval_dir(path: Path, shard_dir: Path) -> Path:
    for parent in path.parents:
        if parent.name.endswith("_lm_eval.jsonl"):
            if not parent.is_relative_to(shard_dir):
                break
            return parent
    raise ValueError(f"Could not locate lm-eval result directory above {path}")


def merge_mtllora(payloads: list[dict[str, Any]], shards: list[str]) -> dict[str, Any]:
    merged = copy.deepcopy(payloads[0])
    for key in TASK_DICT_KEYS:
        combined: dict[str, Any] = {}
        for payload in payloads:
            value = payload.get(key, {})
            if not isinstance(value, dict):
                raise ValueError(f"MTL-LoRA field {key!r} is not a dictionary")
            overlap = set(combined) & set(value)
            if overlap:
                raise ValueError(f"MTL-LoRA field {key!r} contains duplicate tasks: {sorted(overlap)[:5]}")
            combined.update(value)
        merged[key] = combined
    merged["consolidated_shards"] = shards
    return merged


def checkpoint_step(checkpoint: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)_adapter", checkpoint.name)
    if not match:
        raise ValueError(f"Unexpected adapter checkpoint name: {checkpoint.name}")
    return int(match.group(1))


def consolidate(
    *,
    run_id: str,
    shard_root: Path,
    benchmark_root: Path,
    selected_tasks_file: Path,
    checkpoint: Path,
    panel: str,
    belebele_subshards: list[str] | None = None,
) -> dict[str, Any]:
    spec = PANEL_SPECS[panel]
    expected_task_count = int(spec["tasks"])
    shard_counts = dict(spec["shards"])
    expected_breakdown = dict(spec["breakdown"])
    shard_root = shard_root.resolve()
    benchmark_root = benchmark_root.resolve()
    selected_tasks_file = selected_tasks_file.resolve()
    checkpoint = checkpoint.resolve()
    expected_tasks = task_list(selected_tasks_file)
    if len(expected_tasks) != expected_task_count:
        raise ValueError(f"Expected {expected_task_count} tasks, got {len(expected_tasks)}")
    if _task_family_counts(set(expected_tasks)) != expected_breakdown:
        raise ValueError(f"Unexpected full-panel breakdown: {_task_family_counts(set(expected_tasks))}")

    formats: set[str] = set()
    payloads: list[dict[str, Any]] = []
    files: dict[str, Path] = {}
    shard_tasks: dict[str, set[str]] = {}
    for shard, expected_count in shard_counts.items():
        source_shards = belebele_subshards if shard == "belebele" and belebele_subshards else [shard]
        selected_union: set[str] = set()
        for source_shard in source_shards:
            shard_dir = shard_root / source_shard
            selected = set(task_list(shard_dir / "selected_tasks.txt"))
            result_format, path, payload = result_file(shard_dir)
            observed = result_tasks(payload)
            if observed != selected:
                raise ValueError(
                    f"{run_id}/{source_shard}: selected/result mismatch "
                    f"(missing={sorted(selected - observed)[:5]}, extra={sorted(observed - selected)[:5]})"
                )
            if selected_union & selected:
                raise ValueError(f"{run_id}/{shard}: overlapping subshards")
            selected_union.update(selected)
            formats.add(result_format)
            payloads.append(payload)
            files[source_shard] = path
            shard_tasks[source_shard] = selected
        if len(selected_union) != expected_count:
            raise ValueError(f"{run_id}/{shard}: expected {expected_count} selected tasks, got {len(selected_union)}")

    for left, left_tasks in shard_tasks.items():
        for right, right_tasks in shard_tasks.items():
            if left < right and left_tasks & right_tasks:
                raise ValueError(f"{run_id}: shards {left} and {right} overlap")
    observed_union = set().union(*shard_tasks.values())
    if observed_union != set(expected_tasks):
        raise ValueError(
            f"{run_id}: shard union mismatch "
            f"(missing={sorted(set(expected_tasks) - observed_union)[:5]}, "
            f"extra={sorted(observed_union - set(expected_tasks))[:5]})"
        )
    if len(formats) != 1:
        raise ValueError(f"{run_id}: mixed result formats across shards: {sorted(formats)}")

    provenance = adapter_checkpoint_provenance(
        checkpoint,
        expected_step=checkpoint_step(checkpoint),
        require_direct_save_marker=True,
    )
    if not provenance.get("valid"):
        raise ValueError(f"{run_id}: invalid checkpoint provenance: {provenance}")

    benchmark_root.mkdir(parents=True, exist_ok=True)
    staging = benchmark_root / f".{run_id}.staging.{os.getpid()}"
    staging.mkdir()
    result_format = next(iter(formats))
    if result_format == "lm_eval":
        for shard, path in files.items():
            source_dir = lm_eval_dir(path, shard_root / shard)
            name = source_dir.name.removesuffix("_lm_eval.jsonl")
            shutil.copytree(source_dir, staging / f"{name}_{shard}_lm_eval.jsonl")
    else:
        merged = merge_mtllora(payloads, list(shard_counts))
        (staging / "no_language_ids.json").write_text(
            json.dumps(merged, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )

    target = benchmark_root / run_id
    backup = None
    if target.exists():
        backup = benchmark_root / f"{run_id}.pre_shard_backup.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        target.rename(backup)
    staging.rename(target)

    canonical_selected = benchmark_root / f"selected_tasks.{run_id}.txt"
    canonical_selected.write_text("\n".join(expected_tasks) + "\n", encoding="utf-8")
    selected_summary = benchmark_root / f"selected_tasks_summary.{run_id}.json"
    selected_summary.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "num_selected_tasks": expected_task_count,
                "expected_task_breakdown": expected_breakdown,
                "source": str(selected_tasks_file),
                "consolidated_shards": shard_counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    run = {
        "run_id": run_id,
        "evaluated_checkpoint": str(checkpoint),
        "adapter_provenance": provenance,
        "evaluated_tokenizer": "meta-llama/Llama-3.1-8B",
        "output_dir": str(target),
        "num_output_files": 5 if result_format == "lm_eval" else 1,
        "num_tasks_expected": expected_task_count,
        "num_tasks_seen": expected_task_count,
        "expected_task_breakdown": expected_breakdown,
        "seen_task_breakdown": expected_breakdown,
        "missing_tasks": [],
        "extra_tasks_seen": [],
        "strict_completeness": True,
        "consolidated_shards": shard_counts,
        "result_format": result_format,
    }
    summary_path = benchmark_root / "benchmark_summary.json"
    write_per_run_summary(
        per_run_path=benchmark_root / "run_summaries" / f"{run_id}.json",
        aggregate_path=summary_path,
        metadata={
            "tasks_file": str(selected_tasks_file),
            "selected_tasks_file": str(canonical_selected),
            "selected_tasks_summary": str(selected_summary),
            "resolved_eval_tokenizer": "meta-llama/Llama-3.1-8B",
            "lm_eval_limit": "",
            "embedding_init_manifest": "",
            "num_selected_tasks": expected_task_count,
            "expected_task_breakdown": expected_breakdown,
            "strict_completeness": True,
            "consolidated_from": str(shard_root),
        },
        run=run,
    )
    check_benchmark(
        benchmark_root,
        summary_path,
        dry_run=False,
        check_path=benchmark_root / f"_check.{run_id}.json",
        run_ids={run_id},
    )
    return {
        "status": "complete",
        "panel": panel,
        "run_id": run_id,
        "result_format": result_format,
        "num_tasks": expected_task_count,
        "task_breakdown": expected_breakdown,
        "benchmark_output": str(target),
        "benchmark_summary": str(summary_path),
        "backup": str(backup) if backup else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--selected-tasks-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel", choices=sorted(PANEL_SPECS), default="v1_200")
    parser.add_argument(
        "--belebele-subshards",
        default="",
        help="Comma-separated Belebele shard directories to validate and merge as one family.",
    )
    args = parser.parse_args()
    result = consolidate(
        run_id=args.run_id,
        shard_root=args.shard_root,
        benchmark_root=args.benchmark_root,
        selected_tasks_file=args.selected_tasks_file,
        checkpoint=args.checkpoint,
        panel=args.panel,
        belebele_subshards=[item for item in args.belebele_subshards.split(",") if item],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
