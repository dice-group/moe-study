#!/usr/bin/env python3
"""Fail closed before submitting a full v2 multilingual benchmark panel."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "scripts/eval/latest_checkpoint/lm-evaluation-harness"
EXPECTED = {
    96: {"Belebele": 39, "FLORES": 100, "SIB200": 50, "Taxi1500": 57},
    200: {"Belebele": 64, "FLORES": 166, "SIB200": 83, "Taxi1500": 92},
}


def _family(task: str) -> str:
    if task.startswith("belebele_"):
        return "Belebele"
    if task.startswith("flores_en_perplexity_"):
        return "FLORES"
    if task.startswith("sib200_"):
        return "SIB200"
    if task.startswith("taxi1500_"):
        return "Taxi1500"
    raise ValueError(f"Unknown benchmark task family: {task}")


def _task_path(task: str) -> Path:
    if task.startswith("belebele_"):
        return TASK_ROOT / "lm_eval/tasks/belebele" / f"{task}.yaml"
    if task.startswith("flores_en_perplexity_"):
        return TASK_ROOT / "lm_eval/tasks/flores_en_perplexity/configs" / f"{task}.yaml"
    if task.startswith("sib200_"):
        return TASK_ROOT / "lm_eval/tasks/sib200/configs" / f"{task}.yaml"
    if task.startswith("taxi1500_"):
        return TASK_ROOT / "lm_eval/tasks/taxi1500/configs" / f"{task}.yaml"
    raise ValueError(task)


def _load_tasks(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(panel_dir: Path, panel_size: int, *, instantiate: bool) -> list[str]:
    errors: list[str] = []
    summary_path = panel_dir / f"lm_eval_tasks_{panel_size}_langs_v2.summary.json"
    task_path = panel_dir / f"lm_eval_tasks_{panel_size}_langs_v2.txt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = _load_tasks(task_path)
    expected = EXPECTED[panel_size]

    if len(tasks) != sum(expected.values()):
        errors.append(f"task count={len(tasks)}, expected={sum(expected.values())}")
    if len(tasks) != len(set(tasks)):
        errors.append("task panel contains duplicate task names")
    actual = Counter(_family(task) for task in tasks)
    if dict(actual) != expected:
        errors.append(f"task breakdown={dict(actual)}, expected={expected}")
    if summary.get("task_breakdown") != expected:
        errors.append("summary task breakdown does not match the frozen contract")

    for task in tasks:
        config = _task_path(task)
        if not config.is_file():
            errors.append(f"missing task config: {config}")

    for alias in summary.get("aliases_used", []):
        if alias.get("benchmark") != "Belebele":
            continue
        config = _task_path(f"belebele_{alias['panel']}")
        text = config.read_text(encoding="utf-8") if config.is_file() else ""
        for required in ("fewshot_split: test", "test_split: test", f"dataset_name: {alias['dataset']}"):
            if required not in text:
                errors.append(f"{config.name} is missing required canonical field: {required}")

    if not instantiate:
        return errors

    sys.path.insert(0, str(TASK_ROOT))
    from lm_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager()
    missing = sorted(set(tasks) - set(manager.all_tasks))
    if missing:
        errors.append(f"TaskManager cannot resolve tasks: {missing}")
        return errors
    try:
        task_dict = get_task_dict(tasks, task_manager=manager)
    except Exception as exc:  # pragma: no cover - external task dependencies
        errors.append(f"TaskManager cannot instantiate panel: {type(exc).__name__}: {exc}")
        return errors
    if set(task_dict) != set(tasks):
        errors.append("TaskManager instantiated a task set different from the requested panel")
    for name, task in task_dict.items():
        try:
            docs = task.eval_docs
            if docs is None:
                errors.append(f"{name}: eval_docs is None")
                continue
            next(iter(docs))
        except Exception as exc:  # pragma: no cover - external dataset availability
            errors.append(f"{name}: evaluation documents unavailable: {type(exc).__name__}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=ROOT / "configs/benchmark_panels/v2")
    parser.add_argument("--panels", default="96,200")
    parser.add_argument("--instantiate", action="store_true", help="Load one evaluation document from every task.")
    args = parser.parse_args()

    failed = False
    for raw_size in args.panels.split(","):
        panel_size = int(raw_size)
        errors = validate(args.panel_dir, panel_size, instantiate=args.instantiate)
        if errors:
            failed = True
            print(f"FAIL: v2 {panel_size}-language panel")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS: v2 {panel_size}-language panel")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
