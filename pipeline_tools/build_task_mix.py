#!/usr/bin/env python3
"""Build normalized multitask SFT datasets for pipeline task-mix profiles."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (repo_root / path)


def _label_map(task_name: str, label: Any) -> str:
    value = int(label)
    if task_name == "cola":
        return ["unacceptable", "acceptable"][value]
    if task_name == "sst2":
        return ["negative", "positive"][value]
    if task_name in {"mrpc", "qqp"}:
        return ["not_duplicate", "duplicate"][value]
    if task_name == "mnli":
        return ["entailment", "neutral", "contradiction"][value]
    if task_name in {"qnli", "rte"}:
        return ["entailment", "not_entailment"][value]
    raise ValueError(f"Unsupported label mapping for task {task_name}")


def _normalize_glue_example(task_name: str, example: dict[str, Any]) -> dict[str, Any]:
    if task_name == "cola":
        return {
            "instruction": (
                "Determine whether the following sentence is grammatically acceptable. "
                "Respond with exactly one label: acceptable or unacceptable."
            ),
            "input": str(example["sentence"]),
            "output": _label_map(task_name, example["label"]),
        }
    if task_name == "sst2":
        return {
            "instruction": (
                "Classify the sentiment of the following sentence. "
                "Respond with exactly one label: positive or negative."
            ),
            "input": str(example["sentence"]),
            "output": _label_map(task_name, example["label"]),
        }
    if task_name == "mrpc":
        return {
            "instruction": (
                "Determine whether the two sentences are paraphrases of each other. "
                "Respond with exactly one label: duplicate or not_duplicate."
            ),
            "input": f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}",
            "output": _label_map(task_name, example["label"]),
        }
    if task_name == "qqp":
        return {
            "instruction": (
                "Determine whether the two questions ask the same thing. "
                "Respond with exactly one label: duplicate or not_duplicate."
            ),
            "input": f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}",
            "output": _label_map(task_name, example["label"]),
        }
    if task_name == "mnli":
        return {
            "instruction": (
                "Classify the relationship between the premise and the hypothesis. "
                "Respond with exactly one label: entailment, neutral, or contradiction."
            ),
            "input": f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}",
            "output": _label_map(task_name, example["label"]),
        }
    if task_name == "qnli":
        return {
            "instruction": (
                "Decide whether the sentence answers the question. "
                "Respond with exactly one label: entailment or not_entailment."
            ),
            "input": f"Question: {example['question']}\nSentence: {example['sentence']}",
            "output": _label_map(task_name, example["label"]),
        }
    if task_name == "rte":
        return {
            "instruction": (
                "Determine whether the premise entails the hypothesis. "
                "Respond with exactly one label: entailment or not_entailment."
            ),
            "input": f"Premise: {example['sentence1']}\nHypothesis: {example['sentence2']}",
            "output": _label_map(task_name, example["label"]),
        }
    if task_name == "stsb":
        return {
            "instruction": (
                "Rate the semantic similarity of the two sentences on a scale from 0.0 to 5.0. "
                "Respond with a numeric score only."
            ),
            "input": f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}",
            "output": f"{float(example['label']):.1f}",
        }
    raise ValueError(f"Unsupported GLUE task: {task_name}")


def _normalize_local_alpaca(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "instruction": str(example.get("instruction", "")),
        "input": str(example.get("input", "")),
        "output": str(example.get("output", "")),
    }


def _load_source_records(repo_root: Path, task_cfg: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    source = task_cfg.get("source", {})
    source_type = str(source.get("type", "")).strip()
    if source_type == "hf":
        from datasets import load_dataset

        path = str(source.get("path", "")).strip()
        name = source.get("name")
        split_name = str(source.get(f"{split}_split", source.get("split", split))).strip()
        if not path or not split_name:
            raise ValueError(f"Task {task_cfg.get('task_name')} is missing HF source path/split for {split}")
        try:
            dataset = load_dataset(path=path, name=name, split=split_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load Hugging Face dataset path={path!r} name={name!r} split={split_name!r}. "
                "Ensure network access or local dataset cache is available."
            ) from exc
        return [dict(row) for row in dataset]

    if source_type == "json":
        path = _resolve_path(repo_root, str(source.get("path", "")).strip())
        if not path.exists():
            raise FileNotFoundError(f"Task source not found: {path}")
        payload = _load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list JSON payload in {path}")
        filter_key = str(source.get("filter_key", "")).strip()
        if filter_key:
            filter_value = source.get("filter_value")
            payload = [row for row in payload if isinstance(row, dict) and row.get(filter_key) == filter_value]
        return [dict(row) for row in payload if isinstance(row, dict)]

    raise ValueError(f"Unsupported task source type: {source_type!r}")


def _normalize_records(task_cfg: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_name = str(task_cfg["task_name"])
    builder = str(task_cfg.get("builder", "")).strip()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if builder == "local_alpaca":
            record = _normalize_local_alpaca(row)
        elif builder == task_name:
            record = _normalize_glue_example(task_name, row)
        else:
            raise ValueError(f"Unsupported builder {builder!r} for task {task_name}")
        record.update(
            {
                "task_id": int(task_cfg["task_id"]),
                "task_name": task_name,
                "task_family": str(task_cfg["task_family"]),
                "source_dataset": str(task_cfg.get("source_dataset", task_name)),
            }
        )
        normalized.append(record)
    return normalized


def _split_records(
    task_cfg: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_source = task_cfg.get("eval_source")
    if eval_source:
        raise ValueError("Inline eval_source is not supported; use source.<eval_split> instead.")

    source = task_cfg.get("source", {})
    source_type = str(source.get("type", "")).strip()
    if source_type == "hf" and str(source.get("validation_split", "")).strip():
        train_records = records
        return train_records, []

    holdout_examples = int(task_cfg.get("holdout_examples", 0) or 0)
    if holdout_examples <= 0:
        return records, []

    rng = random.Random(seed + int(task_cfg["task_id"]))
    shuffled = list(records)
    rng.shuffle(shuffled)
    eval_records = shuffled[:holdout_examples]
    train_records = shuffled[holdout_examples:]
    return train_records, eval_records


def _sample_to_target(rows: list[dict[str, Any]], *, target: int, seed: int) -> list[dict[str, Any]]:
    if target <= 0:
        return list(rows)
    rng = random.Random(seed)
    if len(rows) >= target:
        picked = list(rows)
        rng.shuffle(picked)
        return picked[:target]
    expanded = list(rows)
    while len(expanded) < target:
        expanded.append(dict(rng.choice(rows)))
    rng.shuffle(expanded)
    return expanded[:target]


def _build_dataset(spec: dict[str, Any], repo_root: Path, output_dir: Path) -> dict[str, Any]:
    mix_name = str(spec.get("mix_name", "")).strip()
    if not mix_name:
        raise ValueError("Spec missing mix_name")
    tasks = spec.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Spec must contain a non-empty tasks list")

    seed = int(spec.get("seed", 42))
    dataset_dir = output_dir / "dataset"
    train_dataset_name = str(spec.get("train_dataset_name", f"{mix_name}_train")).strip()
    eval_dataset_name = str(spec.get("eval_dataset_name", f"{mix_name}_eval")).strip()
    train_file = dataset_dir / f"{train_dataset_name}.jsonl"
    eval_file = dataset_dir / f"{eval_dataset_name}.jsonl"
    task_map_path = output_dir / "task_map.json"

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    task_entries: list[dict[str, Any]] = []

    for task_cfg in tasks:
        task_name = str(task_cfg.get("task_name", "")).strip()
        if not task_name:
            raise ValueError("Task entry missing task_name")
        task_id = int(task_cfg["task_id"])
        raw_train_rows = _load_source_records(repo_root, task_cfg, split="train")
        normalized_train_rows = _normalize_records(task_cfg, raw_train_rows)
        split_train_rows, split_eval_rows = _split_records(task_cfg, normalized_train_rows, seed=seed)

        source = task_cfg.get("source", {})
        if str(source.get("type", "")).strip() == "hf" and str(source.get("validation_split", "")).strip():
            raw_eval_rows = _load_source_records(repo_root, task_cfg, split="validation")
            split_eval_rows = _normalize_records(task_cfg, raw_eval_rows)

        train_target = int(task_cfg.get("train_target_examples", len(split_train_rows)) or 0)
        eval_target = int(task_cfg.get("eval_target_examples", len(split_eval_rows)) or 0)
        task_seed = seed + task_id * 17
        sampled_train_rows = _sample_to_target(split_train_rows, target=train_target, seed=task_seed)
        sampled_eval_rows = _sample_to_target(split_eval_rows, target=eval_target, seed=task_seed + 1)
        if not sampled_train_rows:
            raise ValueError(f"Task {task_name} produced zero train rows")
        if not sampled_eval_rows:
            raise ValueError(f"Task {task_name} produced zero eval rows")

        train_rows.extend(sampled_train_rows)
        eval_rows.extend(sampled_eval_rows)
        task_entries.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "task_family": str(task_cfg["task_family"]),
                "benchmark_task": str(task_cfg.get("benchmark_task", task_name)),
                "train_rows": len(sampled_train_rows),
                "eval_rows": len(sampled_eval_rows),
            }
        )

    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 1).shuffle(eval_rows)

    dataset_info = {
        train_dataset_name: {
            "file_name": train_file.name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        eval_dataset_name: {
            "file_name": eval_file.name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
    }

    _write_jsonl(train_file, train_rows)
    _write_jsonl(eval_file, eval_rows)
    _dump_json(dataset_dir / "dataset_info.json", dataset_info)
    _dump_json(task_map_path, {"mix_name": mix_name, "tasks": task_entries})

    manifest = {
        "source_type": "task_mix",
        "mix_name": mix_name,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_name": train_dataset_name,
        "eval_dataset_name": eval_dataset_name,
        "train_file": str(train_file.resolve()),
        "eval_file": str(eval_file.resolve()),
        "task_map_path": str(task_map_path.resolve()),
        "task_count": len(task_entries),
        "selection": {"tasks": [entry["task_name"] for entry in task_entries]},
    }
    _dump_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build normalized multitask datasets.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    spec_path = Path(args.spec).resolve()
    output_dir = Path(args.output_dir).resolve()

    spec = _load_json(spec_path)
    manifest = _build_dataset(spec, repo_root, output_dir)
    print(json.dumps({"ok": True, "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
