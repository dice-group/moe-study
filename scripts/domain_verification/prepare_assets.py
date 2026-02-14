#!/usr/bin/env python3
"""Prepare and validate domain-verification assets.

Checks:
- Base model accessibility
- Hugging Face dataset accessibility
- lm-eval task availability

Optional:
- Materialize datasets into local HF cache via `--download-datasets`
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import AutoConfig


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    subset: Optional[str]
    split: str


DATASETS = [
    DatasetSpec("databricks/databricks-dolly-15k", None, "train"),
    DatasetSpec("gsm8k", "main", "train"),
    DatasetSpec("gsm8k", "main", "test"),
    DatasetSpec("sahil2801/CodeAlpaca-20k", None, "train"),
]

TASKS = [
    "gsm8k",
    "mmlu",
    "mmlu_stem",
    "humaneval",
    "belebele_arb_Arab",
    "belebele_arb_Latn",
    "belebele_azj_Latn",
    "belebele_eng_Latn",
    "belebele_hun_Latn",
    "belebele_ind_Latn",
    "belebele_jpn_Jpan",
    "belebele_kor_Hang",
    "belebele_rus_Cyrl",
    "belebele_vie_Latn",
]


def check_models(models: list[str]) -> list[str]:
    failures: list[str] = []
    for model in models:
        try:
            cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
            print(f"[OK] model={model} model_type={cfg.model_type}")
        except Exception as exc:  # pragma: no cover - external dependency
            msg = f"[ERR] model={model} reason={type(exc).__name__}: {str(exc).splitlines()[0]}"
            print(msg)
            failures.append(msg)
    return failures


def check_datasets(download_datasets: bool) -> list[str]:
    failures: list[str] = []
    for spec in DATASETS:
        kwargs = {
            "path": spec.path,
            "name": spec.subset,
            "split": spec.split,
        }
        try:
            if download_datasets:
                ds = load_dataset(**kwargs)
                print(
                    f"[OK] dataset={spec.path} subset={spec.subset or '-'} split={spec.split} "
                    f"rows={len(ds)} cols={list(ds.column_names)}"
                )
            else:
                ds = load_dataset(**kwargs, streaming=True)
                first = next(iter(ds))
                print(
                    f"[OK] dataset={spec.path} subset={spec.subset or '-'} split={spec.split} "
                    f"sample_keys={sorted(first.keys())}"
                )
        except Exception as exc:  # pragma: no cover - external dependency
            msg = (
                f"[ERR] dataset={spec.path} subset={spec.subset or '-'} split={spec.split} "
                f"reason={type(exc).__name__}: {str(exc).splitlines()[0]}"
            )
            print(msg)
            failures.append(msg)
    return failures


def check_tasks() -> list[str]:
    failures: list[str] = []
    try:
        from lm_eval.tasks import TaskManager
    except Exception as exc:  # pragma: no cover - external dependency
        msg = f"[ERR] lm_eval import failed: {type(exc).__name__}: {exc}"
        print(msg)
        return [msg]

    task_manager = TaskManager()
    all_tasks = set(task_manager.all_tasks)
    for task in TASKS:
        if task in all_tasks:
            print(f"[OK] lm_eval_task={task}")
        else:
            msg = f"[ERR] lm_eval_task_missing={task}"
            print(msg)
            failures.append(msg)

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate domain-verification models, datasets, and eval tasks.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B"],
        help="Model ids/paths to validate.",
    )
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Materialize full datasets into HF cache (default only streaming-sample checks).",
    )
    args = parser.parse_args()

    failures: list[str] = []
    failures.extend(check_models(args.models))
    failures.extend(check_datasets(args.download_datasets))
    failures.extend(check_tasks())

    if failures:
        print("\nValidation finished with errors:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nValidation succeeded. Environment is ready for domain-verification runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
