"""Generic artifact helpers for sweep validation notebooks.

This module intentionally contains only path, JSON, adapter, and trainer-state
loading utilities. The actual validation criteria stay in the notebook.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_path(value: Any, root: Path) -> Path | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if text.startswith("per_task_json:"):
        text = text.split(":", 2)[-1]
    path = Path(text)
    return path if path.is_absolute() else root / path


def result_path_kind(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "missing"
    return "per_task_json" if str(value).startswith("per_task_json:") else "file"


def source_root_from_result(result_path: Path | None) -> Path | None:
    if result_path is None or "benchmarks" not in result_path.parts:
        return None
    return Path(*result_path.parts[: result_path.parts.index("benchmarks")])


def run_dir(source_root: Path | None, run_id: str) -> Path | None:
    return None if source_root is None else source_root / "runs" / run_id


def adapter_dir_from_result(result_path: Path | None, source_root: Path | None, run_id: str) -> Path | None:
    if result_path is not None:
        match = re.search(r"__runs__(?P<run>.*?)__(?P<ckpt>checkpoint-\d+_adapter)", str(result_path))
        if match and source_root is not None:
            return source_root / "runs" / run_id / match.group("ckpt")
    rd = run_dir(source_root, run_id)
    if rd is None or not rd.exists():
        return None
    if (rd / "adapter_config.json").exists():
        return rd
    adapters = sorted(
        rd.glob("checkpoint-*_adapter"),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1))
        if re.search(r"checkpoint-(\d+)", p.name)
        else -1,
    )
    return adapters[-1] if adapters else None


def selected_checkpoint_step(adapter_dir: Path | None) -> int | float:
    if adapter_dir is None:
        return float("nan")
    match = re.search(r"checkpoint-(\d+)_adapter", adapter_dir.name)
    return int(match.group(1)) if match else float("nan")


def norm_targets(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                return norm_targets(json.loads(value))
            except Exception:
                pass
        return sorted({part.strip() for part in value.split(",") if part.strip()})
    if isinstance(value, (list, tuple, set)):
        return sorted({str(part).strip() for part in value if str(part).strip()})
    return []


def load_training_metrics(rd: Path | None) -> dict[str, Any]:
    if rd is None or not rd.exists():
        return {}
    train_results = read_json(rd / "train_results.json") or read_json(rd / "all_results.json")
    state = read_json(rd / "trainer_state.json")
    out: dict[str, Any] = {
        "train_results_present": bool(train_results),
        "trainer_state_present": bool(state),
    }
    for key in ("train_loss", "train_runtime", "num_input_tokens_seen", "epoch", "train_steps_per_second"):
        if key in train_results:
            out[key] = train_results[key]
    for key in ("global_step", "max_steps", "train_batch_size", "num_train_epochs"):
        if key in state:
            out[key] = state[key]
    if "num_input_tokens_seen" not in out and "num_input_tokens_seen" in state:
        out["num_input_tokens_seen"] = state["num_input_tokens_seen"]
    return out
