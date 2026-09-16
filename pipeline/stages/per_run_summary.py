"""Atomic per-run summary writes with a deterministic aggregate view."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

from ..runtime import _dump_json


def write_per_run_summary(
    *,
    per_run_path: Path,
    aggregate_path: Path,
    metadata: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    """Write one run atomically, then rebuild the shared summary under a file lock."""
    per_run_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = aggregate_path.with_name(f".{aggregate_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        _dump_json(per_run_path, {"metadata": metadata, "run": run})

        payloads: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(per_run_path.parent.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot aggregate unreadable run summary: {path}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("run"), dict):
                raise ValueError(f"Invalid run summary: {path}")
            if payload["run"].get("run_id") != path.stem:
                raise ValueError(f"Run ID does not match summary filename: {path}")
            payloads.append((path, payload))

        first_metadata = payloads[0][1].get("metadata", {}) if payloads else {}
        aggregate = dict(first_metadata) if isinstance(first_metadata, dict) else {}
        aggregate["runs"] = sorted(
            (payload["run"] for _, payload in payloads),
            key=lambda row: str(row.get("run_id", "")),
        )
        aggregate["per_run_summaries"] = [str(path.resolve()) for path, _ in payloads]
        _dump_json(aggregate_path, aggregate)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return aggregate
