from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


TRAINABLE_RE = re.compile(
    r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)\s*\|\|\s*trainable%:\s*([0-9.]+)"
)


def safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def safe_json_dumps(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        try:
            out = float(value)
            return out if math.isfinite(out) else None
        except ValueError:
            return None
    return None


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def nonfinite_display(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    numeric = float(value)
    if math.isnan(numeric):
        return "nan"
    if math.isinf(numeric):
        return "inf" if numeric > 0 else "-inf"
    return ""


def flatten_payload(payload: Any, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key in sorted(payload):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_payload(payload[key], next_prefix))
        return rows
    if isinstance(payload, list):
        rows.append((prefix, safe_json_dumps(payload)))
        return rows
    rows.append((prefix, "" if payload is None else str(payload)))
    return rows
