#!/usr/bin/env python3
"""Export proof pipeline results to a styled Excel workbook.

Inputs:
- summary CSV produced by run_multilingual_proof.py
- benchmark directory containing *_lm_eval.jsonl outputs
- benchmark summary JSON (optional, for metadata)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


HEADER_FILL = PatternFill(fill_type="solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _iter_eval_rows(benchmark_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file in sorted(benchmark_dir.rglob("*_lm_eval.jsonl")):
        run_id = file.parent.name
        for line in file.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # JSONL row format from tests/smoke/lm_eval_checkpoint_local.sh or mock script.
            if isinstance(payload, dict) and isinstance(payload.get("task"), str):
                value = payload.get("value")
                if isinstance(value, (int, float)):
                    rows.append(
                        {
                            "run_id": run_id,
                            "task": payload["task"],
                            "metric": str(payload.get("metric", "")),
                            "value": float(value),
                            "wandb_project": str(payload.get("wandb_project", "")),
                            "wandb_prefix": str(payload.get("wandb_prefix", "")),
                        }
                    )
                continue

            # Fallback for non-JSONL lm-eval object payloads with a "results" map.
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, dict):
                continue
            for task, metrics in results.items():
                if not isinstance(task, str) or not isinstance(metrics, dict):
                    continue
                metric_key = ""
                metric_val: float | None = None
                for candidate in ("acc_norm,none", "word_perplexity,none", "byte_perplexity,none"):
                    val = metrics.get(candidate)
                    if isinstance(val, (int, float)):
                        metric_key = candidate
                        metric_val = float(val)
                        break
                if metric_val is None:
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "metric": metric_key,
                        "value": metric_val,
                        "wandb_project": "",
                        "wandb_prefix": "",
                    }
                )
    return rows


def _autosize(ws) -> None:
    for idx, col_cells in enumerate(ws.columns, start=1):
        max_len = 0
        for cell in col_cells:
            val = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(val))
        ws.column_dimensions[get_column_letter(idx)].width = min(max(10, max_len + 2), 60)


def _style_header(ws, header_row: int = 1) -> None:
    for cell in ws[header_row]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return cleaned.strip("-") or "na"


def _model_slug(model_name_or_path: str) -> str:
    value = model_name_or_path.strip()
    if "/" in value:
        value = value.split("/")[-1]
    value = value.replace("Llama", "l").replace("llama", "l").replace(".", "-")
    return _sanitize_slug(value)


def _load_baseline_rows(baseline_cache_root: Path, model_name_or_path: str) -> list[dict[str, Any]]:
    slug = _model_slug(model_name_or_path)
    metrics_path = baseline_cache_root / slug / "metrics_long.jsonl"
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        value = payload.get("value")
        task = payload.get("task")
        if not isinstance(task, str) or not isinstance(value, (int, float)):
            continue
        rows.append(
            {
                "run_id": str(payload.get("run_id", f"baseline::{slug}")),
                "task": task,
                "metric": str(payload.get("metric", "")),
                "value": float(value),
                "wandb_project": str(payload.get("wandb_project", "")),
                "wandb_prefix": str(payload.get("wandb_prefix", "")),
            }
        )
    return rows


def build_workbook(
    *,
    summary_rows: list[dict[str, str]],
    eval_rows: list[dict[str, Any]],
    benchmark_summary_path: Path | None,
    baseline_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    wb = Workbook()

    ws_summary = wb.active
    ws_summary.title = "summary"
    summary_cols = [
        "run_id",
        "kind",
        "train_loss",
        "eval_loss",
        "language_prior_loss",
        "router_grad_present_frac",
        "expert_load_cv",
        "active_expert_frac",
        "effective_tokens_per_sec",
        "train_runtime",
    ]
    ws_summary.append(summary_cols)
    for row in summary_rows:
        ws_summary.append([row.get(col, "") for col in summary_cols])
    _style_header(ws_summary)
    ws_summary.freeze_panes = "A2"
    _autosize(ws_summary)

    # Wide benchmark table: row=run, col=task.
    combined_eval_rows = list(eval_rows)
    if baseline_rows:
        combined_eval_rows.extend(baseline_rows)

    ws_wide = wb.create_sheet("benchmark_wide")
    runs = sorted({str(r["run_id"]) for r in combined_eval_rows})
    tasks = sorted({str(r["task"]) for r in combined_eval_rows})
    values_by_key: dict[tuple[str, str], float] = {}
    for row in combined_eval_rows:
        values_by_key[(str(row["run_id"]), str(row["task"]))] = float(row["value"])

    ws_wide.append(["run_id", *tasks])
    for run_id in runs:
        ws_wide.append([run_id, *[values_by_key.get((run_id, task), "") for task in tasks]])
    _style_header(ws_wide)
    ws_wide.freeze_panes = "A2"
    _autosize(ws_wide)

    # Long benchmark table for filtering/pivots.
    ws_long = wb.create_sheet("benchmark_long")
    long_cols = ["run_id", "task", "metric", "value", "wandb_project", "wandb_prefix"]
    ws_long.append(long_cols)
    for row in eval_rows:
        ws_long.append([row.get(col, "") for col in long_cols])
    _style_header(ws_long)
    ws_long.freeze_panes = "A2"
    _autosize(ws_long)

    # Comparison sheet vs cached base model (if available).
    ws_cmp = wb.create_sheet("vs_base_model")
    cmp_cols = [
        "run_id",
        "task",
        "run_value",
        "baseline_value",
        "delta",
        "relative_improvement",
        "direction",
    ]
    ws_cmp.append(cmp_cols)
    baseline_by_task = {str(row["task"]): float(row["value"]) for row in baseline_rows}
    for row in eval_rows:
        task = str(row["task"])
        if task not in baseline_by_task:
            continue
        run_val = float(row["value"])
        base_val = baseline_by_task[task]
        if task.startswith("belebele_"):
            delta = run_val - base_val
            rel = (run_val - base_val) / abs(base_val) if base_val != 0 else None
            direction = "higher_better"
        else:
            delta = run_val - base_val
            rel = (base_val - run_val) / abs(base_val) if base_val != 0 else None
            direction = "lower_better"
        ws_cmp.append(
            [
                str(row["run_id"]),
                task,
                run_val,
                base_val,
                delta,
                rel if rel is not None else "",
                direction,
            ]
        )
    _style_header(ws_cmp)
    ws_cmp.freeze_panes = "A2"
    _autosize(ws_cmp)

    # Metadata sheet.
    ws_meta = wb.create_sheet("meta")
    ws_meta.append(["field", "value"])
    ws_meta.append(["summary_csv_rows", len(summary_rows)])
    ws_meta.append(["benchmark_eval_rows", len(eval_rows)])
    ws_meta.append(["baseline_eval_rows", len(baseline_rows)])
    if benchmark_summary_path and benchmark_summary_path.exists():
        payload = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
        ws_meta.append(["num_selected_tasks", payload.get("num_selected_tasks", "")])
        ws_meta.append(["selected_tasks_file", payload.get("selected_tasks_file", "")])
    _style_header(ws_meta)
    _autosize(ws_meta)

    # Light value formatting.
    for ws in (ws_summary, ws_wide, ws_long):
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                val = _safe_float(cell.value)
                if val is not None:
                    cell.number_format = "0.0000"
    for ws in (ws_cmp,):
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                val = _safe_float(cell.value)
                if val is not None:
                    cell.number_format = "0.0000"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export proof run results to Excel.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--benchmark-summary-json", default="")
    parser.add_argument("--baseline-cache-root", default="")
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--output-xlsx", required=True)
    args = parser.parse_args()

    summary_rows = _read_summary_csv(Path(args.summary_csv))
    eval_rows = _iter_eval_rows(Path(args.benchmark_dir))
    benchmark_summary = Path(args.benchmark_summary_json) if args.benchmark_summary_json.strip() else None
    baseline_rows: list[dict[str, Any]] = []
    if args.baseline_cache_root.strip() and args.model_name_or_path.strip():
        baseline_rows = _load_baseline_rows(Path(args.baseline_cache_root), args.model_name_or_path)

    build_workbook(
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        benchmark_summary_path=benchmark_summary,
        baseline_rows=baseline_rows,
        out_path=Path(args.output_xlsx),
    )
    print(f"[OK] wrote {args.output_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
