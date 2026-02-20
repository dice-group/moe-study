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
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
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


def _metric_direction(task: str, metric: str) -> str:
    metric_l = metric.lower()
    if task.startswith("belebele_") or "acc" in metric_l or "f1" in metric_l:
        return "higher"
    if "perplexity" in metric_l or "ppl" in metric_l or "loss" in metric_l or "error" in metric_l:
        return "lower"
    return "higher"


def _parse_task_identity(task: str) -> tuple[str, str, str]:
    if task.startswith("belebele_"):
        return "Belebele", task.replace("belebele_", ""), "acc_norm"

    if task.startswith("sib200_"):
        return "SIB200", task.replace("sib200_", ""), "acc"

    if task.startswith("flores_en_perplexity_"):
        pair = task.replace("flores_en_perplexity_", "")
        if "-" in pair:
            src, tgt = pair.split("-", 1)
            if src == "eng_Latn":
                return "FLORES", tgt, "byte_perplexity en->x"
            if tgt == "eng_Latn":
                return "FLORES", src, "byte_perplexity x->en"
            return "FLORES", f"{src}-{tgt}", "byte_perplexity"

    prefix = task.split("_", 1)[0] if "_" in task else "other"
    return prefix.upper(), "", ""


def _select_best(values: dict[str, float], better: str) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    if better == "lower":
        key = min(values, key=values.get)
    else:
        key = max(values, key=values.get)
    return key, values[key]


def _run_label_maps(summary_rows: list[dict[str, str]], run_ids: list[str]) -> tuple[dict[str, str], list[str]]:
    run_to_kind: dict[str, str] = {}
    for row in summary_rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        kind = str(row.get("kind", "")).strip() or "run"
        run_to_kind[run_id] = kind

    base_kind_order = {
        "baseline": 0,
        "lora": 1,
        "hydralora": 2,
        "cola": 3,
        "mola": 4,
        "adamole": 5,
        "run": 99,
    }

    labels: dict[str, str] = {}
    seen: dict[str, int] = defaultdict(int)
    ordered_runs = sorted(
        run_ids,
        key=lambda rid: (base_kind_order.get(run_to_kind.get(rid, "run"), 50), run_to_kind.get(rid, "run"), rid),
    )

    for run_id in ordered_runs:
        if run_id.startswith("baseline::"):
            continue
        base_label = run_to_kind.get(run_id, "run")
        seen[base_label] += 1
        labels[run_id] = base_label if seen[base_label] == 1 else f"{base_label}#{seen[base_label]}"
    return labels, ordered_runs


def _apply_row_heatmap(ws, row_idx: int, better: str, start_col: int, end_col: int) -> None:
    has_num = any(isinstance(ws.cell(row_idx, c).value, (int, float)) for c in range(start_col, end_col + 1))
    if not has_num:
        return
    rng = f"{get_column_letter(start_col)}{row_idx}:{get_column_letter(end_col)}{row_idx}"
    if better == "higher":
        rule = ColorScaleRule(
            start_type="min",
            start_color="F8696B",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",
            end_type="max",
            end_color="63BE7B",
        )
    else:
        rule = ColorScaleRule(
            start_type="min",
            start_color="63BE7B",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",
            end_type="max",
            end_color="F8696B",
        )
    ws.conditional_formatting.add(rng, rule)


def _write_comparison_sheet(
    ws,
    *,
    summary_rows: list[dict[str, str]],
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> None:
    # Build one value per (run_id, task, metric) and retain all runs for side-by-side view.
    value_map: dict[tuple[str, str, str], float] = {}
    run_ids: set[str] = set()
    row_triplets: set[tuple[str, str, str]] = set()

    for row in [*eval_rows, *baseline_rows]:
        run_id = str(row.get("run_id", "")).strip()
        task = str(row.get("task", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = row.get("value")
        if not run_id or not task or not isinstance(value, (int, float)):
            continue
        key = (run_id, task, metric)
        value_map[key] = float(value)
        run_ids.add(run_id)
        row_triplets.add((task, metric, _metric_direction(task, metric)))

    run_label_map, ordered_runs = _run_label_maps(summary_rows, sorted(run_ids))
    approach_runs = [rid for rid in ordered_runs if not rid.startswith("baseline::")]
    approach_cols = [run_label_map[rid] for rid in approach_runs]

    headers = [
        "dataset",
        "language",
        "task",
        "metric",
        "better",
        "baseline",
        *approach_cols,
        "best_method",
        "best_value",
        "score_vs_baseline_pct",
    ]
    ws.append(headers)
    _style_header(ws)
    ws.freeze_panes = "A2"

    def _baseline_value(task: str, metric: str) -> float | None:
        for rid in ordered_runs:
            if rid.startswith("baseline::"):
                val = value_map.get((rid, task, metric))
                if isinstance(val, float):
                    return val
        return None

    # Stable sorting for compact scanability: dataset -> language -> metric -> task.
    def _sort_key(item: tuple[str, str, str]) -> tuple[str, str, str, str]:
        task, metric, _ = item
        dataset, language, metric_label = _parse_task_identity(task)
        metric_key = metric_label or metric
        return dataset, language, metric_key, task

    for task, metric, better in sorted(row_triplets, key=_sort_key):
        dataset, language, metric_label = _parse_task_identity(task)
        baseline_val = _baseline_value(task, metric)
        values_by_label: dict[str, float] = {}
        for rid in approach_runs:
            val = value_map.get((rid, task, metric))
            if isinstance(val, float):
                values_by_label[run_label_map[rid]] = val

        candidates = dict(values_by_label)
        if isinstance(baseline_val, float):
            candidates["baseline"] = baseline_val
        best_method, best_value = _select_best(candidates, better)

        score_vs_base: float | None = None
        if isinstance(baseline_val, float) and best_method and best_method != "baseline" and isinstance(best_value, float):
            denom = abs(baseline_val)
            if denom > 0:
                if better == "higher":
                    score_vs_base = (best_value - baseline_val) / denom
                else:
                    score_vs_base = (baseline_val - best_value) / denom

        row_values: list[Any] = [
            dataset,
            language,
            task,
            metric_label or metric,
            better,
            baseline_val if isinstance(baseline_val, float) else "",
        ]
        for col in approach_cols:
            row_values.append(values_by_label.get(col, ""))
        row_values.extend(
            [
                best_method or "",
                best_value if isinstance(best_value, float) else "",
                score_vs_base if isinstance(score_vs_base, float) else "",
            ]
        )
        ws.append(row_values)

    # Number formats and row-wise heatmap across baseline + approach columns.
    first_numeric_col = 6
    last_numeric_col = 5 + 1 + len(approach_cols)
    for r in range(2, ws.max_row + 1):
        for c in range(first_numeric_col, ws.max_column + 1):
            cell = ws.cell(r, c)
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.0000"
        better = str(ws.cell(r, 5).value or "")
        if better in {"higher", "lower"} and last_numeric_col >= first_numeric_col:
            _apply_row_heatmap(ws, r, better, first_numeric_col, last_numeric_col)

    # Basic column sizing for quick comparison reading.
    widths = {
        1: 12,  # dataset
        2: 14,  # language
        3: 42,  # task
        4: 24,  # metric
        5: 10,  # better
        6: 12,  # baseline
    }
    for idx, _ in enumerate(approach_cols, start=7):
        widths[idx] = 12
    widths[6 + len(approach_cols) + 1] = 16  # best_method
    widths[6 + len(approach_cols) + 2] = 12  # best_value
    widths[6 + len(approach_cols) + 3] = 20  # score_vs_baseline_pct
    for col_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width


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

    # New side-by-side comparison view (baseline + all approaches per task/language).
    ws_comp = wb.create_sheet("comparison_8b")
    _write_comparison_sheet(
        ws_comp,
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        baseline_rows=baseline_rows,
    )

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
