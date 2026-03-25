#!/usr/bin/env python3
"""Export proof pipeline results to a styled Excel workbook.

Inputs:
- summary CSV produced by run_multilingual_proof.py
- benchmark directory containing *_lm_eval.jsonl outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


HEADER_FILL = PatternFill(fill_type="solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
SECTION_FILL = PatternFill(fill_type="solid", fgColor="305496")
SECTION_FONT = Font(color="FFFFFF", bold=True, size=11)
AVERAGE_FILL = PatternFill(fill_type="solid", fgColor="E2ECF9")
AVERAGE_FONT = Font(color="1F2937", bold=True)


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _iter_eval_rows(benchmark_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not benchmark_dir.exists():
        return rows

    def _ingest_payload(payload: Any, *, run_id: str) -> None:
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
            return

        # Fallback for lm-eval object payloads with a "results" map.
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            return
        for task, metrics in results.items():
            if not isinstance(task, str) or not isinstance(metrics, dict):
                continue
            if task.startswith("belebele_"):
                for candidate in ("acc_norm,none", "acc,none"):
                    val = metrics.get(candidate)
                    if isinstance(val, (int, float)):
                        rows.append(
                            {
                                "run_id": run_id,
                                "task": task,
                                "metric": candidate,
                                "value": float(val),
                                "wandb_project": "",
                                "wandb_prefix": "",
                            }
                        )
                        break
                continue
            if task.startswith("flores_en_perplexity_"):
                for candidate in ("byte_perplexity,none", "word_perplexity,none", "bits_per_byte,none"):
                    val = metrics.get(candidate)
                    if isinstance(val, (int, float)):
                        rows.append(
                            {
                                "run_id": run_id,
                                "task": task,
                                "metric": candidate,
                                "value": float(val),
                                "wandb_project": "",
                                "wandb_prefix": "",
                            }
                        )
                continue
            for candidate in ("acc_norm,none", "acc,none", "byte_perplexity,none", "word_perplexity,none"):
                val = metrics.get(candidate)
                if isinstance(val, (int, float)):
                    rows.append(
                        {
                            "run_id": run_id,
                            "task": task,
                            "metric": candidate,
                            "value": float(val),
                            "wandb_project": "",
                            "wandb_prefix": "",
                        }
                    )
                    break

    for path in sorted(benchmark_dir.rglob("*_lm_eval.jsonl")):
        run_id = path.parent.name
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                _ingest_payload(payload, run_id=run_id)
            continue
        if path.is_dir():
            for result_file in sorted(path.rglob("results_*.json")):
                try:
                    payload = json.loads(result_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                _ingest_payload(payload, run_id=run_id)

    # Current benchmark script output: per-run lm-eval JSON files.
    for json_file in sorted(benchmark_dir.rglob("with_language_ids_*.json")) + sorted(
        benchmark_dir.rglob("no_language_ids.json")
    ):
        if not json_file.is_file():
            continue
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        _ingest_payload(payload, run_id=json_file.parent.name)
    return rows


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
    if (
        "perplexity" in metric_l
        or "ppl" in metric_l
        or "loss" in metric_l
        or "error" in metric_l
        or "bits_per_byte" in metric_l
        or "bpb" in metric_l
    ):
        return "lower"
    return "higher"


def _metric_name(metric: str) -> str:
    if "," in metric:
        return metric.split(",", 1)[0]
    return metric.strip()


def _parse_task_identity(task: str, metric: str) -> tuple[str, str, str]:
    metric_name = _metric_name(metric)
    if task.startswith("belebele_"):
        return "Belebele", task.replace("belebele_", ""), metric_name or "acc_norm"

    if task.startswith("sib200_"):
        return "SIB200", task.replace("sib200_", ""), metric_name or "acc"

    if task.startswith("taxi1500_"):
        return "Taxi1500", task.replace("taxi1500_", ""), metric_name or "acc"

    if task.startswith("flores_en_perplexity_"):
        pair = task.replace("flores_en_perplexity_", "")
        if "-" in pair:
            src, tgt = pair.split("-", 1)
            base_metric = metric_name or "byte_perplexity"
            if src == "eng_Latn":
                return "FLORES", tgt, f"{base_metric} en->x"
            if tgt == "eng_Latn":
                return "FLORES", src, f"{base_metric} x->en"
            return "FLORES", f"{src}-{tgt}", base_metric

    prefix = task.split("_", 1)[0] if "_" in task else "other"
    return prefix.upper(), "", metric_name


def _select_best(values: dict[str, float], better: str) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    if better == "lower":
        key = min(values, key=values.get)
    else:
        key = max(values, key=values.get)
    return key, values[key]


def _run_label_maps(summary_rows: list[dict[str, str]], run_ids: list[str]) -> tuple[dict[str, str], list[str]]:
    def _infer_kind(run_id: str) -> str:
        rid = run_id.lower()
        if run_id.startswith("baseline::"):
            return "baseline"
        for kind in ("hydralora", "adamole", "cola", "mola", "lora"):
            if kind in rid:
                return kind
        return "run"

    run_to_kind: dict[str, str] = {}
    for row in summary_rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        kind = str(row.get("kind", "")).strip() or "run"
        run_to_kind[run_id] = kind
    for run_id in run_ids:
        if run_id not in run_to_kind:
            run_to_kind[run_id] = _infer_kind(run_id)

    base_kind_order = {
        "baseline": 0,
        "lora": 1,
        "lora_r8": 1,
        "lora_r32": 1,
        "hydralora": 2,
        "cola": 3,
        "mola": 4,
        "adamole": 5,
        "run": 99,
    }

    labels: dict[str, str] = {}
    ordered_runs = sorted(
        run_ids,
        key=lambda rid: (base_kind_order.get(run_to_kind.get(rid, "run"), 50), run_to_kind.get(rid, "run"), rid),
    )

    for run_id in ordered_runs:
        if run_id.startswith("baseline::"):
            continue
        # Keep full approach IDs (run_id) for report columns.
        labels[run_id] = run_id
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


def _load_resource_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    train_seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = str(row.get("code", "")).strip()
            script = str(row.get("script", "")).strip()
            if not code or not script:
                continue
            key = f"{code}_{script}"
            split = str(row.get("split", "")).strip().lower()
            availability = str(row.get("resource_availability", "")).strip().upper() or "unknown"
            if split == "train":
                out[key] = availability
                train_seen.add(key)
                continue
            if key not in train_seen and key not in out:
                out[key] = availability
    return out


def _write_comparison_sheet(
    ws,
    *,
    summary_rows: list[dict[str, str]],
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    resource_map: dict[str, str],
) -> None:
    value_map: dict[tuple[str, str, str], float] = {}
    run_ids: set[str] = set()
    row_triplets: set[tuple[str, str, str]] = set()
    baseline_metric_map: dict[tuple[str, str], float] = {}

    for row in eval_rows:
        run_id = str(row.get("run_id", "")).strip()
        task = str(row.get("task", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = row.get("value")
        if not run_id or not task or not isinstance(value, (int, float)):
            continue
        value_map[(run_id, task, metric)] = float(value)
        run_ids.add(run_id)
        row_triplets.add((task, metric, _metric_direction(task, metric)))

    for row in baseline_rows:
        run_id = str(row.get("run_id", "")).strip()
        task = str(row.get("task", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = row.get("value")
        if not run_id or not task or not isinstance(value, (int, float)):
            continue
        value_map[(run_id, task, metric)] = float(value)
        run_ids.add(run_id)
        baseline_metric_map[(task, _metric_name(metric))] = float(value)

    run_label_map, ordered_runs = _run_label_maps(summary_rows, sorted(run_ids))
    approach_runs = [rid for rid in ordered_runs if not rid.startswith("baseline::")]
    approach_cols = [run_label_map[rid] for rid in approach_runs]
    label_to_run = {run_label_map[rid]: rid for rid in approach_runs if rid in run_label_map}

    def _baseline_value(task: str, metric: str) -> float | None:
        for rid in ordered_runs:
            if rid.startswith("baseline::"):
                val = value_map.get((rid, task, metric))
                if isinstance(val, float):
                    return val
        return baseline_metric_map.get((task, _metric_name(metric)))

    def _sort_key(item: tuple[str, str, str]) -> tuple[int, int, str, str]:
        task, metric, _ = item
        dataset, language, metric_label = _parse_task_identity(task, metric)
        metric_family = (metric_label or metric).split()[0]
        dataset_rank = {"Belebele": 0, "FLORES": 1}.get(dataset, 99)
        family_rank = {"byte_perplexity": 0, "word_perplexity": 1, "bits_per_byte": 2, "acc_norm": 3, "acc": 4}.get(
            metric_family, 99
        )
        return dataset_rank, family_rank, language, task

    table_rows: list[dict[str, Any]] = []
    for task, metric, better in sorted(row_triplets, key=_sort_key):
        dataset, language, metric_label = _parse_task_identity(task, metric)
        metric_display = metric_label or metric
        metric_family = metric_display.split()[0]
        baseline_val = _baseline_value(task, metric)
        values_by_label: dict[str, float] = {}
        for rid in approach_runs:
            val = value_map.get((rid, task, metric))
            if isinstance(val, float):
                values_by_label[run_label_map[rid]] = val
        best_method, best_value = _select_best(values_by_label, better)
        resource = resource_map.get(language, "unknown") if language else "unknown"
        table_rows.append(
            {
                "benchmark": dataset,
                "metric": metric_display,
                "metric_family": metric_family,
                "language": language or task,
                "resource_availability": resource,
                "better": better,
                "baseline": baseline_val if isinstance(baseline_val, float) else "",
                "approach_values": values_by_label,
                "best_approach": best_method or "",
                "best_value": best_value if isinstance(best_value, float) else "",
                "block_key": (dataset, metric_family),
            }
        )

    title_cols = 9 + max(1, len(approach_cols))
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=title_cols)
    ws.cell(1, 1, "Compact Benchmark Comparison")
    ws.cell(1, 1).font = Font(color="FFFFFF", bold=True, size=13)
    ws.cell(1, 1).fill = PatternFill(fill_type="solid", fgColor="203864")
    ws.cell(1, 1).alignment = Alignment(horizontal="left", vertical="center")

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=title_cols)
    ws.cell(2, 1, "FLORES blocks are split by metric: byte_perplexity first, then other metrics.")
    ws.cell(2, 1).font = Font(color="1F2937", size=10)
    ws.cell(2, 1).fill = PatternFill(fill_type="solid", fgColor="DCE6F1")
    ws.cell(2, 1).alignment = Alignment(horizontal="left", vertical="center")

    next_row = 4

    # Horizontal eval-loss block.
    run_eval_loss: dict[str, float] = {}
    for row in summary_rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        loss = _safe_float(row.get("eval_loss"))
        if isinstance(loss, float):
            run_eval_loss[run_id] = loss
    eval_values = {label: run_eval_loss.get(label_to_run[label]) for label in approach_cols if label in label_to_run}
    if any(isinstance(v, float) for v in eval_values.values()):
        eval_headers = ["metric", *approach_cols, "best_approach", "best_value"]
        end_col = len(eval_headers)
        ws.merge_cells(start_row=next_row, start_column=1, end_row=next_row, end_column=end_col)
        ws.cell(next_row, 1, "Final Eval Loss (Approaches, lower is better)")
        ws.cell(next_row, 1).fill = SECTION_FILL
        ws.cell(next_row, 1).font = SECTION_FONT
        ws.cell(next_row, 1).alignment = Alignment(horizontal="left", vertical="center")
        ws.append(eval_headers)
        _style_header(ws, header_row=next_row + 1)
        row = ["eval_loss"]
        for label in approach_cols:
            row.append(eval_values.get(label, ""))
        valid_eval = {k: float(v) for k, v in eval_values.items() if isinstance(v, (int, float))}
        best_label, best_loss = _select_best(valid_eval, "lower")
        row.extend([best_label or "", best_loss if isinstance(best_loss, float) else ""])
        ws.append(row)
        data_row = next_row + 2
        for c in range(2, 2 + len(approach_cols)):
            cell = ws.cell(data_row, c)
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.0000"
        if approach_cols:
            _apply_row_heatmap(ws, data_row, "lower", 2, 1 + len(approach_cols))
        next_row = data_row + 2

    headers = [
        "benchmark",
        "metric",
        "language",
        "resource_availability",
        "better",
        "baseline",
        *approach_cols,
        "best_approach",
        "best_value",
    ]
    first_approach_col = 7
    last_approach_col = first_approach_col + len(approach_cols) - 1

    def _write_block(start_row: int, block_name: str, block_rows: list[dict[str, Any]]) -> int:
        end_col = len(headers)
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=end_col)
        ws.cell(start_row, 1, block_name)
        ws.cell(start_row, 1).fill = SECTION_FILL
        ws.cell(start_row, 1).font = SECTION_FONT
        ws.cell(start_row, 1).alignment = Alignment(horizontal="left", vertical="center")

        header_row = start_row + 1
        ws.append(headers)
        _style_header(ws, header_row=header_row)

        row_cursor = header_row + 1
        for item in block_rows:
            row_values: list[Any] = [
                item["benchmark"],
                item["metric"],
                item["language"],
                item["resource_availability"],
                item["better"],
                item["baseline"],
            ]
            for col in approach_cols:
                row_values.append(item["approach_values"].get(col, ""))
            row_values.extend([item["best_approach"], item["best_value"]])
            ws.append(row_values)
            for c in range(6, len(headers) + 1):
                cell = ws.cell(row_cursor, c)
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "0.0000"
            if approach_cols and str(ws.cell(row_cursor, 5).value or "") in {"higher", "lower"}:
                _apply_row_heatmap(ws, row_cursor, str(ws.cell(row_cursor, 5).value), first_approach_col, last_approach_col)
            row_cursor += 1

        if block_rows:
            direction = str(block_rows[0]["better"] or "higher")
            baseline_vals = [float(r["baseline"]) for r in block_rows if isinstance(r["baseline"], (int, float))]
            avg_values: dict[str, float] = {}
            for col in approach_cols:
                vals = [float(r["approach_values"][col]) for r in block_rows if isinstance(r["approach_values"].get(col), (int, float))]
                if vals:
                    avg_values[col] = sum(vals) / len(vals)
            avg_best_method, avg_best_value = _select_best(avg_values, direction)
            avg_baseline = (sum(baseline_vals) / len(baseline_vals)) if baseline_vals else ""
            avg_row = ["AVG", "average", "all", "", direction, avg_baseline]
            for col in approach_cols:
                avg_row.append(avg_values.get(col, ""))
            avg_row.extend([avg_best_method or "", avg_best_value if isinstance(avg_best_value, float) else ""])
            ws.append(avg_row)
            avg_row_idx = row_cursor
            for c in range(1, len(headers) + 1):
                cell = ws.cell(avg_row_idx, c)
                cell.fill = AVERAGE_FILL
                cell.font = AVERAGE_FONT
                if c >= 6 and isinstance(cell.value, (int, float)):
                    cell.number_format = "0.0000"
            if approach_cols:
                _apply_row_heatmap(ws, avg_row_idx, direction, first_approach_col, last_approach_col)
            row_cursor += 1
        else:
            ws.append([block_name, "no_data", "", "", "", "", *["" for _ in approach_cols], "", ""])
            row_cursor += 1

        return row_cursor + 1

    block_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        block_map[row["block_key"]].append(row)

    def _block_sort_key(block_key: tuple[str, str]) -> tuple[int, int, str]:
        dataset, family = block_key
        if dataset == "Belebele":
            return (0, {"acc_norm": 0, "acc": 1}.get(family, 99), family)
        if dataset == "FLORES":
            return (1, {"byte_perplexity": 0, "word_perplexity": 1, "bits_per_byte": 2}.get(family, 99), family)
        return (99, 99, f"{dataset}:{family}")

    for block_key in sorted(block_map.keys(), key=_block_sort_key):
        dataset, family = block_key
        if dataset == "FLORES":
            block_name = f"FLORES ({family}) Block"
        elif dataset == "Belebele" and family == "acc_norm":
            block_name = "Belebele Block"
        else:
            block_name = f"{dataset} ({family}) Block"
        next_row = _write_block(next_row, block_name, block_map[block_key])

    ws.freeze_panes = "A4"
    ws.sheet_view.showGridLines = False

    approach_width = max(12, min(40, max((len(col) for col in approach_cols), default=12) + 2))
    best_approach_width = max(16, min(50, max((len(col) for col in approach_cols), default=16) + 4))

    widths = {
        1: 14,  # benchmark
        2: 24,  # metric
        3: 16,  # language
        4: 20,  # resource_availability
        5: 10,  # better
        6: 12,  # baseline
    }
    for idx, _ in enumerate(approach_cols, start=7):
        widths[idx] = approach_width
    widths[7 + len(approach_cols)] = best_approach_width  # best_approach
    widths[8 + len(approach_cols)] = 12  # best_value
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


def _load_baseline_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            task = str(row.get("task", "")).strip()
            metric = str(row.get("metric", "")).strip()
            value = _safe_float(row.get("value"))
            if not task or not metric or value is None:
                continue
            model_slug = str(row.get("model_slug", "")).strip() or "baseline"
            rows.append(
                {
                    "run_id": f"baseline::{model_slug}",
                    "task": task,
                    "metric": metric,
                    "value": float(value),
                    "wandb_project": "",
                    "wandb_prefix": "",
                }
            )
    return rows


def _resolve_baseline_lm_eval_path(path_or_dir: Path) -> Path | None:
    if not path_or_dir.exists():
        return None
    if path_or_dir.is_file() and path_or_dir.suffix == ".json":
        return path_or_dir
    candidates = sorted(path_or_dir.rglob("*_baseline_lm_eval_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_baseline_rows_from_lm_eval_json(path: Path, model_name_or_path: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    source_model = model_name_or_path.strip()
    if not source_model:
        model_args = payload.get("config", {}).get("model_args", {})
        if isinstance(model_args, dict):
            source_model = str(model_args.get("pretrained", "")).strip()
    slug = _model_slug(source_model) if source_model else _sanitize_slug(path.stem.split("_baseline_lm_eval_", 1)[0])
    run_id = f"baseline::{slug}"

    results = payload.get("results")
    if not isinstance(results, dict):
        return []

    rows: list[dict[str, Any]] = []
    for task, metrics in results.items():
        if not isinstance(task, str) or not isinstance(metrics, dict):
            continue

        if task.startswith("belebele_"):
            candidates = ("acc_norm,none", "acc,none")
        elif task.startswith("flores_en_perplexity_"):
            candidates = ("byte_perplexity,none", "word_perplexity,none", "bits_per_byte,none")
        else:
            candidates = ("acc_norm,none", "acc,none", "byte_perplexity,none", "word_perplexity,none", "bits_per_byte,none")

        for metric in candidates:
            value = metrics.get(metric)
            if isinstance(value, (int, float)):
                rows.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "metric": metric,
                        "value": float(value),
                        "wandb_project": "",
                        "wandb_prefix": "",
                    }
                )
                if task.startswith("belebele_"):
                    break
    return rows


def build_workbook(
    *,
    summary_rows: list[dict[str, str]],
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    resource_map: dict[str, str],
    out_path: Path,
) -> None:
    wb = Workbook()

    ws_compact = wb.active
    ws_compact.title = "benchmark_compact"
    _write_comparison_sheet(
        ws_compact,
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        baseline_rows=baseline_rows,
        resource_map=resource_map,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export proof run results to Excel.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument(
        "--benchmark-summary-json",
        default="",
        help="Legacy/no-op: kept for CLI compatibility.",
    )
    parser.add_argument("--baseline-cache-root", default="")
    parser.add_argument(
        "--baseline-lm-eval-json",
        default="outputs/baselines/llama31_8b_full_eval",
        help="Path to baseline lm-eval JSON file or directory containing *_baseline_lm_eval_*.json.",
    )
    parser.add_argument(
        "--baseline-metrics-csv",
        default="configs/baselines/llama31_8b_full_eval/l-3-1-8b_full_baseline_metrics_summary.csv",
    )
    parser.add_argument(
        "--resource-availability-csv",
        default="result_analysis/l2v_nested_samples/fineweb2-language-distribution.csv",
    )
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--output-xlsx", required=True)
    args = parser.parse_args()

    summary_rows = _read_summary_csv(Path(args.summary_csv))
    eval_rows = _iter_eval_rows(Path(args.benchmark_dir))
    baseline_rows: list[dict[str, Any]] = []
    if args.baseline_cache_root.strip() and args.model_name_or_path.strip():
        baseline_rows = _load_baseline_rows(Path(args.baseline_cache_root), args.model_name_or_path)
    baseline_json_rows: list[dict[str, Any]] = []
    if args.baseline_lm_eval_json.strip():
        baseline_json_path = _resolve_baseline_lm_eval_path(Path(args.baseline_lm_eval_json))
        if baseline_json_path is not None:
            baseline_json_rows = _load_baseline_rows_from_lm_eval_json(
                baseline_json_path, args.model_name_or_path
            )
    baseline_csv_rows = _load_baseline_rows_from_csv(Path(args.baseline_metrics_csv))
    if baseline_rows or baseline_json_rows or baseline_csv_rows:
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in [*baseline_rows, *baseline_json_rows, *baseline_csv_rows]:
            key = (str(row["run_id"]), str(row["task"]), str(row["metric"]))
            merged[key] = row
        baseline_rows = list(merged.values())
    resource_map = _load_resource_map(Path(args.resource_availability_csv))

    build_workbook(
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        baseline_rows=baseline_rows,
        resource_map=resource_map,
        out_path=Path(args.output_xlsx),
    )
    print(f"[OK] wrote {args.output_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
