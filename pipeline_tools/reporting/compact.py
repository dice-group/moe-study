from __future__ import annotations

from collections import defaultdict
from typing import Any

from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .common import safe_float
from .excel import AVERAGE_FILL, AVERAGE_FONT, SECTION_FILL, SECTION_FONT, apply_row_heatmap, style_header


def metric_direction(task: str, metric: str) -> str:
    metric_l = metric.lower()
    if task.startswith("belebele_") or "acc" in metric_l or "f1" in metric_l:
        return "higher"
    if "perplexity" in metric_l or "ppl" in metric_l or "loss" in metric_l or "error" in metric_l or "bits_per_byte" in metric_l or "bpb" in metric_l:
        return "lower"
    return "higher"


def metric_name(metric: str) -> str:
    if "," in metric:
        return metric.split(",", 1)[0]
    return metric.strip()


def resource_tier(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"MEDIUM", "MID"}:
        return "MID"
    if text in {"HIGH", "LOW"}:
        return text
    return "UNKNOWN"


def parse_task_identity(task: str, metric: str) -> tuple[str, str, str]:
    metric_base = metric_name(metric)
    if task.startswith("belebele_"):
        return "Belebele", task.replace("belebele_", ""), metric_base or "acc_norm"
    if task.startswith("sib200_"):
        return "SIB200", task.replace("sib200_", ""), metric_base or "acc"
    if task.startswith("taxi1500_"):
        return "Taxi1500", task.replace("taxi1500_", ""), metric_base or "acc"
    if task.startswith("flores_en_perplexity_"):
        pair = task.replace("flores_en_perplexity_", "")
        if "-" in pair:
            src, tgt = pair.split("-", 1)
            base_metric = metric_base or "byte_perplexity"
            if src == "eng_Latn":
                return "FLORES", tgt, f"{base_metric} en->x"
            if tgt == "eng_Latn":
                return "FLORES", src, f"{base_metric} x->en"
            return "FLORES", f"{src}-{tgt}", base_metric
    prefix = task.split("_", 1)[0] if "_" in task else "other"
    return prefix.upper(), "", metric_base


def select_best(values: dict[str, float], better: str) -> tuple[str | None, float | None]:
    if not values:
        return None, None
    key = min(values, key=values.get) if better == "lower" else max(values, key=values.get)
    return key, values[key]


def collect_summary_metric_map(summary_rows: list[dict[str, str]], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in summary_rows:
        run_id = str(row.get("run_id", "")).strip()
        value = safe_float(row.get(key))
        if run_id and isinstance(value, float):
            out[run_id] = value
    return out


def run_label_maps(summary_rows: list[dict[str, str]], run_ids: list[str]) -> tuple[dict[str, str], list[str]]:
    def infer_kind(run_id: str) -> str:
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
        if run_id:
            run_to_kind[run_id] = str(row.get("kind", "")).strip() or "run"
    for run_id in run_ids:
        run_to_kind.setdefault(run_id, infer_kind(run_id))

    base_kind_order = {"baseline": 0, "lora": 1, "lora_r8": 1, "lora_r32": 1, "hydralora": 2, "cola": 3, "mola": 4, "adamole": 5, "run": 99}
    ordered_runs = sorted(run_ids, key=lambda rid: (base_kind_order.get(run_to_kind.get(rid, "run"), 50), run_to_kind.get(rid, "run"), rid))
    labels = {run_id: run_id for run_id in ordered_runs if not run_id.startswith("baseline::")}
    return labels, ordered_runs


def _normalized_score(values: dict[str, float], better: str) -> dict[str, float]:
    if not values:
        return {}
    vals = list(values.values())
    min_val = min(vals)
    max_val = max(vals)
    if max_val == min_val:
        return {key: 1.0 for key in values}
    if better == "lower":
        return {key: (max_val - value) / (max_val - min_val) for key, value in values.items()}
    return {key: (value - min_val) / (max_val - min_val) for key, value in values.items()}


def _shared_direction(rows: list[dict[str, Any]]) -> str | None:
    directions = {str(row.get("better", "")).strip() for row in rows if str(row.get("better", "")).strip()}
    if len(directions) == 1:
        return next(iter(directions))
    return None


def write_comparison_sheet(ws, *, summary_rows: list[dict[str, str]], eval_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], resource_map: dict[str, str]) -> None:
    value_map: dict[tuple[str, str, str], float] = {}
    run_ids: set[str] = set()
    row_triplets: set[tuple[str, str, str]] = set()
    baseline_metric_map: dict[tuple[str, str], float] = {}
    eval_tasks: set[str] = set()
    for row in eval_rows:
        run_id, task, metric, value = str(row.get("run_id", "")).strip(), str(row.get("task", "")).strip(), str(row.get("metric", "")).strip(), row.get("value")
        if run_id and task and isinstance(value, (int, float)):
            value_map[(run_id, task, metric)] = float(value)
            run_ids.add(run_id)
            row_triplets.add((task, metric, metric_direction(task, metric)))
            eval_tasks.add(task)
    for row in baseline_rows:
        run_id, task, metric, value = str(row.get("run_id", "")).strip(), str(row.get("task", "")).strip(), str(row.get("metric", "")).strip(), row.get("value")
        if run_id and task and isinstance(value, (int, float)):
            value_map[(run_id, task, metric)] = float(value)
            run_ids.add(run_id)
            baseline_metric_map[(task, metric_name(metric))] = float(value)
            # Allow baseline CSV metrics to fill in missing metric families for tasks that
            # are already part of the evaluated benchmark set, without expanding to every
            # tracked baseline task in the repository.
            if (not eval_tasks) or (task in eval_tasks):
                row_triplets.add((task, metric, metric_direction(task, metric)))

    run_label_map, ordered_runs = run_label_maps(summary_rows, sorted(run_ids))
    approach_runs = [rid for rid in ordered_runs if not rid.startswith("baseline::")]
    approach_cols = [run_label_map[rid] for rid in approach_runs]
    label_to_run = {run_label_map[rid]: rid for rid in approach_runs if rid in run_label_map}

    def baseline_value(task: str, metric: str) -> float | None:
        for rid in ordered_runs:
            if rid.startswith("baseline::"):
                val = value_map.get((rid, task, metric))
                if isinstance(val, float):
                    return val
        return baseline_metric_map.get((task, metric_name(metric)))

    def sort_key(item: tuple[str, str, str]) -> tuple[int, int, str, str]:
        task, metric, _ = item
        dataset, language, metric_label = parse_task_identity(task, metric)
        family = (metric_label or metric).split()[0]
        dataset_rank = {"Belebele": 0, "FLORES": 1}.get(dataset, 99)
        family_rank = {"byte_perplexity": 0, "word_perplexity": 1, "bits_per_byte": 2, "acc_norm": 3, "acc": 4}.get(family, 99)
        return dataset_rank, family_rank, language, task

    table_rows: list[dict[str, Any]] = []
    for task, metric, better in sorted(row_triplets, key=sort_key):
        dataset, language, metric_label = parse_task_identity(task, metric)
        metric_display = metric_label or metric
        baseline_val = baseline_value(task, metric)
        values_by_label = {run_label_map[rid]: val for rid in approach_runs if isinstance((val := value_map.get((rid, task, metric))), float)}
        scored_values = dict(values_by_label)
        if isinstance(baseline_val, float):
            scored_values["baseline"] = baseline_val
        best_method, best_value = select_best(scored_values, better)
        table_rows.append({"benchmark": dataset, "metric": metric_display, "metric_family": metric_display.split()[0], "language": language or task, "resource_availability": resource_map.get(language, "unknown") if language else "unknown", "better": better, "baseline": baseline_val if isinstance(baseline_val, float) else "", "approach_values": values_by_label, "best_approach": best_method or "", "best_value": best_value if isinstance(best_value, float) else "", "block_key": (dataset, metric_display.split()[0])})

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
    training_summary_specs = [("eval_loss", "lower", "0.0000"), ("train_loss", "lower", "0.0000"), ("train_runtime", "lower", "0.0000"), ("effective_tokens_per_sec", "higher", "0.0000"), ("num_input_tokens_seen", "higher", "0"), ("trainable_params", "lower", "0"), ("trainable_pct", "lower", "0.0000")]
    training_metric_maps = {key: collect_summary_metric_map(summary_rows, key) for key, _, _ in training_summary_specs}
    if any(training_metric_maps[key] for key, _, _ in training_summary_specs):
        metric_headers = ["metric", *approach_cols, "best_approach", "best_value"]
        ws.merge_cells(start_row=next_row, start_column=1, end_row=next_row, end_column=len(metric_headers))
        ws.cell(next_row, 1, "Training Summary (Approaches)")
        ws.cell(next_row, 1).fill = SECTION_FILL
        ws.cell(next_row, 1).font = SECTION_FONT
        ws.append(metric_headers)
        style_header(ws, header_row=next_row + 1)
        data_row = next_row + 2
        for metric_key, better, number_format in training_summary_specs:
            values = {label: training_metric_maps[metric_key].get(label_to_run[label]) for label in approach_cols if label in label_to_run}
            if not any(isinstance(v, float) for v in values.values()):
                continue
            best_label, best_value = select_best({k: float(v) for k, v in values.items() if isinstance(v, (int, float))}, better)
            ws.append([metric_key, *[values.get(label, "") for label in approach_cols], best_label or "", best_value if isinstance(best_value, float) else ""])
            for c in range(2, 2 + len(approach_cols)):
                cell = ws.cell(data_row, c)
                if isinstance(cell.value, (int, float)):
                    cell.number_format = number_format
            if approach_cols:
                apply_row_heatmap(ws, data_row, better, 2, 1 + len(approach_cols))
            data_row += 1
        next_row = data_row + 1

    headers = ["benchmark", "metric", "language", "resource_availability", "better", "baseline", *approach_cols, "best_approach", "best_value"]
    baseline_col = 6
    first_approach_col = 7
    last_approach_col = first_approach_col + len(approach_cols) - 1

    def write_block(start_row: int, block_name: str, block_rows: list[dict[str, Any]]) -> int:
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=len(headers))
        ws.cell(start_row, 1, block_name)
        ws.cell(start_row, 1).fill = SECTION_FILL
        ws.cell(start_row, 1).font = SECTION_FONT
        ws.append(headers)
        style_header(ws, header_row=start_row + 1)
        row_cursor = start_row + 2
        for item in block_rows:
            ws.append([item["benchmark"], item["metric"], item["language"], item["resource_availability"], item["better"], item["baseline"], *[item["approach_values"].get(col, "") for col in approach_cols], item["best_approach"], item["best_value"]])
            apply_row_heatmap(ws, row_cursor, str(item["better"]), baseline_col, max(baseline_col, last_approach_col))
            row_cursor += 1
        if block_rows:
            direction = str(block_rows[0]["better"] or "higher")
            baseline_vals = [float(r["baseline"]) for r in block_rows if isinstance(r["baseline"], (int, float))]
            avg_values = {col: sum(vals) / len(vals) for col in approach_cols if (vals := [float(r["approach_values"][col]) for r in block_rows if isinstance(r["approach_values"].get(col), (int, float))])}
            avg_baseline = (sum(baseline_vals) / len(baseline_vals)) if baseline_vals else ""
            avg_scored_values = dict(avg_values)
            if isinstance(avg_baseline, float):
                avg_scored_values["baseline"] = avg_baseline
            avg_best_method, avg_best_value = select_best(avg_scored_values, direction)
            ws.append(["AVG", "average", "all", "", direction, avg_baseline, *[avg_values.get(col, "") for col in approach_cols], avg_best_method or "", avg_best_value if isinstance(avg_best_value, float) else ""])
            for c in range(1, len(headers) + 1):
                ws.cell(row_cursor, c).fill = AVERAGE_FILL
                ws.cell(row_cursor, c).font = AVERAGE_FONT
            apply_row_heatmap(ws, row_cursor, direction, baseline_col, max(baseline_col, last_approach_col))
            row_cursor += 1
        return row_cursor + 1

    block_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        block_map[row["block_key"]].append(row)
    def block_sort_key(block_key: tuple[str, str]) -> tuple[int, int, str]:
        dataset, family = block_key
        if dataset == "Belebele":
            return (0, {"acc_norm": 0, "acc": 1}.get(family, 99), family)
        if dataset == "FLORES":
            return (1, {"byte_perplexity": 0, "word_perplexity": 1, "bits_per_byte": 2}.get(family, 99), family)
        return (99, 99, f"{dataset}:{family}")
    for block_key in sorted(block_map.keys(), key=block_sort_key):
        dataset, family = block_key
        block_name = f"FLORES ({family}) Block" if dataset == "FLORES" else ("Belebele Block" if dataset == "Belebele" and family == "acc_norm" else f"{dataset} ({family}) Block")
        next_row = write_block(next_row, block_name, block_map[block_key])

    aggregate_headers = ["benchmark", "tier", "score_type", "baseline", *approach_cols, "best_approach", "best_value"]
    def write_aggregate_block(start_row: int, block_name: str, rows: list[dict[str, Any]], *, normalized: bool) -> int:
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=len(aggregate_headers))
        ws.cell(start_row, 1, block_name)
        ws.cell(start_row, 1).fill = SECTION_FILL
        ws.cell(start_row, 1).font = SECTION_FONT
        ws.append(aggregate_headers)
        style_header(ws, header_row=start_row + 1)
        row_idx = start_row + 2
        group_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in rows:
            tier = resource_tier(item.get("resource_availability"))
            group_map[(str(item["benchmark"]), tier)].append(item)
            group_map[(str(item["benchmark"]), "ALL")].append(item)
        all_rows = list(rows)
        if all_rows:
            group_map[("CROSS", "ALL")] = all_rows
            for tier in ("HIGH", "MID", "LOW", "UNKNOWN"):
                tier_rows = [item for item in all_rows if resource_tier(item.get("resource_availability")) == tier]
                if tier_rows:
                    group_map[("CROSS", tier)] = tier_rows
        benchmarks = sorted({key[0] for key in group_map if key[0] != "CROSS"})
        if any(key[0] == "CROSS" for key in group_map):
            benchmarks.append("CROSS")
        ordered_keys: list[tuple[str, str]] = []
        for benchmark_name in benchmarks:
            for tier in ("ALL", "HIGH", "MID", "LOW"):
                ordered_keys.append((benchmark_name, tier))
            if (benchmark_name, "UNKNOWN") in group_map:
                ordered_keys.append((benchmark_name, "UNKNOWN"))
        for benchmark_name, tier in ordered_keys:
            grouped_rows = group_map.get((benchmark_name, tier), [])
            baseline_vals: list[float] = []
            approach_acc: dict[str, list[float]] = {label: [] for label in approach_cols}
            group_direction = "higher" if normalized else (_shared_direction(grouped_rows) or "higher")
            for item in grouped_rows:
                values_for_norm: dict[str, float] = {}
                if isinstance(item.get("baseline"), (int, float)):
                    values_for_norm["baseline"] = float(item["baseline"])
                for label in approach_cols:
                    val = item["approach_values"].get(label)
                    if isinstance(val, (int, float)):
                        values_for_norm[label] = float(val)
                scored_values = _normalized_score(values_for_norm, str(item["better"])) if normalized else values_for_norm
                if isinstance(scored_values.get("baseline"), float):
                    baseline_vals.append(scored_values["baseline"])
                for label in approach_cols:
                    if isinstance(scored_values.get(label), float):
                        approach_acc[label].append(scored_values[label])
            baseline_avg = (sum(baseline_vals) / len(baseline_vals)) if baseline_vals else ""
            approach_avg_map = {label: (sum(vals) / len(vals)) for label, vals in approach_acc.items() if vals}
            scored_avg_values = dict(approach_avg_map)
            if isinstance(baseline_avg, float):
                scored_avg_values["baseline"] = baseline_avg
            best_label, best_value = select_best(scored_avg_values, group_direction)
            ws.append([benchmark_name, tier, "normalized_avg" if normalized else "raw_avg", baseline_avg, *[approach_avg_map.get(label, "") for label in approach_cols], best_label or "", best_value if isinstance(best_value, float) else ""])
            apply_row_heatmap(ws, row_idx, group_direction, 4, 4 + len(approach_cols))
            row_idx += 1
        return row_idx + 1

    if table_rows:
        next_row = write_aggregate_block(next_row, "Benchmark Tier Averages", table_rows, normalized=False)
        next_row = write_aggregate_block(next_row, "Normalized Benchmark Scores", table_rows, normalized=True)

    ws.freeze_panes = "A4"
    ws.sheet_view.showGridLines = False
    approach_width = max(12, min(40, max((len(col) for col in approach_cols), default=12) + 2))
    best_approach_width = max(16, min(50, max((len(col) for col in approach_cols), default=16) + 4))
    widths = {1: 14, 2: 24, 3: 16, 4: 20, 5: 10, 6: 12}
    for idx, _ in enumerate(approach_cols, start=7):
        widths[idx] = approach_width
    widths[7 + len(approach_cols)] = best_approach_width
    widths[8 + len(approach_cols)] = 12
    for col_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width
