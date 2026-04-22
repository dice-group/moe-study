from __future__ import annotations

from collections import defaultdict
from typing import Any

from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .common import safe_float
from .excel import (
    AVERAGE_FILL,
    AVERAGE_FONT,
    SECTION_FILL,
    SECTION_FONT,
    apply_positive_magnitude_heatmap,
    apply_pvalue_heatmap,
    apply_row_heatmap,
    apply_signed_effect_heatmap,
    style_best_cell,
    style_header,
    style_significance_cell,
)
from .significance import build_significance_results


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
    ordered_runs = sorted(
        run_ids,
        key=lambda rid: (base_kind_order.get(run_to_kind.get(rid, "run"), 50), run_to_kind.get(rid, "run"), rid),
    )
    labels = {run_id: run_id for run_id in ordered_runs if not run_id.startswith("baseline::")}
    return labels, ordered_runs


def _shared_direction(rows: list[dict[str, Any]]) -> str | None:
    directions = {str(row.get("better", "")).strip() for row in rows if str(row.get("better", "")).strip()}
    if len(directions) == 1:
        return next(iter(directions))
    return None


def _format_stat(value: float | None) -> float | str:
    if value is None:
        return ""
    return float(value)


def _metric_order(metric_label: str) -> tuple[int, int]:
    mapping = {
        "acc_norm": (0, 0),
        "acc": (0, 0),
        "byte_perplexity en->x": (0, 0),
        "byte_perplexity x->en": (0, 1),
        "word_perplexity en->x": (1, 0),
        "word_perplexity x->en": (1, 1),
        "bits_per_byte en->x": (2, 0),
        "bits_per_byte x->en": (2, 1),
    }
    return mapping.get(metric_label, (99, 99))


def _dataset_rank(dataset: str) -> int:
    return {"Belebele": 0, "FLORES": 1, "SIB200": 2, "Taxi1500": 3}.get(dataset, 99)


def _block_sort_key(block_key: tuple[str, str]) -> tuple[int, int, int, str]:
    dataset, metric_family = block_key
    family_rank, direction_rank = _metric_order(metric_family)
    return _dataset_rank(dataset), family_rank, direction_rank, metric_family


def write_comparison_sheet(
    ws,
    *,
    summary_rows: list[dict[str, str]],
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    resource_map: dict[str, str],
) -> None:
    value_map: dict[tuple[str, str, str], float] = {}
    display_value_map: dict[tuple[str, str, str], str] = {}
    run_ids: set[str] = set()
    row_triplets: set[tuple[str, str, str]] = set()
    baseline_metric_map: dict[tuple[str, str], float] = {}
    baseline_display_map: dict[tuple[str, str], str] = {}
    eval_tasks: set[str] = set()

    for row in eval_rows:
        run_id = str(row.get("run_id", "")).strip()
        task = str(row.get("task", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = row.get("value")
        display_value = str(row.get("display_value", "")).strip()
        if run_id and task and isinstance(value, (int, float)):
            value_map[(run_id, task, metric)] = float(value)
            run_ids.add(run_id)
            row_triplets.add((task, metric, metric_direction(task, metric)))
            eval_tasks.add(task)
        elif run_id and task and metric and display_value:
            display_value_map[(run_id, task, metric)] = display_value
            run_ids.add(run_id)
            row_triplets.add((task, metric, metric_direction(task, metric)))
            eval_tasks.add(task)

    for row in baseline_rows:
        run_id = str(row.get("run_id", "")).strip()
        task = str(row.get("task", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = row.get("value")
        display_value = str(row.get("display_value", "")).strip()
        if run_id and task and isinstance(value, (int, float)):
            value_map[(run_id, task, metric)] = float(value)
            run_ids.add(run_id)
            baseline_metric_map[(task, metric_name(metric))] = float(value)
            if (not eval_tasks) or (task in eval_tasks):
                row_triplets.add((task, metric, metric_direction(task, metric)))
        elif run_id and task and metric and display_value:
            run_ids.add(run_id)
            baseline_display_map[(task, metric_name(metric))] = display_value
            if (not eval_tasks) or (task in eval_tasks):
                row_triplets.add((task, metric, metric_direction(task, metric)))

    run_label_map, ordered_runs = run_label_maps(summary_rows, sorted(run_ids))
    approach_runs = [run_id for run_id in ordered_runs if not run_id.startswith("baseline::")]
    approach_cols = [run_label_map[run_id] for run_id in approach_runs]
    label_to_run = {run_label_map[run_id]: run_id for run_id in approach_runs if run_id in run_label_map}

    def baseline_value(task: str, metric: str) -> float | str | None:
        for run_id in ordered_runs:
            if run_id.startswith("baseline::"):
                value = value_map.get((run_id, task, metric))
                if isinstance(value, float):
                    return value
                display_value = display_value_map.get((run_id, task, metric))
                if display_value:
                    return display_value
        numeric_baseline = baseline_metric_map.get((task, metric_name(metric)))
        if isinstance(numeric_baseline, float):
            return numeric_baseline
        return baseline_display_map.get((task, metric_name(metric)))

    def sort_key(item: tuple[str, str, str]) -> tuple[int, int, int, str, str]:
        task, metric, _ = item
        dataset, language, metric_label = parse_task_identity(task, metric)
        family_rank, direction_rank = _metric_order(metric_label or metric)
        return _dataset_rank(dataset), family_rank, direction_rank, language, task

    table_rows: list[dict[str, Any]] = []
    for task, metric, better in sorted(row_triplets, key=sort_key):
        benchmark, language, metric_label = parse_task_identity(task, metric)
        metric_display = metric_label or metric
        baseline_val = baseline_value(task, metric)
        values_by_label = {
            run_label_map[run_id]: value
            for run_id in approach_runs
            if isinstance((value := value_map.get((run_id, task, metric))), float)
        }
        display_values_by_label = {
            run_label_map[run_id]: display_value_map[(run_id, task, metric)]
            for run_id in approach_runs
            if (run_id, task, metric) in display_value_map and run_id in run_label_map
        }
        scored_values = dict(values_by_label)
        if isinstance(baseline_val, float):
            scored_values["baseline"] = baseline_val
        best_method, best_value = select_best(scored_values, better)
        table_rows.append(
            {
                "benchmark": benchmark,
                "metric": metric_display,
                "metric_family": metric_display,
                "language": language or task,
                "resource_availability": resource_map.get(language, "unknown") if language else "unknown",
                "better": better,
                "baseline": baseline_val if isinstance(baseline_val, float) else (baseline_val or ""),
                "approach_values": values_by_label,
                "approach_display_values": display_values_by_label,
                "best_approach": best_method or "",
                "best_value": best_value if isinstance(best_value, float) else "",
                "block_key": (benchmark, metric_display),
            }
        )

    title_cols = 9 + max(1, len(approach_cols))
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=title_cols)
    ws.cell(1, 1, "Compact Benchmark Comparison")
    ws.cell(1, 1).font = Font(color="FFFFFF", bold=True, size=13)
    ws.cell(1, 1).fill = PatternFill(fill_type="solid", fgColor="203864")
    ws.cell(1, 1).alignment = Alignment(horizontal="left", vertical="center")

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=title_cols)
    ws.cell(2, 1, "Comparison blocks keep FLORES metric directions separate; significance reports per-benchmark Friedman tests and group-wise Wilcoxon follow-ups.")
    ws.cell(2, 1).font = Font(color="1F2937", size=10)
    ws.cell(2, 1).fill = PatternFill(fill_type="solid", fgColor="DCE6F1")
    ws.cell(2, 1).alignment = Alignment(horizontal="left", vertical="center")

    next_row = 4
    training_summary_specs = [
        ("eval_loss", "lower", "0.0000"),
        ("train_loss", "lower", "0.0000"),
        ("train_runtime", "lower", "0.0000"),
        ("effective_tokens_per_sec", "higher", "0.0000"),
        ("num_input_tokens_seen", "higher", "0"),
        ("trainable_params", "lower", "0"),
        ("trainable_pct", "lower", "0.0000"),
    ]
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
            values = {
                label: training_metric_maps[metric_key].get(label_to_run[label])
                for label in approach_cols
                if label in label_to_run
            }
            if not any(isinstance(value, float) for value in values.values()):
                continue
            best_label, best_value = select_best(
                {key: float(value) for key, value in values.items() if isinstance(value, (int, float))},
                better,
            )
            ws.append([metric_key, *[values.get(label, "") for label in approach_cols], best_label or "", best_value if isinstance(best_value, float) else ""])
            for col_idx in range(2, 2 + len(approach_cols)):
                cell = ws.cell(data_row, col_idx)
                if isinstance(cell.value, (int, float)):
                    cell.number_format = number_format
            if approach_cols:
                apply_row_heatmap(ws, data_row, better, 2, 1 + len(approach_cols))
            data_row += 1
        next_row = data_row + 1

    headers = ["benchmark", "metric", "language", "resource_availability", "better", "baseline", *approach_cols, "best_approach", "best_value"]
    baseline_col = 6
    last_approach_col = 6 + len(approach_cols)

    def write_block(start_row: int, block_name: str, block_rows: list[dict[str, Any]]) -> int:
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=len(headers))
        ws.cell(start_row, 1, block_name)
        ws.cell(start_row, 1).fill = SECTION_FILL
        ws.cell(start_row, 1).font = SECTION_FONT
        ws.append(headers)
        style_header(ws, header_row=start_row + 1)
        row_cursor = start_row + 2
        for item in block_rows:
            ws.append(
                [
                    item["benchmark"],
                    item["metric"],
                    item["language"],
                    item["resource_availability"],
                    item["better"],
                    item["baseline"],
                    *[item["approach_values"].get(col, item["approach_display_values"].get(col, "")) for col in approach_cols],
                    item["best_approach"],
                    item["best_value"],
                ]
            )
            apply_row_heatmap(ws, row_cursor, str(item["better"]), baseline_col, max(baseline_col, last_approach_col))
            row_cursor += 1
        if block_rows:
            direction = str(block_rows[0]["better"] or "higher")
            baseline_vals = [float(row["baseline"]) for row in block_rows if isinstance(row["baseline"], (int, float))]
            avg_values = {
                col: sum(values) / len(values)
                for col in approach_cols
                if (
                    values := [
                        float(row["approach_values"][col])
                        for row in block_rows
                        if isinstance(row["approach_values"].get(col), (int, float))
                    ]
                )
            }
            avg_baseline = (sum(baseline_vals) / len(baseline_vals)) if baseline_vals else ""
            avg_scored_values = dict(avg_values)
            if isinstance(avg_baseline, float):
                avg_scored_values["baseline"] = avg_baseline
            avg_best_method, avg_best_value = select_best(avg_scored_values, direction)
            ws.append(
                [
                    "AVG",
                    "average",
                    "all",
                    "",
                    direction,
                    avg_baseline,
                    *[avg_values.get(col, "") for col in approach_cols],
                    avg_best_method or "",
                    avg_best_value if isinstance(avg_best_value, float) else "",
                ]
            )
            for col_idx in range(1, len(headers) + 1):
                ws.cell(row_cursor, col_idx).fill = AVERAGE_FILL
                ws.cell(row_cursor, col_idx).font = AVERAGE_FONT
            apply_row_heatmap(ws, row_cursor, direction, baseline_col, max(baseline_col, last_approach_col))
            row_cursor += 1
        return row_cursor + 1

    block_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        block_map[row["block_key"]].append(row)

    for block_key in sorted(block_map.keys(), key=_block_sort_key):
        benchmark, metric_family = block_key
        if benchmark == "Belebele" and metric_family == "acc_norm":
            block_name = "Belebele Block"
        else:
            block_name = f"{benchmark} ({metric_family}) Block"
        next_row = write_block(next_row, block_name, block_map[block_key])

    aggregate_headers = ["benchmark", "tier", "score_type", "baseline", *approach_cols, "best_approach", "best_value"]

    def write_aggregate_block(start_row: int, block_name: str, rows: list[dict[str, Any]]) -> int:
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

        benchmarks = sorted({key[0] for key in group_map}, key=_dataset_rank)

        ordered_keys: list[tuple[str, str]] = []
        for benchmark in benchmarks:
            for tier in ("ALL", "HIGH", "MID", "LOW"):
                ordered_keys.append((benchmark, tier))
            if (benchmark, "UNKNOWN") in group_map:
                ordered_keys.append((benchmark, "UNKNOWN"))

        for benchmark, tier in ordered_keys:
            grouped_rows = group_map.get((benchmark, tier), [])
            if not grouped_rows:
                continue
            baseline_vals = [float(item["baseline"]) for item in grouped_rows if isinstance(item.get("baseline"), (int, float))]
            approach_acc = {
                label: [
                    float(item["approach_values"][label])
                    for item in grouped_rows
                    if isinstance(item["approach_values"].get(label), (int, float))
                ]
                for label in approach_cols
            }
            approach_avg_map = {label: sum(values) / len(values) for label, values in approach_acc.items() if values}
            avg_baseline = (sum(baseline_vals) / len(baseline_vals)) if baseline_vals else ""
            direction = _shared_direction(grouped_rows) or "higher"
            scored_avg_values = dict(approach_avg_map)
            if isinstance(avg_baseline, float):
                scored_avg_values["baseline"] = avg_baseline
            best_label, best_value = select_best(scored_avg_values, direction)
            ws.append(
                [
                    benchmark,
                    tier,
                    "raw_avg",
                    avg_baseline,
                    *[approach_avg_map.get(label, "") for label in approach_cols],
                    best_label or "",
                    best_value if isinstance(best_value, float) else "",
                ]
            )
            apply_row_heatmap(ws, row_idx, direction, 4, 4 + len(approach_cols))
            row_idx += 1
        return row_idx + 1

    def write_stats_block(start_row: int) -> int:
        has_baseline = any(isinstance(row.get("baseline"), (int, float)) for row in table_rows)
        algorithm_labels = (["baseline"] if has_baseline else []) + approach_cols
        if len(algorithm_labels) < 2:
            return start_row

        significance_payload = build_significance_results(
            {
                "algorithm_labels": algorithm_labels,
                "table_rows": table_rows,
            }
        )
        note_rows = [
            (str(item.get("setting", "")), str(item.get("value", "")))
            for item in significance_payload.get("notes", [])
            if isinstance(item, dict)
        ]
        friedman_results = [
            item for item in significance_payload.get("friedman_results", []) if isinstance(item, dict)
        ]
        pairwise_results = [
            item for item in significance_payload.get("pairwise_results", []) if isinstance(item, dict)
        ]

        notes_headers = ["setting", "value"]
        notes_end_col = len(notes_headers)
        friedman_headers = [
            "benchmark_group",
            "unit_definition",
            "n_units",
            "n_algorithms",
            "friedman_statistic",
            "p_raw",
            "significant_0_05",
            "kendalls_w",
            "best_rank_approach",
            "best_avg_rank",
            *[f"rank:{label}" for label in algorithm_labels],
            "note",
        ]
        friedman_end_col = len(friedman_headers)
        pairwise_headers = [
            "benchmark_group",
            "approach_a",
            "approach_b",
            "n_units",
            "wins_a",
            "wins_b",
            "ties",
            "median_aligned_delta",
            "wilcoxon_statistic",
            "p_raw",
            "p_fdr_bh",
            "significant_fdr_0_05",
            "note",
        ]
        pairwise_end_col = len(pairwise_headers)
        section_end_col = max(notes_end_col, friedman_end_col, pairwise_end_col)

        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=section_end_col)
        ws.cell(start_row, 1, "Statistical Significance")
        ws.cell(start_row, 1).fill = SECTION_FILL
        ws.cell(start_row, 1).font = SECTION_FONT

        ws.merge_cells(start_row=start_row + 1, start_column=1, end_row=start_row + 1, end_column=notes_end_col)
        ws.cell(start_row + 1, 1, "Test Notes")
        ws.cell(start_row + 1, 1).fill = SECTION_FILL
        ws.cell(start_row + 1, 1).font = SECTION_FONT
        ws.append(notes_headers)
        style_header(ws, header_row=start_row + 2)

        row_idx = start_row + 3
        for setting, value in note_rows:
            ws.append([setting, value])
            row_idx += 1

        row_idx += 1
        ws.merge_cells(start_row=row_idx, start_column=1, end_row=row_idx, end_column=friedman_end_col)
        ws.cell(row_idx, 1, "Friedman by Benchmark Group")
        ws.cell(row_idx, 1).fill = SECTION_FILL
        ws.cell(row_idx, 1).font = SECTION_FONT
        ws.append(friedman_headers)
        style_header(ws, header_row=row_idx + 1)
        row_idx += 2
        friedman_start_row = row_idx
        rank_col_start = 10
        rank_col_end = rank_col_start + len(algorithm_labels) - 1
        for friedman_result in friedman_results:
            ws.append(
                [
                    friedman_result["group_label"],
                    friedman_result["unit_definition"],
                    friedman_result["n_units"],
                    friedman_result["n_algorithms"],
                    _format_stat(friedman_result["friedman_statistic"]),
                    _format_stat(friedman_result["p_raw"]),
                    "yes" if friedman_result["significant"] else "no" if friedman_result["p_raw"] is not None else "",
                    _format_stat(friedman_result["kendalls_w"]),
                    friedman_result["best_rank_approach"],
                    _format_stat(friedman_result["best_avg_rank"]),
                    *[_format_stat(friedman_result["avg_ranks"].get(label)) for label in algorithm_labels],
                    friedman_result["note"],
                ]
            )
            ws.cell(row_idx, 5).number_format = "0.0000"
            ws.cell(row_idx, 6).number_format = "0.00000000"
            ws.cell(row_idx, 8).number_format = "0.0000"
            ws.cell(row_idx, 10).number_format = "0.0000"
            for col_idx in range(rank_col_start, rank_col_end + 1):
                ws.cell(row_idx, col_idx).number_format = "0.0000"
            style_significance_cell(ws.cell(row_idx, 7))
            if friedman_result["best_rank_approach"]:
                style_best_cell(ws.cell(row_idx, 9))
                best_label = str(friedman_result["best_rank_approach"])
                if best_label in algorithm_labels:
                    best_col = rank_col_start + algorithm_labels.index(best_label)
                    style_best_cell(ws.cell(row_idx, best_col))
            row_idx += 1
        friedman_end_row = row_idx - 1
        apply_pvalue_heatmap(ws, friedman_start_row, friedman_end_row, 6)
        apply_positive_magnitude_heatmap(ws, friedman_start_row, friedman_end_row, 8)

        row_idx += 1
        ws.merge_cells(start_row=row_idx, start_column=1, end_row=row_idx, end_column=pairwise_end_col)
        ws.cell(row_idx, 1, "Pairwise Wilcoxon by Benchmark Group (BH-FDR corrected)")
        ws.cell(row_idx, 1).fill = SECTION_FILL
        ws.cell(row_idx, 1).font = SECTION_FONT
        ws.append(pairwise_headers)
        style_header(ws, header_row=row_idx + 1)
        row_idx += 2
        pairwise_start_row = row_idx
        if not pairwise_results:
            ws.append(["", "", "", "", "", "", "", "", "", "", "", "", "no benchmark groups passed Friedman at alpha=0.05"])
            row_idx += 1
        for result in pairwise_results:
            ws.append(
                [
                    result["group_label"],
                    result["label_a"],
                    result["label_b"],
                    result["n_pairs"],
                    result["wins_a"],
                    result["wins_b"],
                    result["ties"],
                    _format_stat(result["median_delta"]),
                    _format_stat(result["wilcoxon_statistic"]),
                    _format_stat(result["p_raw"]),
                    _format_stat(result["p_fdr"]),
                    "yes" if result["significant_fdr"] else "no" if result["p_fdr"] is not None else "",
                    result["note"],
                ]
            )
            ws.cell(row_idx, 8).number_format = "0.0000"
            ws.cell(row_idx, 9).number_format = "0.0000"
            ws.cell(row_idx, 10).number_format = "0.00000000"
            ws.cell(row_idx, 11).number_format = "0.00000000"
            style_significance_cell(ws.cell(row_idx, 12))
            row_idx += 1
        pairwise_end_row = row_idx - 1
        apply_signed_effect_heatmap(ws, pairwise_start_row, pairwise_end_row, 8)
        apply_pvalue_heatmap(ws, pairwise_start_row, pairwise_end_row, 10)
        apply_pvalue_heatmap(ws, pairwise_start_row, pairwise_end_row, 11)

        return row_idx + 1

    if table_rows:
        next_row = write_aggregate_block(next_row, "Benchmark Tier Averages", table_rows)
        next_row = write_stats_block(next_row)

    ws.freeze_panes = "A4"
    ws.sheet_view.showGridLines = False

    approach_width = max(12, min(40, max((len(col) for col in approach_cols), default=12) + 2))
    widths = {
        1: 18,
        2: 26,
        3: 18,
        4: 20,
        5: 12,
        6: 14,
    }
    for idx, _ in enumerate(approach_cols, start=7):
        widths[idx] = approach_width
    widths[7 + len(approach_cols)] = max(18, approach_width)
    widths[8 + len(approach_cols)] = 14

    for col_idx in range(1, 20 + len(approach_cols)):
        widths.setdefault(col_idx, 16)

    for col_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width
