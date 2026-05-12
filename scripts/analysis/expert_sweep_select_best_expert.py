import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.rank_sweep_select_best_rank import (  # noqa: E402
    PRIMARY_METRICS,
    _normalized_scores,
    _read_json,
    _safe_float,
    _task_benchmark,
    _task_direction,
    _task_language,
)

DEFAULT_ROOTS = [
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_working_calibrated",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_ready",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_final_rescue_nosync",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_deterministic_budget_filtered_save",
    "outputs/pipeline_runs/moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_signal_3p6h",
]
DEFAULT_OUTPUT_DIR = (
    "outputs/pipeline_runs/"
    "moe_study_expert_head_sweep_r2_noctua2_a100_4gpu_10_langs_ready/"
    "analysis/expert_selection"
)

RESULT_TS_RE = re.compile(r"results_(.+)\.json$")
EXPERT_RE = re.compile(r"_e(\d+)(?:_|$)")
TOPK_RE = re.compile(r"_k(\d+)(?:_|$)")
HEADS_RE = re.compile(r"_heads(\d+)$")
NUP_RE = re.compile(r"_nup(\d+)$")
COLA_B_RE = re.compile(r"_b(\d+)$")


def _result_timestamp(path: Path) -> str:
    match = RESULT_TS_RE.match(path.name)
    return match.group(1) if match else ""


def _observed_run_ids(roots: list[Path]) -> list[str]:
    ids: set[str] = set()
    for root in roots:
        for subdir in ("runs", "benchmarks"):
            path = root / subdir
            if path.exists():
                ids.update(child.name for child in path.iterdir() if child.is_dir())
    return sorted(ids)


def _kind_from_id(run_id: str) -> str:
    if run_id.startswith("adamoe_1"):
        return "mola"
    if run_id.startswith("mov_"):
        return "movlora"
    return run_id.split("_", 1)[0]


def _metadata(run_id: str) -> dict[str, Any]:
    kind = _kind_from_id(run_id)
    expert_match = EXPERT_RE.search(run_id)
    topk_match = TOPK_RE.search(run_id)
    heads_match = HEADS_RE.search(run_id)
    nup_match = NUP_RE.search(run_id)
    cola_b_match = COLA_B_RE.search(run_id)

    expert_count = None
    axis_label = "experts"
    if expert_match:
        expert_count = int(expert_match.group(1))
    elif heads_match:
        expert_count = int(heads_match.group(1))
        axis_label = "heads"
    elif nup_match:
        expert_count = int(nup_match.group(1))
        axis_label = "up_projections"
    elif cola_b_match:
        expert_count = int(cola_b_match.group(1))
        axis_label = "cola_B"

    top_k = int(topk_match.group(1)) if topk_match else None
    active_experts = top_k if top_k is not None else expert_count
    return {
        "run_id": run_id,
        "kind": kind,
        "expert_count": expert_count,
        "top_k": top_k,
        "active_experts": active_experts,
        "axis_label": axis_label,
    }


def _latest_result(run_id: str, roots: list[Path]) -> tuple[Path | None, Path | None]:
    candidates: list[tuple[str, float, Path, Path]] = []
    for root in roots:
        run_dir = root / "benchmarks" / run_id
        if not run_dir.exists():
            continue
        for path in run_dir.rglob("results_*.json"):
            candidates.append((_result_timestamp(path), path.stat().st_mtime, root, path))
    if not candidates:
        return None, None
    _, _, root, path = max(candidates, key=lambda item: (item[0], item[1], str(item[3])))
    return root, path


def _ignore_per_task_json(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("results_")
        or name.startswith("selected_tasks")
        or name in {"benchmark_summary.json", "summary.json"}
        or name.endswith("_check.json")
    )


def _per_task_result_candidate(root: Path, run_id: str) -> tuple[float, list[Path], dict[str, Any]] | None:
    run_dir = root / "benchmarks" / run_id
    if not run_dir.exists():
        return None

    files: list[Path] = []
    merged: dict[str, Any] = {}
    for path in sorted(run_dir.rglob("*.json")):
        if _ignore_per_task_json(path):
            continue
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        results = data.get("results", {})
        if not isinstance(results, dict):
            continue
        task_results = {
            str(task): metrics
            for task, metrics in results.items()
            if _task_benchmark(str(task)) and isinstance(metrics, dict)
        }
        if not task_results:
            continue
        files.append(path)
        merged.update(task_results)

    if not merged:
        return None
    return max(path.stat().st_mtime for path in files), files, merged


def _result_payload(run_id: str, roots: list[Path]) -> tuple[Path | None, list[Path], dict[str, Any]]:
    combined_candidates: list[tuple[str, float, Path, Path]] = []
    per_task_candidates: list[tuple[float, Path, list[Path], dict[str, Any]]] = []
    for root in roots:
        run_dir = root / "benchmarks" / run_id
        if not run_dir.exists():
            continue
        for path in run_dir.rglob("results_*.json"):
            combined_candidates.append((_result_timestamp(path), path.stat().st_mtime, root, path))
        per_task = _per_task_result_candidate(root, run_id)
        if per_task:
            max_mtime, files, results = per_task
            per_task_candidates.append((max_mtime, root, files, results))

    for _timestamp, _mtime, root, path in sorted(
        combined_candidates,
        key=lambda item: (item[0], item[1], str(item[3])),
        reverse=True,
    ):
        data = _read_json(path)
        results = data.get("results", {})
        if isinstance(results, dict) and results:
            return root, [path], results

    if per_task_candidates:
        _mtime, root, files, results = max(per_task_candidates, key=lambda item: (item[0], len(item[2]), str(item[1])))
        return root, files, results

    return None, [], {}


def _collect(roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for run_id in _observed_run_ids(roots):
        metadata = _metadata(run_id)
        if metadata["expert_count"] is None:
            continue
        root, result_paths, results = _result_payload(run_id, roots)
        if len(result_paths) == 1:
            result_json = str(result_paths[0])
        elif result_paths:
            result_json = f"per_task_json:{len(result_paths)}:{result_paths[0].parent}"
        else:
            result_json = ""
        run_rows.append(
            {
                **metadata,
                "benchmark_root": root.name if root else "",
                "result_json": result_json,
            }
        )
        if not isinstance(results, dict):
            continue
        for task, metrics in results.items():
            benchmark = _task_benchmark(str(task))
            if not benchmark or not isinstance(metrics, dict):
                continue
            metric_name, better = PRIMARY_METRICS[benchmark]
            value = _safe_float(metrics.get(metric_name))
            if value is None:
                continue
            raw_rows.append(
                {
                    **metadata,
                    "benchmark": benchmark,
                    "task": task,
                    "language": _task_language(str(task)),
                    "direction": _task_direction(str(task)),
                    "metric": metric_name.split(",", 1)[0],
                    "better": better,
                    "value": value,
                }
            )
    return run_rows, raw_rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _summarize_runs(run_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run_benchmark: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in scored_rows:
        by_run_benchmark[(str(row["run_id"]), str(row["benchmark"]))].append(float(row["normalized_score"]))

    out: list[dict[str, Any]] = []
    for run in run_rows:
        run_id = str(run["run_id"])
        bench_scores = {bench: _mean(by_run_benchmark.get((run_id, bench), [])) for bench in PRIMARY_METRICS}
        aggregate = _mean([value for value in bench_scores.values() if math.isfinite(value)])
        expert_count = run.get("expert_count")
        active_experts = run.get("active_experts")
        out.append(
            {
                **run,
                "aggregate_score": aggregate,
                "belebele_score": bench_scores["Belebele"],
                "flores_score": bench_scores["FLORES"],
                "sib200_score": bench_scores["SIB200"],
                "taxi1500_score": bench_scores["Taxi1500"],
                "score_per_expert": aggregate / float(expert_count) if expert_count else float("nan"),
                "score_per_active_expert": aggregate / float(active_experts) if active_experts else float("nan"),
                "num_scored_items": sum(len(by_run_benchmark.get((run_id, bench), [])) for bench in PRIMARY_METRICS),
            }
        )
    return sorted(out, key=lambda row: (-(row["aggregate_score"] if math.isfinite(row["aggregate_score"]) else -1), str(row["run_id"])))


def _exact_metric_value(row: dict[str, Any]) -> float:
    value = float(row["value"])
    return 2.0**value if row["benchmark"] == "FLORES" else value


def _summarize_exact_runs(run_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run_benchmark: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in raw_rows:
        by_run_benchmark[(str(row["run_id"]), str(row["benchmark"]))].append(_exact_metric_value(row))

    out: list[dict[str, Any]] = []
    for run in run_rows:
        run_id = str(run["run_id"])
        values = {bench: _mean(by_run_benchmark.get((run_id, bench), [])) for bench in PRIMARY_METRICS}
        if not any(math.isfinite(value) for value in values.values()):
            continue
        out.append(
            {
                **run,
                "belebele_acc": values["Belebele"],
                "flores_byte_perplexity": values["FLORES"],
                "sib200_f1": values["SIB200"],
                "taxi1500_f1": values["Taxi1500"],
            }
        )
    return sorted(out, key=lambda row: str(row["run_id"]))


def _summarize_group(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not math.isfinite(float(row.get("aggregate_score", float("nan")))):
            continue
        if any(row.get(field) in ("", None) for field in fields):
            continue
        groups[tuple(row[field] for field in fields)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in groups.items():
        item = {field: value for field, value in zip(fields, key)}
        scores = [float(row["aggregate_score"]) for row in group]
        item.update(
            {
                "n_runs": len(group),
                "mean_aggregate_score": _mean(scores),
                "best_aggregate_score": max(scores),
                "best_run_id": max(group, key=lambda row: float(row["aggregate_score"]))["run_id"],
                "mean_score_per_expert": _mean([float(row["score_per_expert"]) for row in group if math.isfinite(float(row["score_per_expert"]))]),
                "mean_score_per_active_expert": _mean(
                    [float(row["score_per_active_expert"]) for row in group if math.isfinite(float(row["score_per_active_expert"]))]
                ),
                "mean_belebele_score": _mean([float(row["belebele_score"]) for row in group if math.isfinite(float(row["belebele_score"]))]),
                "mean_flores_score": _mean([float(row["flores_score"]) for row in group if math.isfinite(float(row["flores_score"]))]),
                "mean_sib200_score": _mean([float(row["sib200_score"]) for row in group if math.isfinite(float(row["sib200_score"]))]),
                "mean_taxi1500_score": _mean([float(row["taxi1500_score"]) for row in group if math.isfinite(float(row["taxi1500_score"]))]),
            }
        )
        out.append(item)
    return sorted(out, key=lambda row: (-float(row["mean_aggregate_score"]), tuple(str(row[field]) for field in fields)))


def _summarize_exact_group(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if any(row.get(field) in ("", None) for field in fields):
            continue
        groups[tuple(row[field] for field in fields)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in groups.items():
        item = {field: value for field, value in zip(fields, key)}
        item.update(
            {
                "n_runs": len(group),
                "mean_belebele_acc": _mean([float(row["belebele_acc"]) for row in group if math.isfinite(float(row["belebele_acc"]))]),
                "mean_flores_byte_perplexity": _mean(
                    [float(row["flores_byte_perplexity"]) for row in group if math.isfinite(float(row["flores_byte_perplexity"]))]
                ),
                "mean_sib200_f1": _mean([float(row["sib200_f1"]) for row in group if math.isfinite(float(row["sib200_f1"]))]),
                "mean_taxi1500_f1": _mean([float(row["taxi1500_f1"]) for row in group if math.isfinite(float(row["taxi1500_f1"]))]),
            }
        )
        out.append(item)
    return sorted(out, key=lambda row: tuple(str(row[field]) for field in fields))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plain_count_ticks(ax, counts: list[int]) -> None:
    ticks = [tick for tick in [2, 4, 8, 16, 32] if tick in set(counts)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])


def _row_scaled(values: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in values:
        finite = [value for value in row if math.isfinite(value)]
        if not finite:
            out.append([float("nan") for _ in row])
            continue
        lo, hi = min(finite), max(finite)
        span = hi - lo if hi != lo else 1.0
        out.append([(value - lo) / span if math.isfinite(value) else float("nan") for value in row])
    return out


def _draw_heatmap(ax, rows: list[str], cols: list[str], values: list[list[float]], title: str, *, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "viridis") -> Any:
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for y, _row in enumerate(rows):
        for x, _col in enumerate(cols):
            value = values[y][x]
            if math.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 0.55 else "black")
    return image


def _write_plots(
    output_dir: Path,
    run_scores: list[dict[str, Any]],
    kind_exact_summary: list[dict[str, Any]],
    expert_summary: list[dict[str, Any]],
    kind_expert_summary: list[dict[str, Any]],
    large_topk_summary: list[dict[str, Any]],
    large_topk_exact_summary: list[dict[str, Any]],
) -> None:
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    counts = sorted({int(row["expert_count"]) for row in run_scores if row.get("expert_count")})

    fig, ax = plt.subplots(figsize=(10, 6))
    top = list(reversed(run_scores[:15]))
    ax.barh([row["run_id"] for row in top], [float(row["aggregate_score"]) for row in top], color="#4C78A8")
    ax.set(xlabel="Normalized aggregate score", title="Top expert/head sweep checkpoints", xlim=(0, 1))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "expert_top_runs.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kind_expert_summary:
        by_kind[str(row["kind"])].append(row)
    for kind, rows in sorted(by_kind.items()):
        rows = sorted(rows, key=lambda row: int(row["expert_count"]))
        ax.plot([int(row["expert_count"]) for row in rows], [float(row["mean_aggregate_score"]) for row in rows], marker="o", label=kind)
    ax.set_xscale("log", base=2)
    _plain_count_ticks(ax, counts)
    ax.set(xlabel="Expert/head count", ylabel="Mean normalized aggregate score", title="Mean score by count and method family", ylim=(0, 1.04))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "expert_score_by_kind_count.png", dpi=180)
    plt.close(fig)

    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    rows = sorted(expert_summary, key=lambda row: int(row["expert_count"]))
    ax_curve.plot([int(row["expert_count"]) for row in rows], [float(row["mean_aggregate_score"]) for row in rows], marker="o", label="mean score")
    ax_curve.plot([int(row["expert_count"]) for row in rows], [float(row["best_aggregate_score"]) for row in rows], marker="o", label="best run")
    ax_curve.set_xscale("log", base=2)
    _plain_count_ticks(ax_curve, counts)
    ax_curve.set(xlabel="Expert/head count", ylabel="Normalized aggregate score", title="Overall expert-count trend", ylim=(0, 1.04))
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(fontsize=8)

    pivot_rows = sorted({str(row["kind"]) for row in kind_expert_summary})
    pivot_cols = [str(count) for count in counts]
    values = [[float("nan") for _ in pivot_cols] for _ in pivot_rows]
    lookup = {(str(row["kind"]), str(int(row["expert_count"]))): float(row["mean_aggregate_score"]) for row in kind_expert_summary}
    for y, kind in enumerate(pivot_rows):
        for x, count in enumerate(pivot_cols):
            values[y][x] = lookup.get((kind, count), float("nan"))
    image = ax_heat.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_heat.set_title("Mean score heatmap")
    ax_heat.set_xticks(range(len(pivot_cols)), pivot_cols)
    ax_heat.set_yticks(range(len(pivot_rows)), pivot_rows)
    for y, kind in enumerate(pivot_rows):
        for x, count in enumerate(pivot_cols):
            value = values[y][x]
            if math.isfinite(value):
                ax_heat.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value < 0.55 else "black")
    fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots / "expert_count_curve_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    exact_fields = [
        ("Belebele", "mean_belebele_acc", "Accuracy", (0.0, 1.02)),
        ("FLORES", "mean_flores_byte_perplexity", "Byte perplexity", (1.0, 2.5)),
        ("SIB200", "mean_sib200_f1", "F1", (0.0, 1.02)),
        ("Taxi1500", "mean_taxi1500_f1", "F1", (0.0, 1.02)),
    ]
    for ax, (label, field, ylabel, ylim) in zip(axes.flat, exact_fields):
        for kind in sorted({str(row["kind"]) for row in kind_exact_summary}):
            rows_for_kind = sorted([row for row in kind_exact_summary if row["kind"] == kind], key=lambda row: int(row["expert_count"]))
            if not rows_for_kind:
                continue
            x = [int(row["expert_count"]) for row in rows_for_kind]
            y = [float(row[field]) for row in rows_for_kind]
            if label == "FLORES":
                y_plot = [min(value, 2.5) for value in y]
                line = ax.plot(x, y_plot, marker="o", linewidth=1.2, markersize=4, label=kind)[0]
                capped_x = [xx for xx, yy in zip(x, y) if yy > 2.5]
                if capped_x:
                    ax.scatter(capped_x, [2.5] * len(capped_x), marker="^", s=38, color=line.get_color())
            else:
                ax.plot(x, y, marker="o", linewidth=1.2, markersize=4, label=kind)
        ax.set_xscale("log", base=2)
        _plain_count_ticks(ax, counts)
        ax.set(title=f"{label} by count", xlabel="Expert/head count", ylabel=ylabel, ylim=ylim)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(plots / "expert_exact_benchmark_family_by_count.png", dpi=180)
    plt.close(fig)

    large_counts = [8, 16, 32]
    topk_values = [1, 2, 4, 8, 16]
    topk_rows = [f"{row['kind']}_e{int(row['expert_count'])}" for row in large_topk_summary]
    topk_rows = sorted(dict.fromkeys(topk_rows), key=lambda value: (int(value.rsplit("_e", 1)[1]), value))
    cols = [str(value) for value in topk_values]
    lookup = {
        (f"{row['kind']}_e{int(row['expert_count'])}", str(int(row["top_k"]))): float(row["mean_aggregate_score"])
        for row in large_topk_summary
        if int(row["expert_count"]) in large_counts
    }
    values = [[lookup.get((row, col), float("nan")) for col in cols] for row in topk_rows]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(topk_rows))))
    image = _draw_heatmap(ax, topk_rows, cols, values, "Large-count top-k effect: aggregate score")
    ax.set_xlabel("top-k")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots / "expert_topk_large_counts_aggregate_heatmap.png", dpi=180)
    plt.close(fig)

    exact_lookup_fields = [
        ("Belebele acc", "mean_belebele_acc", False),
        ("FLORES byte ppl", "mean_flores_byte_perplexity", True),
        ("SIB200 F1", "mean_sib200_f1", False),
        ("Taxi1500 F1", "mean_taxi1500_f1", False),
    ]
    exact_lookup = {
        (f"{row['kind']}_e{int(row['expert_count'])}", str(int(row["top_k"]))): row
        for row in large_topk_exact_summary
        if int(row["expert_count"]) in large_counts
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, max(7, 0.42 * len(topk_rows))))
    for ax, (title, field, cap_flores) in zip(axes.flat, exact_lookup_fields):
        raw_values = []
        label_values = []
        for row_label in topk_rows:
            raw_row = []
            label_row = []
            for col in cols:
                item = exact_lookup.get((row_label, col))
                value = float(item[field]) if item and math.isfinite(float(item[field])) else float("nan")
                label_row.append(value)
                raw_row.append(min(value, 2.5) if cap_flores and math.isfinite(value) else value)
            raw_values.append(raw_row)
            label_values.append(label_row)
        heat_values = _row_scaled(raw_values)
        image = ax.imshow(heat_values, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(topk_rows)), topk_rows)
        for y, row_label in enumerate(topk_rows):
            for x, col in enumerate(cols):
                value = label_values[y][x]
                if math.isfinite(value):
                    text = ">2.5" if cap_flores and value > 2.5 else f"{value:.2f}"
                    color = "white" if heat_values[y][x] < 0.55 else "black"
                    ax.text(x, y, text, ha="center", va="center", fontsize=7, color=color)
        ax.set_xlabel("top-k")
    fig.tight_layout()
    fig.savefig(plots / "expert_topk_large_counts_benchmark_heatmaps.png", dpi=180)
    plt.close(fig)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else ""
    return str(value)


def _efficient_choice(expert_summary: list[dict[str, Any]], tolerance: float = 0.98) -> dict[str, Any]:
    if not expert_summary:
        return {}
    best = max(float(row["mean_aggregate_score"]) for row in expert_summary)
    threshold = best * tolerance
    candidates = [row for row in expert_summary if float(row["mean_aggregate_score"]) >= threshold]
    return min(candidates, key=lambda row: int(row["expert_count"])) if candidates else {}


def _summary_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Count | Runs | Mean score | Best score | Best run | Score/expert | Score/active expert |",
            "|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['expert_count']} | {row['n_runs']} | {_fmt(row['mean_aggregate_score'])} | {_fmt(row['best_aggregate_score'])} | `{row['best_run_id']}` | {_fmt(row['mean_score_per_expert'])} | {_fmt(row['mean_score_per_active_expert'])} |"
        )


def _write_markdown(
    path: Path,
    run_scores: list[dict[str, Any]],
    expert_summary: list[dict[str, Any]],
    routed_moe_summary: list[dict[str, Any]],
    kind_expert_summary: list[dict[str, Any]],
) -> None:
    best_run = run_scores[0] if run_scores else {}
    best_count = expert_summary[0] if expert_summary else {}
    efficient = _efficient_choice(expert_summary)
    routed_best = routed_moe_summary[0] if routed_moe_summary else {}
    routed_efficient = _efficient_choice(routed_moe_summary)
    best_by_kind: dict[str, dict[str, Any]] = {}
    for row in kind_expert_summary:
        best_by_kind.setdefault(str(row["kind"]), row)
    lines = [
        "# Expert/Head Sweep Selection",
        "",
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "Scoring uses primary metrics only: Belebele `acc_norm`, FLORES `bits_per_byte` (lower is better), SIB200 `f1`, and Taxi1500 `f1`.",
        "Each task metric is min-max normalized across observed expert/head sweep runs; run scores macro-average benchmark families.",
        "",
        "## Recommendation",
        "",
        f"- Best single run: `{best_run.get('run_id', '')}` with score {_fmt(best_run.get('aggregate_score', ''))}.",
        f"- Best mean expert/head count: `{best_count.get('expert_count', '')}` with mean score {_fmt(best_count.get('mean_aggregate_score', ''))}.",
        f"- Efficient count at >=98% of best mean score: `{efficient.get('expert_count', '')}` with mean score {_fmt(efficient.get('mean_aggregate_score', ''))}.",
        f"- For the shared routed-MoE subset (`mola`, `movlora`, `hmora`), best mean count is `{routed_best.get('expert_count', '')}` and the >=98% efficient count is `{routed_efficient.get('expert_count', '')}`.",
        "",
        "Interpretation: the count-wise average favors small counts because higher expert counts are slower/harder to optimize under this fixed budget, and because the feasible matrix is unbalanced. The strongest single run can still come from a larger-count family, but the count-level efficient choice reflects the mean over all observed methods at that count.",
        "",
        "## Plots",
        "",
        "![expert_top_runs](plots/expert_top_runs.png)",
        "",
        "![expert_score_by_kind_count](plots/expert_score_by_kind_count.png)",
        "",
        "![expert_count_curve_heatmap](plots/expert_count_curve_heatmap.png)",
        "",
        "![expert_exact_benchmark_family_by_count](plots/expert_exact_benchmark_family_by_count.png)",
        "",
        "![expert_topk_large_counts_aggregate_heatmap](plots/expert_topk_large_counts_aggregate_heatmap.png)",
        "",
        "![expert_topk_large_counts_benchmark_heatmaps](plots/expert_topk_large_counts_benchmark_heatmaps.png)",
        "",
        "## Expert Count Summary",
        "",
    ]
    _summary_table(lines, expert_summary)
    lines.extend(["", "## Routed MoE Count Summary", ""])
    _summary_table(lines, routed_moe_summary)
    lines.extend(["", "## Best Count By Family", "", "| Kind | Count | Score | Best run | Runs |", "|---|---:|---:|---|---:|"])
    for kind in sorted(best_by_kind):
        row = best_by_kind[kind]
        lines.append(f"| `{kind}` | {row['expert_count']} | {_fmt(row['mean_aggregate_score'])} | `{row['best_run_id']}` | {row['n_runs']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the best expert/head count from combined Noctua2 expert sweep results.")
    parser.add_argument("--root", action="append", default=[], help="Output root to scan. May be repeated.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    roots = [REPO_ROOT / value for value in (args.root or DEFAULT_ROOTS)]
    output_dir = REPO_ROOT / args.output_dir
    run_rows, raw_rows = _collect(roots)
    if not raw_rows:
        raise SystemExit("No expert/head benchmark result metrics found.")
    scored_rows = _normalized_scores(raw_rows)
    run_scores = _summarize_runs(run_rows, scored_rows)
    exact_run_scores = _summarize_exact_runs(run_rows, raw_rows)
    scored_run_scores = [row for row in run_scores if math.isfinite(float(row.get("aggregate_score", float("nan"))))]
    expert_summary = _summarize_group(scored_run_scores, ["expert_count"])
    routed_moe_summary = _summarize_group([row for row in scored_run_scores if row["kind"] in {"mola", "movlora", "hmora"}], ["expert_count"])
    kind_expert_summary = _summarize_group(scored_run_scores, ["kind", "expert_count"])
    kind_exact_summary = _summarize_exact_group(exact_run_scores, ["kind", "expert_count"])
    large_topk_rows = [
        row
        for row in scored_run_scores
        if row.get("top_k") not in ("", None) and row.get("expert_count") in {8, 16, 32}
    ]
    large_topk_summary = _summarize_group(large_topk_rows, ["kind", "expert_count", "top_k"])
    large_topk_exact_summary = _summarize_exact_group(
        [
            row
            for row in exact_run_scores
            if row.get("top_k") not in ("", None) and row.get("expert_count") in {8, 16, 32}
        ],
        ["kind", "expert_count", "top_k"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "expert_selection_raw_metrics.csv", scored_rows)
    _write_csv(output_dir / "expert_selection_run_scores.csv", run_scores)
    _write_csv(output_dir / "expert_selection_exact_run_scores.csv", exact_run_scores)
    _write_csv(output_dir / "expert_selection_expert_count_summary.csv", expert_summary)
    _write_csv(output_dir / "expert_selection_routed_moe_count_summary.csv", routed_moe_summary)
    _write_csv(output_dir / "expert_selection_kind_expert_summary.csv", kind_expert_summary)
    _write_csv(output_dir / "expert_selection_kind_exact_summary.csv", kind_exact_summary)
    _write_csv(output_dir / "expert_selection_large_topk_summary.csv", large_topk_summary)
    _write_csv(output_dir / "expert_selection_large_topk_exact_summary.csv", large_topk_exact_summary)
    _write_plots(
        output_dir,
        scored_run_scores,
        kind_exact_summary,
        expert_summary,
        kind_expert_summary,
        large_topk_summary,
        large_topk_exact_summary,
    )
    _write_markdown(output_dir / "expert_selection_report.md", scored_run_scores, expert_summary, routed_moe_summary, kind_expert_summary)

    print(f"[expert_sweep_select_best_expert] wrote {output_dir}")
    if expert_summary:
        efficient = _efficient_choice(expert_summary)
        print(
            "[expert_sweep_select_best_expert] best_count="
            f"{expert_summary[0]['expert_count']} score={float(expert_summary[0]['mean_aggregate_score']):.4f}"
        )
        print(
            "[expert_sweep_select_best_expert] efficient_count="
            f"{efficient.get('expert_count')} score={float(efficient.get('mean_aggregate_score', float('nan'))):.4f}"
        )
    if scored_run_scores:
        print(
            "[expert_sweep_select_best_expert] best_run="
            f"{scored_run_scores[0]['run_id']} score={float(scored_run_scores[0]['aggregate_score']):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
