import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_tools.reporting.significance import build_significance_results

DEFAULT_PROFILE = "moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated"
DEFAULT_ROOTS = [
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated",
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_kvdown",
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_rescue_moelora_vonly",
    "outputs/pipeline_runs/moe_study_rank_expert_sweep_noctua2_a100_4gpu_failed_recovery_full",
]
DEFAULT_OUTPUT_DIR = (
    "outputs/pipeline_runs/"
    "moe_study_rank_expert_sweep_noctua2_a100_4gpu_10_langs_alltargets_calibrated/"
    "analysis/rank_selection"
)

RANK_RE = re.compile(r"_r(\d+)$")
TOTAL_RANK_RE = re.compile(r"_tr(\d+)_")
RESULT_TS_RE = re.compile(r"results_(.+)\.json$")

PRIMARY_METRICS = {
    "Belebele": ("acc_norm,none", "higher"),
    "FLORES": ("bits_per_byte,none", "lower"),
    "SIB200": ("f1,none", "higher"),
    "Taxi1500": ("f1,none", "higher"),
}


def _load_profile(config_dir: Path, profile_name: str) -> dict[str, Any]:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from pipeline.config_loader import deep_merge, load_pipeline_config
    from pipeline.runtime import _resolve_profile

    profile = _resolve_profile(load_pipeline_config(config_dir), profile_name)
    profile["_common_lora"] = profile["training"].get("lora", {})
    profile["_deep_merge"] = deep_merge
    return profile


def _task_benchmark(task: str) -> str | None:
    if task.startswith("belebele_"):
        return "Belebele"
    if task.startswith("flores_en_perplexity_"):
        return "FLORES"
    if task.startswith("sib200_"):
        return "SIB200"
    if task.startswith("taxi1500_"):
        return "Taxi1500"
    return None


def _task_language(task: str) -> str:
    if task.startswith("belebele_"):
        return task.replace("belebele_", "", 1)
    if task.startswith("sib200_"):
        return task.replace("sib200_", "", 1)
    if task.startswith("taxi1500_"):
        return task.replace("taxi1500_", "", 1)
    if task.startswith("flores_en_perplexity_"):
        pair = task.replace("flores_en_perplexity_", "", 1)
        if "-" in pair:
            src, tgt = pair.split("-", 1)
            return tgt if src == "eng_Latn" else src
        return pair
    return task


def _task_direction(task: str) -> str:
    pair = task.replace("flores_en_perplexity_", "", 1)
    if task.startswith("flores_en_perplexity_") and "-" in pair:
        src, tgt = pair.split("-", 1)
        if src == "eng_Latn":
            return "en_to_x"
        if tgt == "eng_Latn":
            return "x_to_en"
    return ""


def _result_timestamp(path: Path) -> str:
    match = RESULT_TS_RE.match(path.name)
    return match.group(1) if match else ""


def _latest_result(run_dir: Path) -> Path | None:
    results = sorted(run_dir.rglob("results_*.json"), key=lambda p: (_result_timestamp(p), p.stat().st_mtime))
    if results:
        return results[-1]
    candidates = [
        path
        for path in sorted(run_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime)
        if not path.name.startswith(("selected_tasks", "selected_tasks_summary", "benchmark_summary", "_check"))
    ]
    for path in reversed(candidates):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            return path
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _run_metadata(run: dict[str, Any], common_lora: dict[str, Any], deep_merge) -> dict[str, Any]:
    run_id = str(run["id"])
    kind = str(run["kind"])
    params = run.get("params", {}) if isinstance(run.get("params"), dict) else {}
    lora = deep_merge(common_lora, run.get("lora", {}))
    configured_rank = _as_int(lora.get("rank"))
    total_match = TOTAL_RANK_RE.search(run_id)
    rank_match = RANK_RE.search(run_id)
    total_rank = _as_int(total_match.group(1)) if total_match else None
    rank_label = ""
    rank_semantics = "lora_rank"
    if total_rank is not None:
        rank_label = f"tr{total_rank}"
        rank_semantics = "total_rank"
    elif rank_match:
        rank_label = f"r{rank_match.group(1)}"
        total_rank = configured_rank
    elif kind == "movlora":
        rank_label = "rankless"
        rank_semantics = "rankless_ia3"

    return {
        "run_id": run_id,
        "kind": kind,
        "rank_label": rank_label,
        "total_rank": total_rank,
        "configured_lora_rank": configured_rank,
        "rank_semantics": rank_semantics,
        "num_experts": (
            params.get("moelora_num_experts")
            or params.get("mola_num_experts")
            or params.get("hmora_num_experts")
            or params.get("cola_num_experts")
            or params.get("hydralora_num_experts")
        ),
        "top_k": (
            params.get("moelora_sparse_top_k")
            or params.get("mola_top_k")
            or params.get("hmora_top_k")
            or params.get("cola_top_k")
            or params.get("hydralora_top_k")
        ),
    }


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _find_run_root(run_id: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if (root / "benchmarks" / run_id).exists():
            return root
    return None


def _collect_raw_rows(profile: dict[str, Any], roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_by_run: dict[str, dict[str, Any]] = {}
    run_rows: list[dict[str, Any]] = []
    for run in profile["training"]["runs"]:
        metadata = _run_metadata(run, profile["_common_lora"], profile["_deep_merge"])
        metadata_by_run[metadata["run_id"]] = metadata
        root = _find_run_root(metadata["run_id"], roots)
        result_path = _latest_result(root / "benchmarks" / metadata["run_id"]) if root else None
        run_rows.append({**metadata, "benchmark_root": root.name if root else "", "result_json": str(result_path) if result_path else ""})

    raw_rows: list[dict[str, Any]] = []
    for run_row in run_rows:
        result_path = Path(str(run_row["result_json"])) if run_row["result_json"] else None
        if not result_path or not result_path.exists():
            continue
        data = _read_json(result_path)
        results = data.get("results", {})
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
                    **{key: run_row[key] for key in ("run_id", "kind", "rank_label", "total_rank", "configured_lora_rank", "rank_semantics")},
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


def _normalized_scores(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["benchmark"]), str(row["task"]), str(row["metric"]))].append(row)

    scored: list[dict[str, Any]] = []
    for rows in grouped.values():
        values = [float(row["value"]) for row in rows]
        lo = min(values)
        hi = max(values)
        span = hi - lo
        for row in rows:
            value = float(row["value"])
            if span == 0:
                score = 0.5
            elif row["better"] == "lower":
                score = (hi - value) / span
            else:
                score = (value - lo) / span
            scored.append({**row, "normalized_score": score})
    return scored


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _summarize_runs(run_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run_benchmark: dict[tuple[str, str], list[float]] = defaultdict(list)
    raw_by_run_benchmark: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in scored_rows:
        key = (str(row["run_id"]), str(row["benchmark"]))
        by_run_benchmark[key].append(float(row["normalized_score"]))
        raw_by_run_benchmark[key].append(float(row["value"]))

    out: list[dict[str, Any]] = []
    for run in run_rows:
        run_id = str(run["run_id"])
        bench_scores = {bench: _mean(by_run_benchmark.get((run_id, bench), [])) for bench in PRIMARY_METRICS}
        aggregate = _mean([value for value in bench_scores.values() if math.isfinite(value)])
        out.append(
            {
                **run,
                "aggregate_score": aggregate,
                "belebele_score": bench_scores["Belebele"],
                "flores_score": bench_scores["FLORES"],
                "sib200_score": bench_scores["SIB200"],
                "taxi1500_score": bench_scores["Taxi1500"],
                "num_scored_items": sum(len(by_run_benchmark.get((run_id, bench), [])) for bench in PRIMARY_METRICS),
            }
        )
    return sorted(out, key=lambda row: (-(row["aggregate_score"] if math.isfinite(row["aggregate_score"]) else -1), str(row["run_id"])))


def _summarize_group(run_scores: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in run_scores:
        if any(str(row.get(field, "")).strip() == "" for field in group_fields):
            continue
        aggregate = row.get("aggregate_score")
        if not isinstance(aggregate, (int, float)) or not math.isfinite(float(aggregate)):
            continue
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)

    out: list[dict[str, Any]] = []
    for key, rows in groups.items():
        aggregate_values = [float(row["aggregate_score"]) for row in rows if math.isfinite(float(row["aggregate_score"]))]
        if not aggregate_values:
            continue
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
                "n_runs": len(rows),
                "mean_aggregate_score": _mean(aggregate_values),
                "mean_belebele_score": _mean([float(row["belebele_score"]) for row in rows if math.isfinite(float(row["belebele_score"]))]),
                "mean_flores_score": _mean([float(row["flores_score"]) for row in rows if math.isfinite(float(row["flores_score"]))]),
                "mean_sib200_score": _mean([float(row["sib200_score"]) for row in rows if math.isfinite(float(row["sib200_score"]))]),
                "mean_taxi1500_score": _mean([float(row["taxi1500_score"]) for row in rows if math.isfinite(float(row["taxi1500_score"]))]),
                "run_ids": ";".join(str(row["run_id"]) for row in sorted(rows, key=lambda row: str(row["run_id"]))),
            }
        )
        out.append(item)
    return sorted(out, key=lambda row: (-float(row["mean_aggregate_score"]), tuple(str(row.get(field, "")) for field in group_fields)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _friedman_rank_payload(scored_rows: list[dict[str, Any]], ranks: list[int], kinds: set[str]) -> dict[str, Any]:
    algorithm_labels = [f"rank_{rank}" for rank in ranks]
    by_unit: dict[tuple[str, str, str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in scored_rows:
        kind = str(row.get("kind", ""))
        rank = row.get("total_rank")
        if kind not in kinds or rank not in ranks:
            continue
        key = (
            kind,
            str(row["benchmark"]),
            str(row["task"]),
            str(row["metric"]),
            str(row["better"]),
        )
        by_unit[key][int(rank)] = row

    table_rows: list[dict[str, Any]] = []
    for (_kind, benchmark, task, metric, better), rank_rows in sorted(by_unit.items()):
        if any(rank not in rank_rows for rank in ranks):
            continue
        table_rows.append(
            {
                "benchmark": benchmark,
                "metric_family": metric,
                "task": task,
                "better": better,
                "approach_values": {
                    f"rank_{rank}": float(rank_rows[rank]["value"])
                    for rank in ranks
                },
            }
        )
    return {"algorithm_labels": algorithm_labels, "table_rows": table_rows}


def _friedman_rows(significance: dict[str, Any], algorithm_labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in significance.get("friedman_results", []):
        if not isinstance(item, dict):
            continue
        row = {
            "group_label": item.get("group_label", ""),
            "n_units": item.get("n_units", ""),
            "n_algorithms": item.get("n_algorithms", ""),
            "friedman_statistic": item.get("friedman_statistic", ""),
            "p_raw": item.get("p_raw", ""),
            "significant": item.get("significant", ""),
            "kendalls_w": item.get("kendalls_w", ""),
            "best_rank": str(item.get("best_rank_approach", "")).replace("rank_", ""),
            "best_avg_rank": item.get("best_avg_rank", ""),
            "note": item.get("note", ""),
        }
        avg_ranks = item.get("avg_ranks", {})
        if isinstance(avg_ranks, dict):
            for label in algorithm_labels:
                row[f"avg_{label}"] = avg_ranks.get(label, "")
        rows.append(row)
    return rows


def _pairwise_rows(significance: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in significance.get("pairwise_results", []):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["label_a"] = str(row.get("label_a", "")).replace("rank_", "")
        row["label_b"] = str(row.get("label_b", "")).replace("rank_", "")
        rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else ""
    return str(value)


def _write_markdown(
    path: Path,
    run_scores: list[dict[str, Any]],
    rank_summary: list[dict[str, Any]],
    kind_rank_summary: list[dict[str, Any]],
    moe_rank_summary: list[dict[str, Any]],
    friedman_rows: list[dict[str, Any]],
) -> None:
    best_run = run_scores[0] if run_scores else {}
    best_rank = rank_summary[0] if rank_summary else {}
    best_moe_rank = moe_rank_summary[0] if moe_rank_summary else {}
    lines = [
        "# Rank Sweep Selection",
        "",
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "Scoring uses primary metrics only: Belebele `acc_norm`, FLORES `bits_per_byte` (lower is better), SIB200 `f1`, and Taxi1500 `f1`.",
        "Each task metric is min-max normalized across runs; run scores macro-average benchmark families so FLORES does not dominate by task count.",
        "",
        "## Recommendation",
        "",
        f"- Best single run: `{best_run.get('run_id', '')}` with score {_fmt(best_run.get('aggregate_score', ''))}.",
        f"- Best rank averaged across all rank-bearing runs: `{best_rank.get('total_rank', '')}` with mean score {_fmt(best_rank.get('mean_aggregate_score', ''))}.",
        f"- Best shared MoE rank across `moelora`, `mola`, and `hmora`: `{best_moe_rank.get('total_rank', '')}` with mean score {_fmt(best_moe_rank.get('mean_aggregate_score', ''))}.",
        "",
        "## Top Runs",
        "",
        "| Run ID | Kind | Rank | Score | Belebele | FLORES | SIB200 | Taxi1500 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_scores[:15]:
        lines.append(
            f"| `{row['run_id']}` | `{row['kind']}` | {row.get('total_rank', '')} | {_fmt(row['aggregate_score'])} | {_fmt(row['belebele_score'])} | {_fmt(row['flores_score'])} | {_fmt(row['sib200_score'])} | {_fmt(row['taxi1500_score'])} |"
        )

    lines.extend(["", "## Shared MoE Rank Summary", "", "| Total rank | Runs | Mean score | Belebele | FLORES | SIB200 | Taxi1500 |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in moe_rank_summary:
        lines.append(
            f"| {row['total_rank']} | {row['n_runs']} | {_fmt(row['mean_aggregate_score'])} | {_fmt(row['mean_belebele_score'])} | {_fmt(row['mean_flores_score'])} | {_fmt(row['mean_sib200_score'])} | {_fmt(row['mean_taxi1500_score'])} |"
        )

    lines.extend(
        [
            "",
            "## Friedman Rank Test By Benchmark",
            "",
            "Ranks are compared as algorithms using paired `(method, task)` rows for `moelora`, `mola`, and `hmora` at shared total ranks 16, 32, 64, 128, 256, and 512.",
            "Lower average Friedman rank is better.",
            "",
            "| Benchmark | Units | p | Kendall W | Winning rank | Avg rank | Significant |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in friedman_rows:
        lines.append(
            f"| {row['group_label']} | {row['n_units']} | {_fmt(row['p_raw'])} | {_fmt(row['kendalls_w'])} | {row['best_rank']} | {_fmt(row['best_avg_rank'])} | {'yes' if row['significant'] else 'no'} |"
        )

    lines.extend(["", "## Best Rank By Family", "", "| Kind | Total rank | Score | Runs |", "|---|---:|---:|---:|"])
    best_by_kind: dict[str, dict[str, Any]] = {}
    for row in kind_rank_summary:
        best_by_kind.setdefault(str(row["kind"]), row)
    for kind in sorted(best_by_kind):
        row = best_by_kind[kind]
        lines.append(f"| `{kind}` | {row['total_rank']} | {_fmt(row['mean_aggregate_score'])} | {row['n_runs']} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the best rank from completed Noctua2 rank-sweep benchmark results.")
    parser.add_argument("--config", default="configs/pipeline")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    profile = _load_profile(REPO_ROOT / args.config, args.profile)
    roots = [REPO_ROOT / value for value in (args.root or DEFAULT_ROOTS)]
    output_dir = REPO_ROOT / args.output_dir

    run_rows, raw_rows = _collect_raw_rows(profile, roots)
    if not raw_rows:
        raise SystemExit("No benchmark result metrics found.")
    scored_rows = _normalized_scores(raw_rows)
    run_scores = _summarize_runs(run_rows, scored_rows)

    rankable = [row for row in run_scores if row.get("total_rank") not in ("", None) and str(row.get("kind")) != "movlora"]
    rank_summary = _summarize_group(rankable, ["total_rank"])
    kind_rank_summary = _summarize_group(rankable, ["kind", "total_rank"])
    moe_rows = [row for row in rankable if row.get("kind") in {"moelora", "mola", "hmora"}]
    common_moe_ranks = {
        rank
        for rank in {row.get("total_rank") for row in moe_rows}
        if {row.get("kind") for row in moe_rows if row.get("total_rank") == rank} == {"moelora", "mola", "hmora"}
    }
    moe_rank_summary = _summarize_group([row for row in moe_rows if row.get("total_rank") in common_moe_ranks], ["total_rank"])
    shared_moe_ranks = sorted(int(rank) for rank in common_moe_ranks)
    friedman_payload = _friedman_rank_payload(scored_rows, shared_moe_ranks, {"moelora", "mola", "hmora"})
    friedman_significance = build_significance_results(friedman_payload)
    friedman_summary_rows = _friedman_rows(friedman_significance, friedman_payload["algorithm_labels"])
    friedman_pairwise_rows = _pairwise_rows(friedman_significance)

    _write_csv(output_dir / "rank_selection_raw_metrics.csv", scored_rows)
    _write_csv(output_dir / "rank_selection_run_scores.csv", run_scores)
    _write_csv(output_dir / "rank_selection_rank_summary.csv", rank_summary)
    _write_csv(output_dir / "rank_selection_kind_rank_summary.csv", kind_rank_summary)
    _write_csv(output_dir / "rank_selection_moe_rank_summary.csv", moe_rank_summary)
    _write_json(output_dir / "rank_selection_friedman_payload.json", friedman_payload)
    _write_json(output_dir / "rank_selection_friedman_results.json", friedman_significance)
    _write_csv(output_dir / "rank_selection_friedman_by_benchmark.csv", friedman_summary_rows)
    _write_csv(output_dir / "rank_selection_friedman_pairwise.csv", friedman_pairwise_rows)
    _write_markdown(output_dir / "rank_selection_report.md", run_scores, rank_summary, kind_rank_summary, moe_rank_summary, friedman_summary_rows)

    print(f"[rank_sweep_select_best_rank] wrote {output_dir}")
    if moe_rank_summary:
        print(
            "[rank_sweep_select_best_rank] best_shared_moe_rank="
            f"{moe_rank_summary[0]['total_rank']} score={float(moe_rank_summary[0]['mean_aggregate_score']):.4f}"
        )
    if run_scores:
        print(
            "[rank_sweep_select_best_rank] best_run="
            f"{run_scores[0]['run_id']} score={float(run_scores[0]['aggregate_score']):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
