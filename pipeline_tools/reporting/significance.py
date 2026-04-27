import argparse
import json
import sys
from itertools import combinations
from statistics import median
from typing import Any

from scipy.stats import friedmanchisquare, rankdata, wilcoxon

"""
All logic regarding the significance tests is outsourced into here so that i can easier review, check, debug and adapt it
I did this since most of the report implementation can be handled by agents without review necessary since its mostly just excel formatting and presentation
"""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _shared_direction(rows: list[dict[str, Any]]) -> str | None:
    directions = {str(row.get("better", "")).strip() for row in rows if str(row.get("better", "")).strip()}
    if len(directions) == 1:
        return next(iter(directions))
    return None


def _aligned_value(value: float, better: str) -> float:
    return -value if better == "lower" else value


def _bh_adjust(p_values: list[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(p_values)
    if not p_values:
        return adjusted
    indexed = sorted(
        [(idx, 1.0 if p is None else float(p)) for idx, p in enumerate(p_values)],
        key=lambda item: item[1],
    )
    total = len(indexed)
    adjusted_sorted: list[float] = []
    for rank, (_, p_value) in enumerate(indexed, start=1):
        adjusted_sorted.append(min(1.0, float(p_value) * total / rank))
    for idx in range(total - 2, -1, -1):
        adjusted_sorted[idx] = min(adjusted_sorted[idx], adjusted_sorted[idx + 1])
    for (original_idx, _), adj in zip(indexed, adjusted_sorted):
        adjusted[original_idx] = adj
    return adjusted


def _metric_group(metric_family: str) -> str:
    family = str(metric_family)
    for prefix in ("byte_perplexity", "word_perplexity", "bits_per_byte"):
        if family.startswith(prefix):
            return prefix
    return family


def _dataset_rank(dataset: str) -> int:
    return {"Belebele": 0, "FLORES": 1, "SIB200": 2, "Taxi1500": 3}.get(dataset, 99)


def _stats_group_sort_key(group_key: tuple[str, str]) -> tuple[int, int, str]:
    benchmark, metric_group = group_key
    group_rank = {
        "acc_norm": 0,
        "acc": 0,
        "byte_perplexity": 0,
        "word_perplexity": 1,
        "bits_per_byte": 2,
    }.get(metric_group, 99)
    return _dataset_rank(benchmark), group_rank, metric_group


def _stats_group_label(group_key: tuple[str, str]) -> str:
    benchmark, metric_group = group_key
    if benchmark == "FLORES":
        return f"FLORES / {metric_group}"
    return benchmark


def _build_group_friedman_result(
    group_key: tuple[str, str], block_rows: list[dict[str, Any]], algorithm_labels: list[str]
) -> dict[str, Any]:
    complete_rows: list[dict[str, float]] = []
    rank_rows: list[list[float]] = []
    better = _shared_direction(block_rows) or "higher"
    note = ""

    for item in block_rows:
        row_values: dict[str, float] = {}
        valid = True
        for label in algorithm_labels:
            value = item["baseline"] if label == "baseline" else item["approach_values"].get(label)
            if not isinstance(value, (int, float)):
                valid = False
                break
            row_values[label] = float(value)
        if not valid:
            continue
        aligned_values = {label: _aligned_value(value, better) for label, value in row_values.items()}
        complete_rows.append(aligned_values)
        rank_rows.append(list(rankdata([-aligned_values[label] for label in algorithm_labels], method="average")))

    n_units = len(complete_rows)
    n_algorithms = len(algorithm_labels)
    avg_rank_map: dict[str, float] = {}
    statistic: float | None = None
    p_raw: float | None = None
    kendalls_w: float | None = None
    best_rank_approach = ""
    best_avg_rank: float | None = None

    if n_algorithms < 3:
        note = "need at least 3 algorithms"
    elif n_units < 2:
        note = "need at least 2 complete paired task rows"
    else:
        try:
            vectors = [[row[label] for row in complete_rows] for label in algorithm_labels]
            result = friedmanchisquare(*vectors) # returns friedmans chi-square value and pvalue
            statistic = float(result.statistic)
            p_raw = float(result.pvalue)    # raw p-value without correction
            kendalls_w = statistic / (n_units * (n_algorithms - 1))   # Kendall's W, measures agreement
            avg_rank_map = {
                label: _mean([rank_row[idx] for rank_row in rank_rows])
                for idx, label in enumerate(algorithm_labels)
            }
            best_rank_approach = min(avg_rank_map, key=avg_rank_map.get)    # approach with lowest average rank is best
            best_avg_rank = avg_rank_map[best_rank_approach]
        except ValueError as exc:
            note = str(exc)

    return {
        "group_label": _stats_group_label(group_key),
        "unit_definition": "complete paired task rows",
        "n_units": n_units,
        "n_algorithms": n_algorithms,
        "friedman_statistic": statistic,
        "p_raw": p_raw,
        "significant": bool(p_raw is not None and p_raw < 0.05),
        "kendalls_w": kendalls_w,
        "best_rank_approach": best_rank_approach,
        "best_avg_rank": best_avg_rank,
        "avg_ranks": avg_rank_map,
        "note": note,
    }


def _build_group_two_algorithm_result(
    group_key: tuple[str, str], block_rows: list[dict[str, Any]], algorithm_labels: list[str]
) -> dict[str, Any]:
    note = ""
    complete_rows = 0
    for item in block_rows:
        row_values = [
            item["baseline"] if label == "baseline" else item["approach_values"].get(label)
            for label in algorithm_labels
        ]
        if all(isinstance(value, (int, float)) for value in row_values):
            complete_rows += 1
    if len(algorithm_labels) != 2:
        note = "two-algorithm omnibus path requires exactly 2 algorithms"
    elif complete_rows < 2:
        note = "need at least 2 complete paired task rows"
    else:
        note = "Friedman not applicable for 2 algorithms; pairwise Wilcoxon used directly"
    return {
        "group_label": _stats_group_label(group_key),
        "unit_definition": "complete paired task rows",
        "n_units": complete_rows,
        "n_algorithms": len(algorithm_labels),
        "friedman_statistic": None,
        "p_raw": None,
        "significant": False,
        "kendalls_w": None,
        "best_rank_approach": "",
        "best_avg_rank": None,
        "avg_ranks": {},
        "note": note,
    }


def _build_group_pairwise_result(
    group_label: str, block_rows: list[dict[str, Any]], label_a: str, label_b: str
) -> dict[str, Any] | None:
    better = _shared_direction(block_rows) or "higher"
    scores_a: list[float] = []
    scores_b: list[float] = []
    wins_a = 0
    wins_b = 0
    ties = 0
    deltas: list[float] = []

    for item in block_rows:
        value_a = item["baseline"] if label_a == "baseline" else item["approach_values"].get(label_a)
        value_b = item["baseline"] if label_b == "baseline" else item["approach_values"].get(label_b)
        if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
            continue
        score_a = _aligned_value(float(value_a), better)
        score_b = _aligned_value(float(value_b), better)
        diff = score_a - score_b
        scores_a.append(score_a)
        scores_b.append(score_b)
        deltas.append(diff)
        if diff > 0:
            wins_a += 1
        elif diff < 0:
            wins_b += 1
        else:
            ties += 1

    if not scores_a:
        return None

    statistic: float | None = None
    p_raw: float | None = None
    note = ""
    if all(delta == 0 for delta in deltas):
        statistic = 0.0
        p_raw = 1.0
        note = "all paired differences are zero"
    else:
        try:
            result = wilcoxon(scores_a, scores_b, zero_method="wilcox", alternative="two-sided", method="auto") # perform Wilconxon singed rank, two sided
            statistic = float(result.statistic)
            p_raw = float(result.pvalue)
        except ValueError as exc:
            note = str(exc)

    return {
        "group_label": group_label,
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": len(scores_a),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "median_delta": median(deltas) if deltas else None,
        "wilcoxon_statistic": statistic,
        "p_raw": p_raw,
        "p_fdr": None,
        "significant_fdr": False,
        "note": note,
    }


def build_significance_results(payload: dict[str, Any]) -> dict[str, Any]:
    table_rows = payload.get("table_rows", [])
    algorithm_labels = payload.get("algorithm_labels", [])
    if not isinstance(table_rows, list) or not isinstance(algorithm_labels, list):
        raise ValueError("payload must include list fields 'table_rows' and 'algorithm_labels'")

    stats_group_map: dict[tuple[str, str], list[dict[str, Any]]] = {}   # these are the benchmark groups (belebele, sib, taxi flores byte/bit/word)
    for row in table_rows:
        if not isinstance(row, dict):
            continue
        group_key = (str(row.get("benchmark", "")), _metric_group(str(row.get("metric_family", ""))))
        stats_group_map.setdefault(group_key, []).append(row)

    friedman_results: list[dict[str, Any]] = []
    pairwise_results: list[dict[str, Any]] = []
    use_two_algorithm_path = len(algorithm_labels) == 2
    for group_key in sorted(stats_group_map.keys(), key=_stats_group_sort_key):
        group_rows = stats_group_map[group_key]
        if use_two_algorithm_path:
            friedman_result = _build_group_two_algorithm_result(group_key, group_rows, algorithm_labels)
        else:
            friedman_result = _build_group_friedman_result(group_key, group_rows, algorithm_labels)
        friedman_results.append(friedman_result)
        if not (use_two_algorithm_path or friedman_result["significant"]):
            continue
        group_pairwise_results: list[dict[str, Any]] = []
        for label_a, label_b in combinations(algorithm_labels, 2):
            result = _build_group_pairwise_result(friedman_result["group_label"], group_rows, label_a, label_b) # get pariwise wilcoxon result for this pair of algorithms within the group
            if result is not None:
                group_pairwise_results.append(result)
        adjusted = _bh_adjust([result["p_raw"] for result in group_pairwise_results])
        for result, p_fdr in zip(group_pairwise_results, adjusted):
            result["p_fdr"] = p_fdr
            result["significant_fdr"] = bool(p_fdr is not None and p_fdr < 0.05)
        pairwise_results.extend(group_pairwise_results)

    notes = [
        {"setting": "alpha", "value": "0.05"},
        {"setting": "omnibus_test", "value": "Friedman test, run separately for each benchmark group"},
        {
            "setting": "post_hoc_test",
            "value": "Wilcoxon signed-rank, two-sided; for 3+ algorithms only after significant Friedman, and directly for 2-algorithm reports",
        },
        {
            "setting": "multiple_comparison_correction",
            "value": "Benjamini-Hochberg FDR applied within each benchmark group",
        },
        {"setting": "paired_units", "value": "Complete paired task rows within each benchmark group"},
        {
            "setting": "missing_data_rule",
            "value": "Friedman uses complete-case rows across all algorithms; Wilcoxon uses pairwise-complete rows",
        },
        {
            "setting": "flores_grouping",
            "value": "FLORES is split into byte_perplexity, word_perplexity, and bits_per_byte; each group combines both directions",
        },
        {"setting": "rank_interpretation", "value": "Lower average rank is better"},
    ]

    return {
        "notes": notes,
        "friedman_results": friedman_results,
        "pairwise_results": pairwise_results,
    }


def _load_payload(input_json: str) -> dict[str, Any]:
    if input_json:
        with open(input_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("input payload must be a JSON object")
    return payload


def _write_payload(output_json: str, payload: dict[str, Any]) -> None:
    if output_json:
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        return
    json.dump(payload, sys.stdout, ensure_ascii=True, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute report significance results from normalized comparison rows.")
    parser.add_argument("--input-json", default="", help="JSON payload path. Reads stdin when omitted.")
    parser.add_argument("--output-json", default="", help="Output JSON path. Writes stdout when omitted.")
    args = parser.parse_args()

    payload = _load_payload(args.input_json)
    results = build_significance_results(payload)
    _write_payload(args.output_json, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
