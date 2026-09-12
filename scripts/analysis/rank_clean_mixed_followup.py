#!/usr/bin/env python3
"""Rank the audited clean mixed-follow-up configurations with benchmark macro ranks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / (
    "outputs/pipeline_runs/moe_study_mixed_rank_expert_followup_clean22_consolidated_v13"
)
WORKBOOK = RESULT_ROOT / "moe_study_mixed_rank_expert_followup_clean22_consolidated_v13.xlsx"
PROVENANCE = RESULT_ROOT / "provenance.json"
OUTPUT_ROOT = RESULT_ROOT / "analysis/macro_ranking"
BENCHMARKS = ["Belebele", "FLORES", "SIB200", "Taxi1500"]
UNIT_COLS = ["score_family", "task", "language", "metric"]
EXPECTED_FAMILIES = {
    "Belebele": 9,
    "FLORES en->x": 9,
    "FLORES x->en": 9,
    "SIB200": 9,
    "Taxi1500": 6,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _limited_by_family(ranking: pd.DataFrame, count: int, maximum: int) -> list[str]:
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    for row in ranking.itertuples(index=False):
        used = family_counts.get(row.approach, 0)
        if used >= maximum:
            continue
        selected.append(row.run_id)
        family_counts[row.approach] = used + 1
        if len(selected) == count:
            break
    return selected


def main() -> int:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    run_ids = [str(record["run_id"]) for record in provenance["runs"]]
    if len(run_ids) != 22 or len(set(run_ids)) != 22:
        raise SystemExit("provenance must contain exactly 22 unique runs")

    raw = pd.read_excel(WORKBOOK, sheet_name="benchmark_raw")
    metrics = raw[raw["run_id"].astype(str).isin(run_ids)].copy()
    primary = (
        (metrics.benchmark.eq("Belebele") & metrics.metric.eq("acc_norm"))
        | (metrics.benchmark.eq("FLORES") & metrics.metric.astype(str).str.startswith("bits_per_byte "))
        | (metrics.benchmark.eq("SIB200") & metrics.metric.eq("acc"))
        | (metrics.benchmark.eq("Taxi1500") & metrics.metric.eq("acc"))
    )
    metrics = metrics[primary].copy()
    metrics["value"] = pd.to_numeric(metrics["value"], errors="coerce")
    if metrics["value"].isna().any():
        raise SystemExit("primary metrics contain missing or non-numeric values")

    direction = metrics.metric.astype(str).str.extract(r"(en->x|x->en)", expand=False)
    metrics["score_family"] = metrics["benchmark"]
    flores = metrics.benchmark.eq("FLORES")
    metrics.loc[flores, "score_family"] = "FLORES " + direction.loc[flores]
    metrics["score"] = np.where(metrics.better.eq("lower"), -metrics.value, metrics.value)

    rows_per_run = metrics.groupby("run_id").size()
    if set(rows_per_run.index) != set(run_ids) or not rows_per_run.eq(42).all():
        raise SystemExit(f"each run must contain exactly 42 primary units: {rows_per_run.to_dict()}")
    family_counts = metrics.drop_duplicates(UNIT_COLS).groupby("score_family").size().to_dict()
    if family_counts != EXPECTED_FAMILIES:
        raise SystemExit(f"unexpected evaluation-unit panel: {family_counts}")
    duplicates = metrics.groupby(UNIT_COLS + ["run_id"]).size()
    if not duplicates.eq(1).all():
        raise SystemExit("a run has duplicate rows for an evaluation unit")

    pivot = metrics.pivot(index=UNIT_COLS, columns="run_id", values="score").reindex(columns=run_ids)
    if pivot.shape != (42, 22) or pivot.isna().any().any():
        raise SystemExit(f"paired comparison panel must be 42x22, got {pivot.shape}")

    # Friedman-style ranks: rank configurations within each paired evaluation unit.
    unit_ranks = pivot.rank(axis=1, ascending=False, method="average")
    family_ranks = unit_ranks.groupby(level="score_family").mean()

    # FLORES receives one benchmark vote: average its two direction-level rank aggregates.
    family_ranks.loc["FLORES"] = family_ranks.loc[["FLORES en->x", "FLORES x->en"]].mean()
    benchmark_ranks = family_ranks.loc[BENCHMARKS]

    ranking = benchmark_ranks.T.rename_axis("run_id").reset_index()
    kind = metrics.drop_duplicates("run_id").set_index("run_id")["kind"].astype(str)
    ranking.insert(1, "approach", ranking.run_id.map(kind))
    ranking = ranking.rename(columns={benchmark: f"{benchmark}_rank" for benchmark in BENCHMARKS})
    rank_columns = [f"{benchmark}_rank" for benchmark in BENCHMARKS]
    ranking["macro_avg_rank"] = ranking[rank_columns].mean(axis=1)
    ranking = ranking.sort_values(["macro_avg_rank", "run_id"], kind="stable").reset_index(drop=True)
    ranking.insert(0, "macro_order", np.arange(1, len(ranking) + 1))

    family_best = ranking.drop_duplicates("approach", keep="first").reset_index(drop=True)
    selection_options = {
        "unrestricted_top3_configurations": ranking.head(3).run_id.tolist(),
        "top3_unique_approach_families": family_best.head(3).run_id.tolist(),
        "top5_with_at_most_2_per_family": _limited_by_family(ranking, count=5, maximum=2),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT_ROOT / "configuration_macro_ranks.csv", index=False)
    family_best.to_csv(OUTPUT_ROOT / "best_configuration_per_approach.csv", index=False)
    unit_ranks.reset_index().to_csv(OUTPUT_ROOT / "unit_ranks.csv", index=False)
    family_ranks.to_csv(OUTPUT_ROOT / "score_family_ranks.csv")
    benchmark_ranks.to_csv(OUTPUT_ROOT / "benchmark_ranks.csv")
    (OUTPUT_ROOT / "selection_options.json").write_text(
        json.dumps(selection_options, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    methodology = {
        "schema_version": 1,
        "workbook": str(WORKBOOK),
        "workbook_sha256": _sha256(WORKBOOK),
        "provenance": str(PROVENANCE),
        "provenance_sha256": _sha256(PROVENANCE),
        "run_count": 22,
        "primary_units_per_run": 42,
        "score_families": EXPECTED_FAMILIES,
        "ranking": "average-tie rank within each paired task/language/metric unit; lower rank is better",
        "flores": "average en->x and x->en direction-level mean ranks after unit ranking",
        "macro": "equal arithmetic mean of Belebele, FLORES, SIB200, and Taxi1500 benchmark ranks",
        "selection_options": selection_options,
    }
    (OUTPUT_ROOT / "methodology.json").write_text(
        json.dumps(methodology, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(ranking.to_string(index=False))
    print(f"\nWrote ranking evidence to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
