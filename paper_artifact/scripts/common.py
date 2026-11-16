"""Shared, auditable helpers for the RQ1/RQ2 paper tables.

The table pipeline intentionally mirrors notebooks/best_rank_expert_selection.ipynb:
primary metrics only, failed CoLA runs excluded, lower-is-better metrics reversed,
and ranks computed within paired evaluation units.  No cross-workbook score
normalization is used.
"""
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
UNIT_COLS = ["score_family", "task", "language", "metric"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def results_path(cfg: dict, key: str) -> Path:
    path = REPO_ROOT / cfg["paths"]["results_root"] / cfg["paths"][key]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def repo_path(cfg: dict, key: str) -> Path:
    path = REPO_ROOT / cfg["paths"][key]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def output_dir(cfg: dict, kind: str) -> Path:
    path = REPO_ROOT / cfg["output"][f"{kind}_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _primary_metric_mask(raw: pd.DataFrame, specs: list[dict]) -> pd.Series:
    mask = pd.Series(False, index=raw.index)
    metric = raw["metric"].astype(str)
    for spec in specs:
        benchmark = raw["benchmark"].eq(spec["benchmark"])
        metric_match = (
            metric.eq(spec["metric_prefix"])
            if spec.get("exact", True)
            else metric.str.startswith(spec["metric_prefix"])
        )
        mask |= benchmark & metric_match
    return mask


def load_primary(path: Path, cfg: dict) -> pd.DataFrame:
    """Load the notebook's primary, non-baseline, non-failed observations."""
    raw = pd.read_excel(path, sheet_name="benchmark_raw")
    data = raw[_primary_metric_mask(raw, cfg["analysis"]["primary_metrics"])].copy()
    baseline = (
        data["is_baseline"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    data = data[~baseline].copy()
    data = data[~data["kind"].astype(str).isin(cfg["analysis"]["exclude_kinds"])].copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"]).copy()

    direction = data["metric"].astype(str).str.extract(r"(en->x|x->en)", expand=False).fillna("")
    data["score_family"] = data["benchmark"]
    flores = data["benchmark"].eq("FLORES")
    data.loc[flores, "score_family"] = "FLORES " + direction.loc[flores]
    data["score"] = data["value"].where(data["better"].ne("lower"), -data["value"])
    return data


def load_final_panel(path: Path, cfg: dict) -> pd.DataFrame:
    """Load a broader-regime workbook, keeping the unadapted base model.

    The base model is a comparison system in these regimes rather than an
    excluded training run, so the is_baseline filter used for the sweeps is not
    applied here. Resource labels are replaced with the canonical FineWeb2
    assignments. The workbooks carry an older lookup that both left aliases and
    alternate scripts unknown and drew its tier boundaries differently, so the
    replacement fills those gaps and also moves some already-labelled languages
    between tiers.
    """
    raw = pd.read_excel(path, sheet_name="benchmark_raw")
    data = raw[_primary_metric_mask(raw, cfg["analysis"]["primary_metrics"])].copy()
    data = data[~data["kind"].astype(str).isin(cfg["analysis"]["exclude_kinds"])].copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"]).copy()

    direction = data["metric"].astype(str).str.extract(r"(en->x|x->en)", expand=False).fillna("")
    data["score_family"] = data["benchmark"]
    flores = data["benchmark"].eq("FLORES")
    data.loc[flores, "score_family"] = "FLORES " + direction.loc[flores]
    data["score"] = data["value"].where(data["better"].ne("lower"), -data["value"])
    data["resource_availability"] = data["language"].map(_resource_label_map(cfg))
    missing = sorted(data.loc[data["resource_availability"].isna(), "language"].unique())
    if missing:
        raise ValueError(f"No canonical resource label for evaluation languages: {missing}")
    return data


def _resource_label_map(cfg: dict) -> dict[str, str]:
    """Canonical resource tier keyed by the benchmark/FineWeb2 subset code."""
    labels = pd.read_csv(
        repo_path(cfg, "resource_labels_csv"),
        usecols=["subset", "resource_category"],
    ).dropna()
    labels["resource_category"] = labels["resource_category"].astype(str).str.upper()

    conflicts = labels.groupby("subset")["resource_category"].nunique()
    conflicts = sorted(conflicts[conflicts.gt(1)].index)
    if conflicts:
        raise ValueError(f"Conflicting canonical resource labels for: {conflicts}")

    mapping = labels.drop_duplicates("subset").set_index("subset")["resource_category"].to_dict()
    mapping.update(
        {
            language: str(tier).upper()
            for language, tier in cfg["analysis"].get("resource_label_overrides", {}).items()
        }
    )
    allowed = set(cfg["rq3"]["resource_groups"])
    invalid = {language: tier for language, tier in mapping.items() if tier not in allowed}
    if invalid:
        raise ValueError(f"Invalid canonical resource labels: {invalid}")
    return mapping


def final_ranking(data: pd.DataFrame, benchmarks: list[str]) -> pd.DataFrame:
    """Average rank per system within each benchmark, plus the macro average."""
    pivot = data.pivot(index=UNIT_COLS, columns="run_id", values="score")
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete final panel: {pivot.isna().sum().to_dict()}")
    unit_ranks = pivot.rank(axis=1, ascending=False, method="average")
    family_ranks = unit_ranks.groupby(level="score_family").mean()
    benchmark_ranks = combine_flores(family_ranks, benchmarks)
    ranking = benchmark_ranks.T
    ranking["macro_avg_rank"] = benchmark_ranks.mean()
    return ranking.sort_values("macro_avg_rank")


def final_pairwise_tests(data: pd.DataFrame, benchmarks: list[str]) -> pd.DataFrame:
    """All pairwise Wilcoxon tests, Holm-corrected inside each benchmark.

    Correcting over every pair rather than a selected subset keeps the family
    size independent of which systems happen to rank highest.
    """
    rows: list[dict] = []
    for benchmark in benchmarks:
        pivot = data[data["benchmark"].eq(benchmark)].pivot(
            index=["score_family", "task", "language", "metric"],
            columns="run_id",
            values="score",
        )
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete final panel for {benchmark}")
        statistic, p_omnibus = friedmanchisquare(*[pivot[column] for column in pivot.columns])
        for config_a, config_b in combinations(sorted(pivot.columns), 2):
            test = wilcoxon(pivot[config_a], pivot[config_b], alternative="two-sided")
            rows.append(
                {
                    "benchmark": benchmark,
                    "n_units": len(pivot),
                    "friedman_q": statistic,
                    "kendalls_w": statistic / (len(pivot) * (len(pivot.columns) - 1)),
                    "p_friedman": p_omnibus,
                    "config_a": config_a,
                    "config_b": config_b,
                    "median_difference": (pivot[config_a] - pivot[config_b]).median(),
                    "rank_biserial": rank_biserial(pivot[config_a], pivot[config_b]),
                    "p_raw": test.pvalue,
                }
            )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = frame.groupby("benchmark")["p_raw"].transform(lambda p: holm(p))
    return frame


def verdict_vs_baseline(tests: pd.DataFrame, baseline: str, alpha: float) -> pd.DataFrame:
    """Reduce the pairwise tests to one better/tie/worse verdict per system."""
    rows = []
    subset = tests[tests["config_a"].eq(baseline) | tests["config_b"].eq(baseline)]
    for row in subset.itertuples(index=False):
        method = row.config_b if row.config_a == baseline else row.config_a
        sign = -1.0 if row.config_a == baseline else 1.0
        difference = sign * row.median_difference
        effect = sign * row.rank_biserial
        if row.p_holm >= alpha or row.p_friedman >= alpha:
            verdict = "tie"
        elif difference > 0 or (difference == 0 and effect > 0):
            verdict = "better"
        else:
            verdict = "worse"
        rows.append(
            {
                "benchmark": row.benchmark,
                "method": method,
                "median_difference": difference,
                "rank_biserial": effect,
                "p_holm": row.p_holm,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def baseline_magnitudes(
    data: pd.DataFrame,
    baseline: str,
    benchmarks: list[str],
    chance_levels: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Interpretable magnitudes for each system's comparison against the base model.

    The median difference alone is not readable across benchmarks, because
    bits-per-byte and accuracy points live on unrelated scales. These columns put
    each difference back on a scale the reader can judge:

    - ``base_median`` is the base model's own median on the benchmark, in the
      raw metric, so a difference can be sized against the level it moves.
    - ``median_relative_gain_pct`` is the median of the per-case relative
      changes, which is the paired statistic rather than a ratio of two medians.
      Only meaningful for a ratio-scale metric such as bits-per-byte.
    - ``win_rate_pct`` is the share of evaluation cases the system wins, which is
      what the rank-biserial effect weights by rank.
    - ``median_gain_vs_headroom_pct`` sizes the difference against the base
      model's margin above chance, for benchmarks where chance is well defined.

    Signs follow the benchmark's own direction: positive always favours the
    system, on every column.
    """
    chance_levels = chance_levels or {}
    rows: list[dict] = []
    for benchmark in benchmarks:
        subset = data[data["benchmark"].eq(benchmark)]
        values = subset.pivot(index=UNIT_COLS, columns="run_id", values="value")
        scores = subset.pivot(index=UNIT_COLS, columns="run_id", values="score")
        if values.isna().any().any():
            raise ValueError(f"Incomplete final panel for {benchmark}")
        base_median = float(values[baseline].median())
        chance = chance_levels.get(benchmark)
        headroom = None if chance is None else base_median - chance
        for method in values.columns.drop(baseline):
            # score is already sign-corrected, so the gain is the same expression
            # for lower-is-better and higher-is-better metrics.
            gain = scores[method] - scores[baseline]
            relative = 100 * gain / values[baseline].abs()
            median_difference = float(gain.median())
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "n_cases": int(len(gain)),
                    "base_median": base_median,
                    "median_difference": median_difference,
                    "median_relative_gain_pct": float(relative.median()),
                    "win_rate_pct": float(100 * (gain > 0).mean()),
                    "median_gain_vs_headroom_pct": (
                        float("nan")
                        if not headroom
                        else 100 * median_difference / headroom
                    ),
                }
            )
    return pd.DataFrame(rows)


def relative_gain_by_resource(
    data: pd.DataFrame,
    baseline: str,
    benchmark: str,
    groups: list[str],
) -> pd.DataFrame:
    """Median per-case relative gain against the base model, by resource tier.

    The gain-versus-base trend is measured on absolute scores, where a benchmark
    with a wide dynamic range gives its weakest languages more room to move. This
    is the same comparison on a relative scale, which is what tells the two apart.
    """
    subset = data[data["benchmark"].eq(benchmark) & data["resource_availability"].isin(groups)]
    values = subset.pivot(index=["resource_availability", *UNIT_COLS], columns="run_id", values="value")
    scores = subset.pivot(index=["resource_availability", *UNIT_COLS], columns="run_id", values="score")
    if values.isna().any().any():
        raise ValueError(f"Incomplete {benchmark} panel for the resource breakdown")
    relative = scores.drop(columns=baseline).sub(scores[baseline], axis=0)
    relative = 100 * relative.div(values[baseline].abs(), axis=0)
    return relative.groupby(level="resource_availability").median().reindex(groups)


def load_run_params(path: Path) -> pd.DataFrame:
    """Per-run trainable and total parameter counts from a workbook."""
    frame = pd.read_excel(path, sheet_name="run_configs")
    columns = ["run_id", "trainable_params", "all_params", "trainable_pct"]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} run_configs is missing {missing}")
    frame = frame[columns].copy()
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def parameter_summary(data: pd.DataFrame, params: pd.DataFrame, treatment: str) -> pd.DataFrame:
    """Trainable parameters per treatment level, in millions.

    Architectures are averaged within a level before being combined, so each
    architecture of the balanced panel carries the same weight it carries in the
    average positions.  The spread is reported because the rank-independent part
    of an adapter differs by an order of magnitude across architectures.
    """
    runs = data[["run_id", "kind", treatment]].drop_duplicates()
    merged = runs.merge(params[["run_id", "trainable_params"]], on="run_id", how="left")
    if merged["trainable_params"].isna().any():
        unknown = sorted(merged.loc[merged["trainable_params"].isna(), "run_id"])
        raise ValueError(f"No trainable_params recorded for: {unknown}")
    per_kind = merged.groupby([treatment, "kind"], as_index=False)["trainable_params"].mean()
    summary = per_kind.groupby(treatment)["trainable_params"].agg(["mean", "min", "max"]) / 1e6
    return summary.rename(columns=lambda name: f"trainable_millions_{name}")


def parse_configured_rank(run_id: object) -> float:
    run_id = str(run_id)
    total = re.search(r"_tr(\d+)_", run_id)
    components = re.search(r"_e(\d+)(?:_|$)", run_id)
    if total and components:
        return int(total.group(1)) / int(components.group(1))
    rank = re.search(r"_r(\d+)(?:_|$)", run_id)
    return float(rank.group(1)) if rank else np.nan


def parse_component_count(run_id: object) -> float:
    for pattern in [r"_e(\d+)(?:_|$)", r"_heads(\d+)$", r"_nup(\d+)$", r"_b(\d+)$"]:
        match = re.search(pattern, str(run_id))
        if match:
            return float(match.group(1))
    return np.nan


def parse_top_k(run_id: object) -> float:
    match = re.search(r"_k(\d+)(?:_|$)", str(run_id))
    return float(match.group(1)) if match else np.nan


def canonical_config_key(run_id: object, kind: object) -> tuple:
    """Identity of a configuration, independent of how its run id is spelled.

    The stages name the same configuration differently: the expert sweep writes
    the per-expert rank (``moelora_r2_e16_k8``) where the follow-up writes the
    total rank (``moelora_tr32_e16_k8``), and AdaMoE appears as ``adamoe_1_*`` in
    the expert sweep. Intersecting raw run ids therefore misses repeats, while
    (method, per-expert rank, expert count, top-k) identifies them.
    """

    def level(value: float) -> int | None:
        return None if pd.isna(value) else int(value)

    return (
        str(kind),
        level(parse_configured_rank(run_id)),
        level(parse_component_count(run_id)),
        level(parse_top_k(run_id)),
    )


def combine_flores(frame: pd.DataFrame | pd.Series, benchmarks: list[str]):
    result = frame.copy()
    result.loc["FLORES"] = result.loc[["FLORES en->x", "FLORES x->en"]].mean()
    return result.loc[benchmarks]


def average_rank_summary(
    data: pd.DataFrame,
    treatment: str,
    levels: list[int],
    benchmarks: list[str],
    include_architecture_in_block: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return benchmark average ranks, macro ranks, and per-benchmark tests."""
    index = UNIT_COLS + (["kind"] if include_architecture_in_block else [])
    if not include_architecture_in_block:
        data = data.groupby(UNIT_COLS + [treatment], as_index=False)["score"].mean()

    family_ranks: list[pd.Series] = []
    tests: list[dict] = []
    for benchmark in benchmarks:
        subset = data[data["benchmark"].eq(benchmark)]
        pivot = subset.pivot(index=index, columns=treatment, values="score").reindex(columns=levels)
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete {treatment} panel for {benchmark}: {pivot.isna().sum().to_dict()}")
        ranks = pivot.rank(axis=1, ascending=False, method="average").mean()
        ranks.name = benchmark
        family_ranks.append(ranks)
        statistic, p_raw = friedmanchisquare(*[pivot[level] for level in levels])
        tests.append(
            {
                "benchmark": benchmark,
                "n_blocks": len(pivot),
                "friedman_q": statistic,
                "kendalls_w": statistic / (len(pivot) * (len(levels) - 1)),
                "p_raw": p_raw,
            }
        )

    by_benchmark = pd.DataFrame(family_ranks)
    macro = by_benchmark.mean().sort_values()
    test_frame = pd.DataFrame(tests)
    test_frame["p_holm"] = holm(test_frame["p_raw"])
    return by_benchmark, macro, test_frame


def architecture_collapsed_normalized_summary(
    data: pd.DataFrame,
    treatment: str,
    levels: list[int],
    benchmarks: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Robustness view: collapse architectures, normalize within each unit.

    Each evaluation unit has equal weight after averaging architectures. Values
    are min-max normalized only across treatment levels within that same unit.
    Benchmark families are then macro-averaged with equal weight.
    """
    collapsed = data.groupby(UNIT_COLS + [treatment], as_index=False)["score"].mean()
    pivot = collapsed.pivot(index=UNIT_COLS, columns=treatment, values="score").reindex(columns=levels)
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete architecture-collapsed panel for {treatment}")
    lo, hi = pivot.min(axis=1), pivot.max(axis=1)
    normalized = pivot.sub(lo, axis=0).div((hi - lo).replace(0, np.nan), axis=0).fillna(1.0)
    by_family = normalized.groupby(level="score_family").mean()
    by_benchmark = combine_flores(by_family, benchmarks)
    macro = by_benchmark.mean().sort_values(ascending=False)
    return by_benchmark, macro


def followup_ranking(data: pd.DataFrame, benchmarks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce notebook cell 18's equal-benchmark follow-up ranking."""
    pivot = data.pivot(index=UNIT_COLS, columns="run_id", values="score")
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete follow-up panel: {pivot.isna().sum().to_dict()}")
    unit_ranks = pivot.rank(axis=1, ascending=False, method="average")
    family_ranks = unit_ranks.groupby(level="score_family").mean()
    benchmark_ranks = combine_flores(family_ranks, benchmarks)
    ranking = benchmark_ranks.T
    ranking["macro_avg_rank"] = benchmark_ranks.mean()
    approaches = data.drop_duplicates("run_id").set_index("run_id")["kind"]
    ranking.insert(0, "approach", approaches.reindex(ranking.index))
    ranking = ranking.sort_values("macro_avg_rank")
    ranking.insert(0, "overall_rank", range(1, len(ranking) + 1))
    return ranking, pivot


def holm(values) -> np.ndarray:
    values = np.asarray(pd.Series(values, dtype=float), dtype=float)
    order = np.argsort(values)
    ordered = values[order]
    adjusted_ordered = np.maximum.accumulate(
        [(len(values) - index) * value for index, value in enumerate(ordered)]
    )
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def rank_biserial(x: pd.Series, y: pd.Series) -> float:
    difference = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    difference = difference[difference != 0]
    if not len(difference):
        return 0.0
    ranks = rankdata(np.abs(difference), method="average")
    positive = ranks[difference > 0].sum()
    negative = ranks[difference < 0].sum()
    return float((positive - negative) / (positive + negative))


def followup_tests(
    data: pd.DataFrame, benchmarks: list[str], top_configs: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows: list[dict] = []
    pair_rows: list[dict] = []
    for benchmark in benchmarks:
        pivot = data[data["benchmark"].eq(benchmark)].pivot(
            index=["task", "language", "metric"], columns="run_id", values="score"
        )
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete follow-up panel for {benchmark}")
        statistic, p_raw = friedmanchisquare(*[pivot[column] for column in pivot.columns])
        omnibus_rows.append(
            {
                "benchmark": benchmark,
                "n_units": len(pivot),
                "friedman_q": statistic,
                "kendalls_w": statistic / (len(pivot) * (len(pivot.columns) - 1)),
                "p_raw": p_raw,
            }
        )
        for config_a, config_b in combinations(top_configs, 2):
            test = wilcoxon(pivot[config_a], pivot[config_b], alternative="two-sided")
            pair_rows.append(
                {
                    "benchmark": benchmark,
                    "config_a": config_a,
                    "config_b": config_b,
                    "median_difference": (pivot[config_a] - pivot[config_b]).median(),
                    "wilcoxon_w": test.statistic,
                    "p_raw": test.pvalue,
                    "rank_biserial": rank_biserial(pivot[config_a], pivot[config_b]),
                }
            )
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["p_holm"] = holm(omnibus["p_raw"])
    pairs = pd.DataFrame(pair_rows)
    pairs["p_holm"] = holm(pairs["p_raw"])
    return omnibus, pairs


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt_p(p_value: float) -> str:
    if p_value <= 0:
        # Notebook snapshots rounded to seven decimals; a displayed zero means
        # the original value was below half the last retained decimal.
        return r"<5\!\times\!10^{-8}"
    if p_value >= 0.001:
        return f"{p_value:.3f}".rstrip("0").rstrip(".")
    mantissa, exponent = f"{p_value:.1e}".split("e")
    return rf"{mantissa}\!\times\!10^{{{int(exponent)}}}"


def sig_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


# The workbooks spell SIB-200 without the hyphen; the paper always hyphenates,
# so every table that prints a benchmark name goes through this map.
BENCHMARK_LABELS = {"SIB200": "SIB-200"}


def benchmark_label(benchmark: object) -> str:
    return BENCHMARK_LABELS.get(str(benchmark), str(benchmark))


# Captions are prose, and the paper spells small counts out ("the five promoted
# configurations"), so counts interpolated into a caption go through this rather
# than printing as a numeral.
_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five",
                 "six", "seven", "eight", "nine", "ten"]


def spell_count(value: int) -> str:
    return _NUMBER_WORDS[value] if 0 <= value < len(_NUMBER_WORDS) else str(value)


def write_table(
    path: Path,
    col_spec: str,
    header_rows: list[str],
    body_rows: list[str],
    caption: str,
    label: str,
    environment: str = "table",
    size: str = "small",
    resize_to: str | None = None,
    placement: str | None = None,
    col_sep: str | None = None,
) -> None:
    begin_environment = rf"\begin{{{environment}}}"
    if placement is not None:
        begin_environment += rf"[{placement}]"
    content = [
        "% Auto-generated by paper_resources/scripts; do not edit by hand.",
        begin_environment,
        r"\centering",
        rf"\{size}",
    ]
    # Tightening the column separation keeps a table inside its column at the
    # body font size, which \resizebox would only achieve by shrinking the text.
    if col_sep is not None:
        content.append(rf"\setlength{{\tabcolsep}}{{{col_sep}}}")
    if resize_to is not None:
        content.append(rf"\resizebox{{{resize_to}}}{{!}}{{%")
    content.extend([
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        *header_rows,
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    if resize_to is not None:
        content.append("}")
    content.extend([
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\end{{{environment}}}",
        "",
    ])
    path.write_text("\n".join(content))
    print(f"wrote {path}")


def write_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    frame.to_csv(path, index=index)
    print(f"wrote {path}")
