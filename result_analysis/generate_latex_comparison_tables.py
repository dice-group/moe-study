#!/usr/bin/env python3
"""Generate LaTeX tables comparing approaches per model size.

Outputs a .tex file with 3 tables (1B/3B/8B), each grouped by lang tier (10/96/200)
with separators and bolded winners per metric within each tier.

Usage:
  python3 result_analysis/generate_latex_comparison_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO = Path("/home/dontspamtomuch/repos/moe-study")
EVAL_CSV = REPO / "result_analysis" / "wandb_exports" / "moe-study-comparison_eval_runs.csv"
BASELINE_CSV = (
    REPO
    / "result_analysis"
    / "wandb_exports"
    / "htyllm-adapter-lpr-200_lang_cola_eval_runs.csv"
)
OUT_TEX = REPO / "result_analysis" / "approach_comparison_tables.tex"
OUT_CSV = REPO / "result_analysis" / "approach_comparison_tables.csv"

TASK_FILES = {
    10: REPO / "configs" / "lm_eval_tasks_10_langs.txt",
    96: REPO / "configs" / "lm_eval_tasks_96_langs.txt",
    200: REPO / "configs" / "lm_eval_tasks_200_langs.txt",
}

APPROACHES = ["adamole", "mola", "cola", "hydralora", "baseline"]
MODEL_SIZES = ["1B", "3B", "8B"]
TIERS = [10, 96, 200]

MERGE_MAP = {
    "high": "high",
    "medhigh": "mid",
    "medlow": "low",
    "low": "low",
    "not_enough": "low",
}


def load_tasks(path: Path) -> List[str]:
    lines: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def task_to_lang_code(task: str) -> str | None:
    if task.startswith("belebele_"):
        code = task.replace("belebele_", "")
        return code.lower().replace("-", "_")
    if task.startswith("flores_en_perplexity_"):
        rest = task.replace("flores_en_perplexity_", "")
        parts = rest.split("-")
        if len(parts) == 2:
            src, tgt = parts
            if src == "eng_Latn":
                code = tgt
            elif tgt == "eng_Latn":
                code = src
            else:
                code = tgt
        else:
            code = parts[-1]
        return code.lower().replace("-", "_")
    return None


def avg_for_tasks(row: pd.Series, tasks: List[str], metric: str) -> float | None:
    cols = [f"summary.{t}/{metric}" for t in tasks]
    vals = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            vals.append(row[c])
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def avg_for_tasks_from_runs(runs: pd.DataFrame, tasks: List[str], metric: str) -> float | None:
    if runs.empty:
        return None
    cols = [f"summary.{t}/{metric}" for t in tasks]
    vals = []
    for _, row in runs.iterrows():
        for c in cols:
            if c in row and pd.notna(row[c]):
                vals.append(row[c])
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def fmt(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "-"
    return f"{val:.4f}"


def bold_if_best(val: float | None, best: float | None) -> str:
    if val is None or pd.isna(val) or best is None or pd.isna(best):
        return "-"
    if abs(val - best) < 1e-12:
        return f"\\textbf{{{val:.4f}}}"
    return f"{val:.4f}"


def main() -> None:
    runs = pd.read_csv(EVAL_CSV)
    no_ids = runs[runs["name"].str.contains("no_ids_detailed", na=False)].copy()

    pat = re.compile(r"llama-3\.[12]-(?P<model>\d+B).*?(?P<langs>\d+)_langs-(?P<approach>[^-_]+)")

    def parse_name(name: str) -> pd.Series:
        m = pat.search(str(name))
        if not m:
            return pd.Series({"model_size": None, "langs": None, "approach": None})
        return pd.Series({
            "model_size": m.group("model"),
            "langs": int(m.group("langs")),
            "approach": m.group("approach"),
        })

    parsed = no_ids["name"].apply(parse_name)
    no_ids = pd.concat([no_ids, parsed], axis=1)

    approach_map = {
        "adamoe_1": "mola",
        "adamoe1": "mola",
        "mola": "mola",
        "adamole": "adamole",
        "cola": "cola",
        "hydralora": "hydralora",
    }
    no_ids["approach"] = no_ids["approach"].map(lambda x: approach_map.get(x, x))

    # Resource tiers
    res_path = REPO / "result_analysis" / "l2v_nested_samples" / "lang_resource_dataset.tsv"
    resource_df = pd.read_csv(res_path, sep="\t")
    resource_df["resource_merged"] = resource_df["resource_category"].str.lower().map(MERGE_MAP)
    resource_map = resource_df.set_index("lang_code")["resource_merged"].to_dict()

    # Task metadata per tier
    task_meta: Dict[int, pd.DataFrame] = {}
    for tier, path in TASK_FILES.items():
        tasks = load_tasks(path)
        meta_rows = []
        for t in tasks:
            if t.startswith("belebele_"):
                metric = "acc_norm"
            elif t.startswith("flores_en_perplexity_"):
                metric = "byte_perplexity"
            else:
                continue
            lang_code = task_to_lang_code(t)
            meta_rows.append({
                "task": t,
                "metric": metric,
                "lang_code": lang_code,
                "resource": resource_map.get(lang_code),
                "family": None,
            })
        task_meta[tier] = pd.DataFrame(meta_rows)

    # Resource counts per tier per dataset (unique language codes)
    def counts_for_metric(meta: pd.DataFrame, metric: str) -> Dict[str, int]:
        counts = {"high": 0, "mid": 0, "low": 0, "unknown": 0, "total": 0}
        if meta.empty:
            return counts
        uniq = (
            meta[meta["metric"] == metric]
            .dropna(subset=["lang_code"])
            .drop_duplicates(subset=["lang_code"])
        )
        counts["total"] = int(len(uniq))
        for res in ["high", "mid", "low"]:
            counts[res] = int((uniq["resource"] == res).sum())
        counts["unknown"] = counts["total"] - counts["high"] - counts["mid"] - counts["low"]
        return counts

    resource_counts: Dict[int, Dict[str, Dict[str, int]]] = {}
    for tier, meta in task_meta.items():
        resource_counts[tier] = {
            "belebele": counts_for_metric(meta, "acc_norm"),
            "flores": counts_for_metric(meta, "byte_perplexity"),
        }

    # Header lines for output file (resource counts once, at top)
    header_lines: List[str] = []
    header_lines.append("% Auto-generated by generate_latex_comparison_tables.py")
    header_lines.append("% Required packages: booktabs, xcolor, colortbl, makecell, graphicx")
    header_lines.append("H = high, M = mid, L = low - resource \\\\")
    def fmt_counts(counts: Dict[str, int]) -> str:
        return "total=%d (H=%d, M=%d, L=%d, unknown=%d)" % (
            counts["total"],
            counts["high"],
            counts["mid"],
            counts["low"],
            counts["unknown"],
        )

    c10_b = resource_counts.get(10, {}).get("belebele", {})
    c96_b = resource_counts.get(96, {}).get("belebele", {})
    c200_b = resource_counts.get(200, {}).get("belebele", {})
    c10_f = resource_counts.get(10, {}).get("flores", {})
    c96_f = resource_counts.get(96, {}).get("flores", {})
    c200_f = resource_counts.get(200, {}).get("flores", {})

    header_lines.append("Belebele resource counts (unique langs): \\\\")
    header_lines.append(
        "10 langs: %s; \\\\"
        % fmt_counts(c10_b)
    )
    header_lines.append(
        "96 langs: %s; \\\\"
        % fmt_counts(c96_b)
    )
    header_lines.append(
        "200 langs: %s \\\\"
        % fmt_counts(c200_b)
    )
    header_lines.append("")
    header_lines.append("\\noindent FLORES resource counts (unique langs): \\\\")
    header_lines.append(
        "10 langs: %s; \\\\"
        % fmt_counts(c10_f)
    )
    header_lines.append(
        "96 langs: %s; \\\\"
        % fmt_counts(c96_f)
    )
    header_lines.append(
        "200 langs: %s \\\\"
        % fmt_counts(c200_f)
    )
    header_lines.append("")

    # Build per-run records
    records = []
    for _, row in no_ids.iterrows():
        model_size = row.get("model_size")
        langs = row.get("langs")
        approach = row.get("approach")
        if pd.isna(model_size) or pd.isna(langs) or pd.isna(approach):
            continue
        tier = int(langs)
        meta = task_meta.get(tier)
        if meta is None:
            continue

        bele_tasks = meta[meta["metric"] == "acc_norm"]["task"].tolist()
        flo_tasks = meta[meta["metric"] == "byte_perplexity"]["task"].tolist()
        bele_avg = avg_for_tasks(row, bele_tasks, "acc_norm")
        flo_avg = avg_for_tasks(row, flo_tasks, "byte_perplexity")

        res_avgs = {}
        for res in ["high", "mid", "low"]:
            sub = meta[(meta["metric"] == "acc_norm") & (meta["resource"] == res)]["task"].tolist()
            res_avgs[f"bele_{res}"] = avg_for_tasks(row, sub, "acc_norm")
            sub = meta[(meta["metric"] == "byte_perplexity") & (meta["resource"] == res)]["task"].tolist()
            res_avgs[f"flo_{res}"] = avg_for_tasks(row, sub, "byte_perplexity")

        # timestamp for tie-breaks if multiple runs per group
        ts = row.get("summary._timestamp") if "summary._timestamp" in row else None

        records.append({
            "model_size": model_size,
            "langs": tier,
            "approach": approach,
            "bele_all": bele_avg,
            "flo_all": flo_avg,
            "bele_high": res_avgs.get("bele_high"),
            "flo_high": res_avgs.get("flo_high"),
            "bele_mid": res_avgs.get("bele_mid"),
            "flo_mid": res_avgs.get("flo_mid"),
            "bele_low": res_avgs.get("bele_low"),
            "flo_low": res_avgs.get("flo_low"),
            "_ts": ts,
        })

    # Add baseline (from separate project), if available
    baseline_run_ids = {
        "razk494k",
        "yebet51e",
        "ol4o87kg",
        "3gt3bkrt",
        "92o8v53w",
    }
    if BASELINE_CSV.exists():
        try:
            baseline_df = pd.read_csv(BASELINE_CSV)
        except pd.errors.EmptyDataError:
            baseline_df = pd.DataFrame()
        baseline_df = (
            baseline_df[baseline_df["run_id"].isin(baseline_run_ids)]
            if not baseline_df.empty and "run_id" in baseline_df.columns
            else baseline_df
        )
        if not baseline_df.empty:
            bele_runs = baseline_df[baseline_df["name"].str.contains("belebele", na=False)]
            flo_runs = baseline_df[baseline_df["name"].str.contains("flores", na=False)]

            for tier in TIERS:
                meta = task_meta.get(tier)
                if meta is None or meta.empty:
                    continue
                bele_tasks = meta[meta["metric"] == "acc_norm"]["task"].tolist()
                flo_tasks = meta[meta["metric"] == "byte_perplexity"]["task"].tolist()

                def tasks_for(resource: str, metric: str) -> List[str]:
                    return meta[(meta["metric"] == metric) & (meta["resource"] == resource)][
                        "task"
                    ].tolist()

                records.append({
                    "model_size": "8B",
                    "langs": tier,
                    "approach": "baseline",
                    "bele_all": avg_for_tasks_from_runs(bele_runs, bele_tasks, "acc_norm"),
                    "flo_all": avg_for_tasks_from_runs(flo_runs, flo_tasks, "byte_perplexity"),
                    "bele_high": avg_for_tasks_from_runs(bele_runs, tasks_for("high", "acc_norm"), "acc_norm"),
                    "flo_high": avg_for_tasks_from_runs(flo_runs, tasks_for("high", "byte_perplexity"), "byte_perplexity"),
                    "bele_mid": avg_for_tasks_from_runs(bele_runs, tasks_for("mid", "acc_norm"), "acc_norm"),
                    "flo_mid": avg_for_tasks_from_runs(flo_runs, tasks_for("mid", "byte_perplexity"), "byte_perplexity"),
                    "bele_low": avg_for_tasks_from_runs(bele_runs, tasks_for("low", "acc_norm"), "acc_norm"),
                    "flo_low": avg_for_tasks_from_runs(flo_runs, tasks_for("low", "byte_perplexity"), "byte_perplexity"),
                })

    df = pd.DataFrame(records)
    df = df[df["approach"].isin(APPROACHES)]

    # If multiple runs per (model_size, langs, approach), keep the latest by timestamp
    if "_ts" in df.columns:
        df = df.sort_values(["model_size", "langs", "approach", "_ts"]).drop_duplicates(
            subset=["model_size", "langs", "approach"], keep="last"
        )
    df["approach"] = pd.Categorical(df["approach"], categories=APPROACHES, ordered=True)
    df = df.sort_values(["model_size", "langs", "approach"]).drop(columns=["_ts"], errors="ignore")
    df.to_csv(OUT_CSV, index=False)

    metric_cols: List[Tuple[str, bool]] = [
        ("bele_all", True),
        ("flo_all", False),
        ("bele_high", True),
        ("flo_high", False),
        ("bele_mid", True),
        ("flo_mid", False),
        ("bele_low", True),
        ("flo_low", False),
    ]

    lines: List[str] = list(header_lines)
    for model_size in MODEL_SIZES:
        sub = df[df["model_size"] == model_size]
        if sub.empty:
            continue

        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\tiny")
        lines.append("\\setlength{\\tabcolsep}{2pt}")
        lines.append("\\renewcommand{\\arraystretch}{1.05}")
        lines.append("\\rowcolors{3}{blue!6}{white}")
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.append("\\begin{tabular}{lrrrrrrrr}")
        lines.append("\\toprule")
        lines.append("\\rowcolor{blue!18}")
        lines.append("approach & \\makecell{Belebele\\\\acc\\_norm (all)} & \\makecell{FLORES\\\\byte\\_ppl (all)} & \\makecell{Belebele\\\\acc\\_norm (high)} & \\makecell{FLORES\\\\byte\\_ppl (high)} & \\makecell{Belebele\\\\acc\\_norm (mid)} & \\makecell{FLORES\\\\byte\\_ppl (mid)} & \\makecell{Belebele\\\\acc\\_norm (low)} & \\makecell{FLORES\\\\byte\\_ppl (low)} \\\\")
        lines.append("\\midrule")

        for tier in TIERS:
            tier_rows = sub[sub["langs"] == tier]
            if tier_rows.empty:
                continue
            lines.append("\\rowcolor{blue!12}")
            lines.append("\\multicolumn{9}{l}{\\textbf{%s langs}} \\\\" % tier)

            # Compute winners per metric within this tier
            best: Dict[str, float | None] = {}
            for col, higher_better in metric_cols:
                vals = tier_rows[col].dropna().tolist()
                if not vals:
                    best[col] = None
                    continue
                best[col] = max(vals) if higher_better else min(vals)

            for _, r in tier_rows.iterrows():
                row_vals = []
                for col, _ in metric_cols:
                    row_vals.append(bold_if_best(r[col], best[col]))
                lines.append(
                    "%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
                        r["approach"],
                        row_vals[0], row_vals[1], row_vals[2], row_vals[3],
                        row_vals[4], row_vals[5], row_vals[6], row_vals[7],
                    )
                )
            lines.append("\\midrule")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}%")
        lines.append("}")
        lines.append("\\caption{%s models: overall + resource-tier averages (bold = best within tier)}" % model_size)
        lines.append("\\end{table}")
        lines.append("")

    OUT_TEX.write_text("\n".join(lines))
    print("Wrote", OUT_TEX)
    print("Wrote", OUT_CSV)


if __name__ == "__main__":
    main()
