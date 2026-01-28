#!/usr/bin/env python3
"""Generate eval report markdown + plots from W&B projects.

Produces:
- result_analysis/moe_eval_results.md
- result_analysis/plots/*.png

Assumes you are logged in to W&B (wandb.login)."""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

REPO_ROOT = Path("/home/dontspamtomuch/repos/moe-study")
OUT_MD = REPO_ROOT / "result_analysis" / "moe_eval_results.md"
PLOTS_DIR = REPO_ROOT / "result_analysis" / "plots"

ENTITY = "dice-nlp"
PROJ_COMPARISON = "moe-study-comparison"
PROJ_EVAL = "moe-study-comparison_eval"

TIERS = [10, 96, 200]
TIER_LABEL = {10: "high", 96: "mid", 200: "low"}

# Tasks lists used for aggregation
TASK_FILES = {
    10: REPO_ROOT / "configs" / "lm_eval_tasks_10_langs.txt",
    96: REPO_ROOT / "configs" / "lm_eval_tasks_96_langs.txt",
    200: REPO_ROOT / "configs" / "lm_eval_tasks_200_langs.txt",
}

APPROACH_ORDER = ["adamole", "cola", "hydralora", "moelpr", "mola"]
MODEL_ORDER = ["1B", "3B", "8B"]


@dataclass
class EvalLossEntry:
    loss: float | None
    date: str | None
    run_name: str


def _ts_key(run) -> float:
    ts = run.summary.get("_timestamp")
    if ts is not None:
        return float(ts)
    for attr in ("updated_at", "created_at"):
        val = getattr(run, attr, None)
        if val:
            if isinstance(val, str):
                try:
                    return dt.datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
                except Exception:
                    pass
            if hasattr(val, "timestamp"):
                return val.timestamp()
    return 0.0


def _format_date(run) -> str | None:
    ts = run.summary.get("_timestamp")
    if ts is not None:
        return dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc).strftime("%Y-%m-%d")
    updated = getattr(run, "updated_at", None)
    if isinstance(updated, str):
        try:
            return dt.datetime.fromisoformat(updated.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            return None
    if hasattr(updated, "strftime"):
        return updated.strftime("%Y-%m-%d")
    return None


def _parse_tasks(path: Path) -> List[str]:
    tasks = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    return tasks


def _latest_eval_loss(api: wandb.Api) -> Dict[Tuple[int, str, str], EvalLossEntry]:
    """Return latest eval/loss per (tier, approach, model_size)."""
    runs = api.runs(f"{ENTITY}/{PROJ_COMPARISON}")
    pat = re.compile(r"(?P<model>l\d-\d+b).*?(?P<langs>\d+)_langs-(?P<approach>[^-]+)")

    latest: Dict[Tuple[int, str, str], Tuple[float, wandb.apis.public.Run]] = {}
    for r in runs:
        m = pat.search(r.name)
        if not m:
            continue
        tier = int(m.group("langs"))
        approach = m.group("approach")
        model_raw = m.group("model")  # l3-1b
        size_match = re.search(r"l\\d-(\\d+)b", model_raw)
        model_size = f"{int(size_match.group(1))}B" if size_match else "unknown"
        key = (tier, approach, model_size)
        t = _ts_key(r)
        if key not in latest or t > latest[key][0]:
            latest[key] = (t, r)

    out: Dict[Tuple[int, str, str], EvalLossEntry] = {}
    for key, (_, r) in latest.items():
        out[key] = EvalLossEntry(
            loss=r.summary.get("eval/loss"),
            date=_format_date(r),
            run_name=r.name,
        )
    return out


def _latest_eval_aggregates(api: wandb.Api) -> Dict[int, Dict[str, dict]]:
    """Aggregate FLORES byte_ppl and Belebele acc_norm for Llama-3.1-8B eval runs."""
    runs = api.runs(f"{ENTITY}/{PROJ_EVAL}")
    pat = re.compile(r"llama-3\.1-8B.*?(?P<langs>\d+)_langs-(?P<approach>[^-_]+)")

    def has_flores(run) -> bool:
        return any("flores_en_perplexity" in k for k in run.summary.keys())

    latest: Dict[Tuple[int, str], Tuple[float, wandb.apis.public.Run]] = {}
    for r in runs:
        m = pat.search(r.name)
        if not m:
            continue
        if not r.summary:
            continue
        if not has_flores(r):
            continue
        tier = int(m.group("langs"))
        approach = m.group("approach")
        key = (tier, approach)
        t = _ts_key(r)
        if key not in latest or t > latest[key][0]:
            latest[key] = (t, r)

    results: Dict[int, Dict[str, dict]] = {tier: {} for tier in TIERS}

    for tier, task_path in TASK_FILES.items():
        tasks = _parse_tasks(task_path)
        flores = [t for t in tasks if t.startswith("flores_en_perplexity_")]
        bele = [t for t in tasks if t.startswith("belebele_")]
        for approach in APPROACH_ORDER:
            entry = latest.get((tier, approach))
            if not entry:
                results[tier][approach] = {
                    "flores_avg": None,
                    "belebele_avg": None,
                    "flores_n": 0,
                    "belebele_n": 0,
                    "run_name": None,
                }
                continue
            r = entry[1]
            flores_vals = []
            for task in flores:
                val = r.summary.get(f"{task}/byte_perplexity")
                if val is not None:
                    flores_vals.append(val)
            bele_vals = []
            for task in bele:
                val = r.summary.get(f"{task}/acc_norm")
                if val is not None:
                    bele_vals.append(val)
            results[tier][approach] = {
                "flores_avg": sum(flores_vals) / len(flores_vals) if flores_vals else None,
                "belebele_avg": sum(bele_vals) / len(bele_vals) if bele_vals else None,
                "flores_n": len(flores_vals),
                "belebele_n": len(bele_vals),
                "run_name": r.name,
            }
    return results


def _plot_grouped_bar(
    out_path: Path,
    title: str,
    x_labels: List[str],
    series: Dict[str, List[float | None]],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(x_labels))
    width = 0.14

    series_items = list(series.items())
    for i, (name, values) in enumerate(series_items):
        offsets = x + (i - (len(series_items) - 1) / 2) * width
        clean_vals = [v if v is not None else np.nan for v in values]
        ax.bar(offsets, clean_vals, width, label=name)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _render_eval_loss_tables(latest: Dict[Tuple[int, str, str], EvalLossEntry]) -> str:
    lines = []
    for tier in TIERS:
        lines.append(f"### {tier}_langs")
        header = "| Approach | " + " | ".join(MODEL_ORDER) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(MODEL_ORDER) + 1)) + "|")
        for approach in APPROACH_ORDER:
            row = [approach]
            for model_size in MODEL_ORDER:
                entry = latest.get((tier, approach, model_size))
                if not entry or entry.loss is None:
                    row.append("—")
                else:
                    date = entry.date or ""
                    row.append(f"{entry.loss:.4f} ({date})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def _render_eval_agg_tables(results: Dict[int, Dict[str, dict]]) -> str:
    lines = []
    for tier in TIERS:
        lines.append(f"### {TIER_LABEL[tier]} resource ({tier}_langs)")
        lines.append("| Approach | FLORES avg byte_ppl | n_flores | Belebele avg acc_norm | n_belebele |")
        lines.append("|---|---|---|---|---|")
        for approach in APPROACH_ORDER:
            entry = results[tier].get(approach, {})
            f = entry.get("flores_avg")
            b = entry.get("belebele_avg")
            fstr = f"{f:.4f}" if isinstance(f, (int, float)) else "—"
            bstr = f"{b:.4f}" if isinstance(b, (int, float)) else "—"
            lines.append(
                f"| {approach} | {fstr} | {entry.get('flores_n', 0)} | {bstr} | {entry.get('belebele_n', 0)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_best_summary(results: Dict[int, Dict[str, dict]]) -> str:
    lines = []
    for tier in TIERS:
        entries = results[tier]
        flores_candidates = [
            (v["flores_avg"], k) for k, v in entries.items() if isinstance(v.get("flores_avg"), (int, float))
        ]
        bele_candidates = [
            (v["belebele_avg"], k) for k, v in entries.items() if isinstance(v.get("belebele_avg"), (int, float))
        ]
        if flores_candidates:
            flores_best = min(flores_candidates)
            lines.append(f"- {TIER_LABEL[tier]}: FLORES best = {flores_best[1]} ({flores_best[0]:.4f})")
        if bele_candidates:
            bele_best = max(bele_candidates)
            lines.append(f"- {TIER_LABEL[tier]}: Belebele best = {bele_best[1]} ({bele_best[0]:.4f})")
    return "\n".join(lines)


def main() -> None:
    api = wandb.Api()

    latest_loss = _latest_eval_loss(api)
    eval_aggs = _latest_eval_aggregates(api)

    # Plots: eval loss by tier per model size (1B/3B/8B)
    for model_size in MODEL_ORDER:
        series = {}
        for approach in APPROACH_ORDER:
            vals = []
            for tier in TIERS:
                entry = latest_loss.get((tier, approach, model_size))
                vals.append(entry.loss if entry else None)
            series[approach] = vals
        _plot_grouped_bar(
            PLOTS_DIR / f"eval_loss_{model_size}.png",
            f"Eval loss by tier ({model_size})",
            [str(t) for t in TIERS],
            series,
            "eval/loss",
        )

    # Plots: FLORES byte ppl & Belebele acc_norm
    flores_series = {}
    bele_series = {}
    for approach in APPROACH_ORDER:
        flores_series[approach] = [
            eval_aggs[tier].get(approach, {}).get("flores_avg") for tier in TIERS
        ]
        bele_series[approach] = [
            eval_aggs[tier].get(approach, {}).get("belebele_avg") for tier in TIERS
        ]

    _plot_grouped_bar(
        PLOTS_DIR / "flores_byte_ppl.png",
        "FLORES avg byte perplexity by tier",
        [str(t) for t in TIERS],
        flores_series,
        "byte_perplexity",
    )
    _plot_grouped_bar(
        PLOTS_DIR / "belebele_acc_norm.png",
        "Belebele avg acc_norm by tier",
        [str(t) for t in TIERS],
        bele_series,
        "acc_norm",
    )

    # Markdown
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = [
        "# MoE Study Eval Results",
        "",
        f"Generated: {now}",
        "",
        "## Source projects",
        f"- W&B: `{ENTITY}/{PROJ_COMPARISON}`",
        f"- W&B: `{ENTITY}/{PROJ_EVAL}`",
        "",
        "## Eval loss (latest per tier/approach/model)",
        _render_eval_loss_tables(latest_loss),
        "",
        "## Aggregated eval (Llama-3.1-8B)",
        "FLORES uses `byte_perplexity` (word perplexity can be unstable across scripts).",
        _render_eval_agg_tables(eval_aggs),
        "",
        "### Best per tier",
        _render_best_summary(eval_aggs),
        "",
        "## Plots",
        "### Eval loss (per model size)",
        "![](plots/eval_loss_1B.png)",
        "![](plots/eval_loss_3B.png)",
        "![](plots/eval_loss_8B.png)",
        "",
        "### FLORES byte perplexity",
        "![](plots/flores_byte_ppl.png)",
        "",
        "### Belebele acc_norm",
        "![](plots/belebele_acc_norm.png)",
        "",
        "## Notes",
        "- Latest run per group is chosen by `_timestamp` in W&B summary (fallback: updated_at).",
        "- Resource tiers map: 10_langs=high, 96_langs=mid, 200_langs=low.",
        "- Belebele aggregation uses `acc_norm` over the task list in `configs/lm_eval_tasks_{tier}_langs.txt`.",
        "",
    ]

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
