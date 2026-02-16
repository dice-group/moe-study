#!/usr/bin/env python3
"""Export multilingual eval results to a styled Excel workbook.

Layout:
- Sheets: 1B, 3B, 8B, summary (summary is last).
- Each model sheet has 3 side-by-side tier blocks: 200 | 96 | 10.
- Each tier block is split into two independent tables:
  1) Belebele (top)
  2) FLORES (below, separated by spacer rows)
- Approach columns include baseline; 8B baseline is shown as PENDING_8B.
- Each row includes language resource category (high/mid/low/unknown).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import wandb

try:
    from openpyxl import Workbook
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo
except ImportError as exc:
    raise SystemExit(
        "openpyxl is required. Install with: pip install openpyxl or pip install -r requirements.txt"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "result_analysis" / "moe_eval_compact.xlsx"
RESOURCE_TSV = REPO_ROOT / "result_analysis" / "l2v_nested_samples" / "lang_resource_dataset.tsv"

RUN_NAME_PAT = re.compile(
    r"llama-3\.[12]-(?P<model>\d+B).*?(?P<tier>\d+)_langs-(?P<approach>[^-_]+)"
)
BASELINE_PATTERNS = [
    re.compile(r"llama\d+_(?P<size>\d+)b_.*?_base_(?P<tier>\d+)langs", re.IGNORECASE),
    re.compile(r"(?P<size>\d)b[_-]base[_-](?P<tier>\d+)langs", re.IGNORECASE),
]

APPROACH_MAP = {
    "adamole": "adamole",
    "mola": "mola",
    "adamoe_1": "mola",
    "adamoe1": "mola",
    "cola": "cola",
    "hydralora": "hydralora",
    "baseline": "baseline",
}

METRIC_LAYOUT = [
    ("belebele_acc_norm", "Belebele", "acc_norm", "higher"),
    ("flores_byte_ppl_en_to_x", "FLORES", "byte_perplexity en->x", "lower"),
    ("flores_byte_ppl_x_to_en", "FLORES", "byte_perplexity x->en", "lower"),
    ("flores_byte_ppl_avg2dir", "FLORES", "byte_perplexity avg2dir", "lower"),
]

BENCHMARK_ORDER = {"Belebele": 0, "FLORES": 1}
METRIC_ORDER = {metric_key: idx for idx, (metric_key, *_rest) in enumerate(METRIC_LAYOUT)}


def parse_csv_arg(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def parse_tasks(path: Path) -> List[str]:
    tasks: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    return tasks


def normalize_lang_code(lang: str) -> str:
    return lang.lower().replace("-", "_")


def map_resource_category(raw: str | None) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "unknown"
    x = raw.strip().lower().replace("*", "")
    if x.startswith("high"):
        return "high"
    if x.startswith("medhigh") or x == "mid":
        return "mid"
    if x.startswith("medlow") or x.startswith("low") or x.startswith("not_enough"):
        return "low"
    return "unknown"


def load_resource_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        code = normalize_lang_code(str(row.get("lang_code", "")))
        if not code:
            continue
        out[code] = map_resource_category(row.get("resource_category"))
    return out


def parse_tier_task_map(tiers: List[int]) -> Dict[int, Dict[str, dict]]:
    out: Dict[int, Dict[str, dict]] = {}
    for tier in tiers:
        task_path = REPO_ROOT / "configs" / f"lm_eval_tasks_{tier}_langs.txt"
        tasks = parse_tasks(task_path)
        order: List[str] = []
        task_map: Dict[str, dict] = {}

        def ensure_lang(lang: str) -> None:
            if lang not in task_map:
                order.append(lang)
                task_map[lang] = {
                    "bele_task": None,
                    "flo_en_to_x_task": None,
                    "flo_x_to_en_task": None,
                }

        for task in tasks:
            if task.startswith("belebele_"):
                lang = task.replace("belebele_", "")
                ensure_lang(lang)
                task_map[lang]["bele_task"] = task
                continue

            if not task.startswith("flores_en_perplexity_"):
                continue

            pair = task.replace("flores_en_perplexity_", "")
            if "-" not in pair:
                continue
            src, tgt = pair.split("-", 1)
            if src == "eng_Latn":
                ensure_lang(tgt)
                task_map[tgt]["flo_en_to_x_task"] = task
            elif tgt == "eng_Latn":
                ensure_lang(src)
                task_map[src]["flo_x_to_en_task"] = task

        out[tier] = {"order": order, "tasks": task_map}
    return out


def _run_ts(run: wandb.apis.public.Run) -> float:
    ts = run.summary.get("_timestamp")
    if ts is not None:
        try:
            return float(ts)
        except Exception:
            pass
    for attr in ("updated_at", "created_at"):
        val = getattr(run, attr, None)
        if val is None:
            continue
        if isinstance(val, str):
            try:
                return dt.datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
        if hasattr(val, "timestamp"):
            try:
                return float(val.timestamp())
            except Exception:
                continue
    return 0.0


def parse_run_identity(name: str) -> Tuple[str, int, str] | None:
    m = RUN_NAME_PAT.search(name or "")
    if m:
        model = m.group("model")
        tier = int(m.group("tier"))
        approach = APPROACH_MAP.get(m.group("approach"), m.group("approach"))
        return model, tier, approach

    for pat in BASELINE_PATTERNS:
        b = pat.search(name or "")
        if b:
            model = f"{int(b.group('size'))}B"
            tier = int(b.group("tier"))
            return model, tier, "baseline"

    return None


def select_latest_runs(
    api: wandb.Api,
    entity: str,
    project: str,
    models: List[str],
    tiers: List[int],
    approaches: List[str],
) -> Dict[Tuple[str, int, str], wandb.apis.public.Run]:
    runs = api.runs(f"{entity}/{project}")
    wanted_models = set(models)
    wanted_tiers = set(tiers)
    wanted_approaches = set(approaches)
    latest: Dict[Tuple[str, int, str], Tuple[float, wandb.apis.public.Run]] = {}

    for run in runs:
        ident = parse_run_identity(run.name or "")
        if ident is None:
            continue
        model, tier, approach = ident
        if model not in wanted_models or tier not in wanted_tiers or approach not in wanted_approaches:
            continue
        key = (model, tier, approach)
        ts = _run_ts(run)
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, run)

    return {k: v[1] for k, v in latest.items()}


def get_metric(summary: dict, task: str | None, metric: str) -> float | None:
    if not task:
        return None
    val = summary.get(f"{task}/{metric}")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def choose_best(values: Dict[str, object], better: str) -> Tuple[str | None, float | None]:
    valid = [(k, float(v)) for k, v in values.items() if isinstance(v, (int, float))]
    if not valid:
        return None, None
    best = max(valid, key=lambda kv: kv[1]) if better == "higher" else min(valid, key=lambda kv: kv[1])
    return best[0], best[1]


def resolve_run_for_combo(
    latest_runs: Dict[Tuple[str, int, str], wandb.apis.public.Run],
    model: str,
    tier: int,
    approach: str,
) -> wandb.apis.public.Run | None:
    run = latest_runs.get((model, tier, approach))
    if run is not None:
        return run

    # Baselines may only exist as a full-tier run (e.g. 200 langs); reuse the
    # largest available baseline run for that model when tier-specific runs are absent.
    if approach != "baseline":
        return None

    candidates = [
        (t, r)
        for (m, t, a), r in latest_runs.items()
        if m == model and a == "baseline"
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build_model_sheet_df(
    model: str,
    tiers: List[int],
    approaches: List[str],
    task_map_by_tier: Dict[int, Dict[str, dict]],
    latest_runs: Dict[Tuple[str, int, str], wandb.apis.public.Run],
    resource_map: Dict[str, str],
) -> pd.DataFrame:
    rows: List[dict] = []
    for tier in tiers:
        lang_order = task_map_by_tier[tier]["order"]
        tier_map = task_map_by_tier[tier]["tasks"]
        for lang in lang_order:
            tmap = tier_map[lang]
            resource = resource_map.get(normalize_lang_code(lang), "unknown")
            for metric_name, benchmark, metric_label, better in METRIC_LAYOUT:
                if metric_name == "belebele_acc_norm" and not tmap["bele_task"]:
                    continue
                if metric_name == "flores_byte_ppl_en_to_x" and not tmap["flo_en_to_x_task"]:
                    continue
                if metric_name == "flores_byte_ppl_x_to_en" and not tmap["flo_x_to_en_task"]:
                    continue
                if metric_name == "flores_byte_ppl_avg2dir":
                    if not (tmap["flo_en_to_x_task"] or tmap["flo_x_to_en_task"]):
                        continue

                values: Dict[str, object] = {}
                for approach in approaches:
                    run = resolve_run_for_combo(latest_runs, model, tier, approach)
                    if run is None:
                        values[approach] = None
                        continue

                    summary = dict(run.summary)
                    if metric_name == "belebele_acc_norm":
                        val = get_metric(summary, tmap["bele_task"], "acc_norm")
                    elif metric_name == "flores_byte_ppl_en_to_x":
                        val = get_metric(summary, tmap["flo_en_to_x_task"], "byte_perplexity")
                    elif metric_name == "flores_byte_ppl_x_to_en":
                        val = get_metric(summary, tmap["flo_x_to_en_task"], "byte_perplexity")
                    else:
                        v1 = get_metric(summary, tmap["flo_en_to_x_task"], "byte_perplexity")
                        v2 = get_metric(summary, tmap["flo_x_to_en_task"], "byte_perplexity")
                        parts = [v for v in (v1, v2) if isinstance(v, (int, float))]
                        val = float(sum(parts) / len(parts)) if parts else None
                    values[approach] = val

                if all(v is None for v in values.values()):
                    continue

                best_approach, best_value = choose_best(values, better)
                row = {
                    "tier": tier,
                    "benchmark": benchmark,
                    "metric": metric_label,
                    "language": lang,
                    "resource": resource,
                    "better": better,
                    "metric_key": metric_name,
                }
                for approach in approaches:
                    row[approach] = values.get(approach)
                row["best_approach"] = best_approach
                row["best_value"] = best_value
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["benchmark_rank"] = df["benchmark"].map(BENCHMARK_ORDER).fillna(999)
    df["metric_rank"] = df["metric_key"].map(METRIC_ORDER).fillna(999)
    df = df.sort_values(["tier", "benchmark_rank", "metric_rank", "language"]).drop(
        columns=["benchmark_rank", "metric_rank", "metric_key"]
    )
    return df


def build_summary_df(
    models: List[str],
    tiers: List[int],
    approaches: List[str],
    task_map_by_tier: Dict[int, Dict[str, dict]],
    latest_runs: Dict[Tuple[str, int, str], wandb.apis.public.Run],
) -> pd.DataFrame:
    rows: List[dict] = []
    for model in models:
        for tier in tiers:
            for approach in approaches:
                run = resolve_run_for_combo(latest_runs, model, tier, approach)
                if run is None:
                    rows.append(
                        {
                            "model_size": model,
                            "tier": tier,
                            "approach": approach,
                            "run_name": None,
                            "run_id": None,
                            "run_url": None,
                            "belebele_acc_norm_avg": None,
                            "flores_byte_ppl_avg2dir_avg": None,
                            "n_belebele_langs": 0,
                            "n_flores_langs": 0,
                        }
                    )
                    continue

                summary = dict(run.summary)
                langs = task_map_by_tier[tier]["order"]
                tmap = task_map_by_tier[tier]["tasks"]
                bele_vals = []
                flo_vals = []
                for lang in langs:
                    tasks = tmap[lang]
                    b = get_metric(summary, tasks["bele_task"], "acc_norm")
                    if b is not None:
                        bele_vals.append(b)
                    f1 = get_metric(summary, tasks["flo_en_to_x_task"], "byte_perplexity")
                    f2 = get_metric(summary, tasks["flo_x_to_en_task"], "byte_perplexity")
                    parts = [v for v in (f1, f2) if isinstance(v, (int, float))]
                    if parts:
                        flo_vals.append(float(sum(parts) / len(parts)))

                rows.append(
                    {
                        "model_size": model,
                        "tier": tier,
                        "approach": approach,
                        "run_name": run.name,
                        "run_id": run.id,
                        "run_url": run.url,
                        "belebele_acc_norm_avg": (sum(bele_vals) / len(bele_vals)) if bele_vals else None,
                        "flores_byte_ppl_avg2dir_avg": (sum(flo_vals) / len(flo_vals)) if flo_vals else None,
                        "n_belebele_langs": len(bele_vals),
                        "n_flores_langs": len(flo_vals),
                    }
                )
    return pd.DataFrame(rows)


def _set_col_width(ws, col_idx: int, width: float) -> None:
    col = get_column_letter(col_idx)
    existing = ws.column_dimensions[col].width or 0
    if width > existing:
        ws.column_dimensions[col].width = width


def _write_banner(ws, title: str, description: str, end_col: int) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=end_col)
    ws.cell(1, 1, title)
    ws.cell(1, 1).font = Font(color="FFFFFF", bold=True, size=14)
    ws.cell(1, 1).alignment = Alignment(horizontal="left", vertical="center")
    ws.cell(1, 1).fill = PatternFill(fill_type="solid", start_color="203864", end_color="203864")

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=end_col)
    ws.cell(2, 1, description)
    ws.cell(2, 1).font = Font(color="1F2937", size=10)
    ws.cell(2, 1).alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws.cell(2, 1).fill = PatternFill(fill_type="solid", start_color="DCE6F1", end_color="DCE6F1")

    ws.row_dimensions[1].height = 24
    ws.row_dimensions[2].height = 22
    ws.sheet_view.showGridLines = False


def _apply_row_color_scale(ws, row_idx: int, better: str, first_col: int, last_col: int) -> None:
    has_num = any(isinstance(ws.cell(row_idx, c).value, (int, float)) for c in range(first_col, last_col + 1))
    if not has_num:
        return
    rng = f"{get_column_letter(first_col)}{row_idx}:{get_column_letter(last_col)}{row_idx}"
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


def _write_subtable(
    ws,
    bench_name: str,
    bench_df: pd.DataFrame,
    start_row: int,
    start_col: int,
    approaches: List[str],
    table_name: str,
) -> int:
    headers = ["benchmark", "metric", "language", "resource", "better", *approaches, "best_approach", "best_value"]
    end_col = start_col + len(headers) - 1
    thin = Side(style="thin", color="B7C6D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws.merge_cells(start_row=start_row, start_column=start_col, end_row=start_row, end_column=end_col)
    ws.cell(start_row, start_col, f"{bench_name} Block")
    ws.cell(start_row, start_col).font = Font(color="FFFFFF", bold=True, size=10)
    ws.cell(start_row, start_col).alignment = Alignment(horizontal="left", vertical="center")
    ws.cell(start_row, start_col).fill = PatternFill(fill_type="solid", start_color="4F81BD", end_color="4F81BD")

    header_row = start_row + 1
    for idx, h in enumerate(headers):
        c = start_col + idx
        ws.cell(header_row, c, h)
        ws.cell(header_row, c).font = Font(color="FFFFFF", bold=True)
        ws.cell(header_row, c).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(header_row, c).fill = PatternFill(fill_type="solid", start_color="1F4E78", end_color="1F4E78")
        ws.cell(header_row, c).border = border

    body_cols = ["benchmark", "metric", "language", "resource", "better", *approaches, "best_approach", "best_value"]
    body_df = bench_df[body_cols].copy() if not bench_df.empty else pd.DataFrame(columns=body_cols)
    if body_df.empty:
        body_df = pd.DataFrame(
            [{"benchmark": bench_name, "metric": "no_data", "language": None, "resource": "unknown", "better": None}]
        )

    data_start = header_row + 1
    for r_off, (_, row) in enumerate(body_df.iterrows()):
        r = data_start + r_off
        for idx, h in enumerate(headers):
            c = start_col + idx
            val = row[h] if h in row else None
            ws.cell(r, c, val)
            ws.cell(r, c).border = border
            if h in {"benchmark", "metric", "language", "resource", "better", "best_approach"}:
                ws.cell(r, c).alignment = Alignment(horizontal="left", vertical="center")
            else:
                ws.cell(r, c).alignment = Alignment(horizontal="center", vertical="center")
                if isinstance(val, (int, float)):
                    ws.cell(r, c).number_format = "0.0000"

    last_row = data_start + len(body_df) - 1
    first_ap_col = start_col + 5
    last_ap_col = first_ap_col + len(approaches) - 1
    better_col = start_col + 4
    for r in range(data_start, last_row + 1):
        better = ws.cell(r, better_col).value
        if better in {"higher", "lower"}:
            _apply_row_color_scale(ws, r, better, first_ap_col, last_ap_col)

    tab = Table(
        displayName=table_name,
        ref=f"{get_column_letter(start_col)}{header_row}:{get_column_letter(end_col)}{last_row}",
    )
    tab.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(tab)
    return last_row


def _write_tier_block(
    ws,
    tier_df: pd.DataFrame,
    tier: int,
    start_row: int,
    start_col: int,
    approaches: List[str],
    table_name_prefix: str,
) -> None:
    headers = ["benchmark", "metric", "language", "resource", "better", *approaches, "best_approach", "best_value"]
    table_width = len(headers)
    end_col = start_col + table_width - 1

    ws.merge_cells(start_row=start_row, start_column=start_col, end_row=start_row, end_column=end_col)
    ws.cell(start_row, start_col, f"{tier} Languages")
    ws.cell(start_row, start_col).font = Font(color="FFFFFF", bold=True, size=12)
    ws.cell(start_row, start_col).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(start_row, start_col).fill = PatternFill(fill_type="solid", start_color="305496", end_color="305496")

    ws.merge_cells(start_row=start_row + 1, start_column=start_col, end_row=start_row + 1, end_column=end_col)
    ws.cell(start_row + 1, start_col, "Top: Belebele  |  Bottom: FLORES (separate filter blocks)")
    ws.cell(start_row + 1, start_col).font = Font(color="1F2937", italic=True, size=9)
    ws.cell(start_row + 1, start_col).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(start_row + 1, start_col).fill = PatternFill(fill_type="solid", start_color="E2ECF9", end_color="E2ECF9")

    row_cursor = start_row + 2
    bele_df = tier_df[tier_df["benchmark"] == "Belebele"].copy()
    flo_df = tier_df[tier_df["benchmark"] == "FLORES"].copy()
    row_cursor = _write_subtable(
        ws, "Belebele", bele_df, row_cursor, start_col, approaches, f"{table_name_prefix}_B"
    )
    row_cursor += 3  # 2 spacer rows between belebele and flores
    _write_subtable(ws, "FLORES", flo_df, row_cursor, start_col, approaches, f"{table_name_prefix}_F")

    widths = [12, 25, 14, 9, 9] + [12 for _ in approaches] + [14, 11]
    for idx, width in enumerate(widths):
        _set_col_width(ws, start_col + idx, width)


def write_model_sheet(
    ws,
    model: str,
    model_df: pd.DataFrame,
    tiers: List[int],
    approaches: List[str],
) -> None:
    table_width = 5 + len(approaches) + 2
    gap = 3
    tier_order = sorted(tiers, reverse=True)
    start_row = 5
    start_cols = {tier: 1 + i * (table_width + gap) for i, tier in enumerate(tier_order)}
    last_col = max(start_cols.values()) + table_width - 1
    _write_banner(
        ws,
        f"Model {model} - Tiered Eval Comparison",
        "Rows include resource tags (high/mid/low). Baseline column included; 8B baseline is currently PENDING_8B.",
        last_col,
    )

    for idx, tier in enumerate(tier_order):
        _write_tier_block(
            ws=ws,
            tier_df=model_df[model_df["tier"] == tier].copy(),
            tier=tier,
            start_row=start_row,
            start_col=start_cols[tier],
            approaches=approaches,
            table_name_prefix=f"T_{model}_{tier}_{idx}",
        )


def write_summary_sheet(ws, summary_df: pd.DataFrame) -> None:
    cols = [
        "model_size",
        "tier",
        "approach",
        "belebele_acc_norm_avg",
        "flores_byte_ppl_avg2dir_avg",
        "n_belebele_langs",
        "n_flores_langs",
        "run_name",
        "run_id",
        "run_url",
    ]
    sdf = summary_df[cols].sort_values(["model_size", "tier", "approach"]).reset_index(drop=True)
    _write_banner(
        ws,
        "Summary - Tier-Level Averages",
        "Higher is better for Belebele; lower is better for FLORES byte perplexity.",
        len(cols),
    )

    header_row = 4
    for i, c in enumerate(cols, start=1):
        ws.cell(header_row, i, c)
        ws.cell(header_row, i).fill = PatternFill(fill_type="solid", start_color="1F4E78", end_color="1F4E78")
        ws.cell(header_row, i).font = Font(color="FFFFFF", bold=True)
        ws.cell(header_row, i).alignment = Alignment(horizontal="center", vertical="center")

    thin = Side(style="thin", color="B7C6D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for r_off, (_, row) in enumerate(sdf.iterrows(), start=1):
        r = header_row + r_off
        for i, c in enumerate(cols, start=1):
            val = row[c]
            ws.cell(r, i, val)
            ws.cell(r, i).border = border
            if c in {"belebele_acc_norm_avg", "flores_byte_ppl_avg2dir_avg"} and isinstance(val, (int, float)):
                ws.cell(r, i).number_format = "0.0000"
        if isinstance(ws.cell(r, 10).value, str) and ws.cell(r, 10).value.startswith("http"):
            ws.cell(r, 10).hyperlink = ws.cell(r, 10).value
            ws.cell(r, 10).style = "Hyperlink"

    last_row = header_row + len(sdf)
    tab = Table(displayName="T_Summary", ref=f"A{header_row}:{get_column_letter(len(cols))}{last_row}")
    tab.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium9",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(tab)

    ws.freeze_panes = f"A{header_row + 1}"
    ws.sheet_view.showGridLines = False
    for idx, width in enumerate([10, 7, 11, 21, 24, 12, 12, 48, 12, 50], start=1):
        _set_col_width(ws, idx, width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export W&B eval results to a styled Excel workbook.")
    parser.add_argument("--entity", default="dice-nlp")
    parser.add_argument("--project", default="moe-study-comparison_eval")
    parser.add_argument(
        "--baseline-project",
        default=None,
        help="Optional W&B project to source baseline runs from (approach=baseline).",
    )
    parser.add_argument(
        "--baseline-entity",
        default=None,
        help="Optional W&B entity for --baseline-project. Defaults to --entity.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--models", default="1B,3B,8B")
    parser.add_argument("--tiers", default="10,96,200")
    parser.add_argument("--approaches", default="adamole,mola,cola,hydralora,baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_csv_arg(args.models)
    tiers = [int(x) for x in parse_csv_arg(args.tiers)]
    approaches = parse_csv_arg(args.approaches)
    out_path = Path(args.out)

    task_map_by_tier = parse_tier_task_map(tiers)
    resource_map = load_resource_map(RESOURCE_TSV)
    api = wandb.Api()
    latest_runs = select_latest_runs(api, args.entity, args.project, models, tiers, approaches)
    if args.baseline_project:
        baseline_entity = args.baseline_entity or args.entity
        baseline_runs = select_latest_runs(
            api,
            baseline_entity,
            args.baseline_project,
            models,
            tiers,
            ["baseline"],
        )
        latest_runs.update(baseline_runs)

    expected = {(m, t, a) for m in models for t in tiers for a in approaches}
    missing = sorted(
        combo for combo in expected if resolve_run_for_combo(latest_runs, combo[0], combo[1], combo[2]) is None
    )
    if missing:
        print(f"[WARN] Missing latest runs for {len(missing)} combinations.")

    model_frames = {
        model: build_model_sheet_df(model, tiers, approaches, task_map_by_tier, latest_runs, resource_map)
        for model in models
    }
    summary_df = build_summary_df(models, tiers, approaches, task_map_by_tier, latest_runs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)

    for model in models:
        ws = wb.create_sheet(title=model)
        write_model_sheet(ws, model, model_frames[model], tiers=tiers, approaches=approaches)

    summary_ws = wb.create_sheet(title="summary")
    write_summary_sheet(summary_ws, summary_df)

    for ws in wb.worksheets:
        ws.sheet_properties.tabColor = "1F4E78" if ws.title != "summary" else "203864"

    wb.save(out_path)
    print(f"Wrote {out_path}")
    print("Sheets:", ", ".join(models + ["summary"]))


if __name__ == "__main__":
    main()
