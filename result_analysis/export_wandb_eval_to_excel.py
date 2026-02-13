#!/usr/bin/env python3
"""Create a compact, color-coded Excel report from W&B eval runs.

Workbook layout:
- One sheet per model size (1B/3B/8B) with 3 side-by-side tier tables (200, 96, 10).
- Rows grouped by benchmark + metric to compare approaches per language quickly.
- A summary sheet as the last tab.
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
        "openpyxl is required. Install with: pip install openpyxl "
        "or pip install -r requirements.txt"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "result_analysis" / "moe_eval_compact.xlsx"

RUN_NAME_PAT = re.compile(
    r"llama-3\.[12]-(?P<model>\d+B).*?(?P<tier>\d+)_langs-(?P<approach>[^-_]+)"
)

APPROACH_MAP = {
    "adamole": "adamole",
    "mola": "mola",
    "adamoe_1": "mola",
    "adamoe1": "mola",
    "cola": "cola",
    "hydralora": "hydralora",
}

METRIC_ORDER = {
    "belebele_acc_norm": 0,
    "flores_byte_ppl_en_to_x": 1,
    "flores_byte_ppl_x_to_en": 2,
    "flores_byte_ppl_avg2dir": 3,
}

METRIC_LAYOUT = [
    ("belebele_acc_norm", "Belebele", "acc_norm", "higher"),
    ("flores_byte_ppl_en_to_x", "FLORES", "byte_perplexity en->x", "lower"),
    ("flores_byte_ppl_x_to_en", "FLORES", "byte_perplexity x->en", "lower"),
    ("flores_byte_ppl_avg2dir", "FLORES", "byte_perplexity avg2dir", "lower"),
]

BENCHMARK_ORDER = {"Belebele": 0, "FLORES": 1}


def parse_csv_arg(arg: str) -> List[str]:
    return [item.strip() for item in arg.split(",") if item.strip()]


def parse_tasks(path: Path) -> List[str]:
    tasks: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    return tasks


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
                lang = tgt
                ensure_lang(lang)
                task_map[lang]["flo_en_to_x_task"] = task
            elif tgt == "eng_Latn":
                lang = src
                ensure_lang(lang)
                task_map[lang]["flo_x_to_en_task"] = task

        out[tier] = {"order": order, "tasks": task_map}
    return out


def get_metric(summary: dict, task: str | None, metric: str) -> float | None:
    if not task:
        return None
    val = summary.get(f"{task}/{metric}")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def choose_best(values: Dict[str, float | None], better: str) -> Tuple[str | None, float | None]:
    valid = [(k, v) for k, v in values.items() if isinstance(v, (int, float))]
    if not valid:
        return None, None
    if better == "higher":
        best = max(valid, key=lambda kv: kv[1])
    else:
        best = min(valid, key=lambda kv: kv[1])
    return best[0], float(best[1])


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
        m = RUN_NAME_PAT.search(run.name or "")
        if not m:
            continue
        model = m.group("model")
        tier = int(m.group("tier"))
        approach = APPROACH_MAP.get(m.group("approach"), m.group("approach"))
        if model not in wanted_models or tier not in wanted_tiers or approach not in wanted_approaches:
            continue
        key = (model, tier, approach)
        ts = _run_ts(run)
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, run)

    return {k: v[1] for k, v in latest.items()}


def build_model_sheet(
    model: str,
    tiers: List[int],
    approaches: List[str],
    task_map_by_tier: Dict[int, Dict[str, dict]],
    latest_runs: Dict[Tuple[str, int, str], wandb.apis.public.Run],
) -> pd.DataFrame:
    rows: List[dict] = []
    for tier in tiers:
        lang_order = task_map_by_tier[tier]["order"]
        tier_map = task_map_by_tier[tier]["tasks"]
        for lang in lang_order:
            tmap = tier_map[lang]
            for metric_name, benchmark, metric_label, better in METRIC_LAYOUT:
                values: Dict[str, float | None] = {}
                for approach in approaches:
                    run = latest_runs.get((model, tier, approach))
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
                    "better": better,
                }
                for approach in approaches:
                    row[approach] = values.get(approach)
                row["best_approach"] = best_approach
                row["best_value"] = best_value
                row["metric_key"] = metric_name
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["metric_rank"] = df["metric_key"].map(METRIC_ORDER).fillna(999)
    df["benchmark_rank"] = df["benchmark"].map(BENCHMARK_ORDER).fillna(999)
    df = df.sort_values(["tier", "benchmark_rank", "metric_rank", "language"]).drop(
        columns=["metric_rank", "benchmark_rank", "metric_key"]
    )
    return df


def build_summary_sheet(
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
                run = latest_runs.get((model, tier, approach))
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


def _write_sheet_banner(ws, title: str, description: str, end_col: int) -> None:
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


def _add_row_color_scale(ws, row_idx: int, better: str, first_col: int, last_col: int) -> None:
    has_num = any(isinstance(ws.cell(row_idx, c).value, (int, float)) for c in range(first_col, last_col + 1))
    if not has_num:
        return
    row_range = f"{get_column_letter(first_col)}{row_idx}:{get_column_letter(last_col)}{row_idx}"
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
    ws.conditional_formatting.add(row_range, rule)


def _write_tier_table(
    ws,
    tier_df: pd.DataFrame,
    tier: int,
    start_row: int,
    start_col: int,
    approaches: List[str],
    table_name: str,
) -> None:
    headers = ["benchmark", "metric", "language", "better", *approaches, "best_approach", "best_value"]
    table_width = len(headers)
    end_col = start_col + table_width - 1
    title_fill = PatternFill(fill_type="solid", start_color="305496", end_color="305496")
    subtitle_fill = PatternFill(fill_type="solid", start_color="E2ECF9", end_color="E2ECF9")
    header_fill = PatternFill(fill_type="solid", start_color="1F4E78", end_color="1F4E78")
    thin = Side(style="thin", color="B7C6D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws.merge_cells(start_row=start_row, start_column=start_col, end_row=start_row, end_column=end_col)
    ws.cell(start_row, start_col, f"{tier} Languages")
    ws.cell(start_row, start_col).font = Font(color="FFFFFF", bold=True, size=12)
    ws.cell(start_row, start_col).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(start_row, start_col).fill = title_fill

    ws.merge_cells(start_row=start_row + 1, start_column=start_col, end_row=start_row + 1, end_column=end_col)
    ws.cell(start_row + 1, start_col, "Grouped by benchmark and metric; compare approaches row-wise.")
    ws.cell(start_row + 1, start_col).font = Font(color="1F2937", italic=True, size=9)
    ws.cell(start_row + 1, start_col).alignment = Alignment(horizontal="center", vertical="center")
    ws.cell(start_row + 1, start_col).fill = subtitle_fill

    def write_subtable(bench_name: str, bench_df: pd.DataFrame, row_cursor: int, table_suffix: str) -> int:
        ws.merge_cells(start_row=row_cursor, start_column=start_col, end_row=row_cursor, end_column=end_col)
        ws.cell(row_cursor, start_col, f"{bench_name} Block")
        ws.cell(row_cursor, start_col).font = Font(color="FFFFFF", bold=True, size=10)
        ws.cell(row_cursor, start_col).alignment = Alignment(horizontal="left", vertical="center")
        ws.cell(row_cursor, start_col).fill = PatternFill(fill_type="solid", start_color="4F81BD", end_color="4F81BD")

        header_row = row_cursor + 1
        for idx, h in enumerate(headers):
            c = start_col + idx
            ws.cell(header_row, c, h)
            ws.cell(header_row, c).font = Font(color="FFFFFF", bold=True)
            ws.cell(header_row, c).alignment = Alignment(horizontal="center", vertical="center")
            ws.cell(header_row, c).fill = header_fill
            ws.cell(header_row, c).border = border

        body_cols = ["benchmark", "metric", "language", "better", *approaches, "best_approach", "best_value"]
        body_df = bench_df[body_cols].copy() if not bench_df.empty else pd.DataFrame(columns=body_cols)
        if body_df.empty:
            body_df = pd.DataFrame([{"benchmark": bench_name, "metric": "no_data", "language": None, "better": None}])

        data_start = header_row + 1
        for r_off, (_, row) in enumerate(body_df.iterrows()):
            r = data_start + r_off
            for idx, h in enumerate(headers):
                c = start_col + idx
                val = row[h] if h in row else None
                ws.cell(r, c, val)
                ws.cell(r, c).border = border
                if h in {"benchmark", "metric", "language", "better", "best_approach"}:
                    ws.cell(r, c).alignment = Alignment(horizontal="left", vertical="center")
                else:
                    ws.cell(r, c).alignment = Alignment(horizontal="center", vertical="center")
                    if isinstance(val, (int, float)):
                        ws.cell(r, c).number_format = "0.0000"

        last_row = data_start + len(body_df) - 1
        first_ap_col = start_col + 4
        last_ap_col = first_ap_col + len(approaches) - 1
        better_col = start_col + 3
        for r in range(data_start, last_row + 1):
            better = ws.cell(r, better_col).value
            if better in {"higher", "lower"}:
                _add_row_color_scale(ws, r, better, first_ap_col, last_ap_col)

        table_ref = f"{get_column_letter(start_col)}{header_row}:{get_column_letter(end_col)}{last_row}"
        tab = Table(displayName=f"{table_name}_{table_suffix}", ref=table_ref)
        tab.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        ws.add_table(tab)
        return last_row + 2

    row_cursor = start_row + 2
    bele_df = tier_df[tier_df["benchmark"] == "Belebele"].copy()
    flo_df = tier_df[tier_df["benchmark"] == "FLORES"].copy()
    row_cursor = write_subtable("Belebele", bele_df, row_cursor, "B")
    _ = write_subtable("FLORES", flo_df, row_cursor, "F")

    widths = [12, 25, 14, 9] + [11 for _ in approaches] + [14, 11]
    for idx, width in enumerate(widths):
        _set_col_width(ws, start_col + idx, width)


def write_model_sheet(ws, model: str, model_df: pd.DataFrame, tiers: List[int], approaches: List[str]) -> None:
    table_width = 4 + len(approaches) + 2
    gap = 3
    tier_order = sorted(tiers, reverse=True)
    start_row = 5
    start_cols = {tier: 1 + i * (table_width + gap) for i, tier in enumerate(tier_order)}
    last_used_col = max(start_cols.values()) + table_width - 1
    _write_sheet_banner(
        ws,
        f"Model {model} - Tiered Eval Comparison",
        "Three side-by-side tables by language tier. Within each table, rows are grouped by benchmark and metric.",
        last_used_col,
    )

    for idx, tier in enumerate(tier_order):
        tier_df = model_df[model_df["tier"] == tier].copy()
        _write_tier_table(
            ws=ws,
            tier_df=tier_df,
            tier=tier,
            start_row=start_row,
            start_col=start_cols[tier],
            approaches=approaches,
            table_name=f"T_{model}_{tier}_{idx}",
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
    _write_sheet_banner(
        ws,
        "Summary - Tier-Level Averages",
        "Quick comparison of averages and run links. Higher is better for Belebele, lower is better for FLORES byte perplexity.",
        end_col=len(cols),
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
    tab_ref = f"A{header_row}:{get_column_letter(len(cols))}{last_row}"
    tab = Table(displayName="T_Summary", ref=tab_ref)
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
    for idx, width in enumerate([10, 7, 11, 20, 24, 12, 12, 48, 12, 50], start=1):
        _set_col_width(ws, idx, width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export W&B eval results to a compact Excel workbook.")
    parser.add_argument("--entity", default="dice-nlp")
    parser.add_argument("--project", default="moe-study-comparison_eval")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--models", default="1B,3B,8B")
    parser.add_argument("--tiers", default="10,96,200")
    parser.add_argument("--approaches", default="adamole,mola,cola,hydralora")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_csv_arg(args.models)
    tiers = [int(x) for x in parse_csv_arg(args.tiers)]
    approaches = parse_csv_arg(args.approaches)
    out_path = Path(args.out)

    task_map_by_tier = parse_tier_task_map(tiers)
    api = wandb.Api()
    latest_runs = select_latest_runs(api, args.entity, args.project, models, tiers, approaches)

    expected = {(m, t, a) for m in models for t in tiers for a in approaches}
    missing = sorted(expected - set(latest_runs.keys()))
    if missing:
        print(f"[WARN] Missing latest runs for {len(missing)} combinations:")
        for m, t, a in missing:
            print(f"  - model={m} tier={t} approach={a}")

    model_frames = {
        model: build_model_sheet(model, tiers, approaches, task_map_by_tier, latest_runs)
        for model in models
    }
    summary_df = build_summary_sheet(models, tiers, approaches, task_map_by_tier, latest_runs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)

    for model in models:
        ws = wb.create_sheet(title=model)
        write_model_sheet(ws, model, model_frames[model], tiers=tiers, approaches=approaches)

    # Keep summary as the final tab.
    summary_ws = wb.create_sheet(title="summary")
    write_summary_sheet(summary_ws, summary_df)

    for ws in wb.worksheets:
        ws.sheet_properties.tabColor = "1F4E78" if ws.title != "summary" else "203864"
    wb.save(out_path)

    print(f"Wrote {out_path}")
    print("Sheets:", ", ".join(models + ["summary"]))


if __name__ == "__main__":
    main()
