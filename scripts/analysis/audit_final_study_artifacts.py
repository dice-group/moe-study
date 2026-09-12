#!/usr/bin/env python3
"""Fail-closed artifact audit for the final multilingual MoE study.

The audit proves benchmark completeness from ``benchmark_summary.json`` and
report integrity from the XLSX archive.  It deliberately treats the original
MTL-LoRA overflow as a terminal divergent run, not as a missing score.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CONTRACTS = {
    "v1_macro3_96": ("outputs/pipeline_runs/moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1", 183, {"moelora_tr32_e16_k8", "hydralora_r2"}),
    "v1_macro3_200": ("outputs/pipeline_runs/moe_study_final_macro3_noctua2_a100_4gpu_200_langs_72h_v1", 306, {"moelora_tr32_e16_k8", "hydralora_r2"}),
    "v1_macro45_96": ("outputs/pipeline_runs/moe_study_final_macro45_noctua2_a100_4gpu_96_langs_34p5h_v1", 183, {"moelora_tr64_e32_k8", "mola_tr16_e8_k6"}),
    "v1_macro45_200": ("outputs/pipeline_runs/moe_study_final_macro45_noctua2_a100_4gpu_200_langs_72h_v1", 306, {"moelora_tr64_e32_k8", "mola_tr16_e8_k6"}),
    "v1_mtllora5e5_96": ("outputs/pipeline_runs/moe_study_final_mtllora_lr5e5_noctua2_a100_4gpu_96_langs_34p5h_v1", 183, {"mtllora_lr5e5_r8_nup16"}),
    "v1_mtllora5e5_200": ("outputs/pipeline_runs/moe_study_final_mtllora_lr5e5_noctua2_a100_4gpu_200_langs_72h_v1", 306, {"mtllora_lr5e5_r8_nup16"}),
    "v2_macro3_96": ("outputs/pipeline_runs/moe_study_final_benchmark_panel_v2_recovery_20260724/macro3_96", 246, {"moelora_tr32_e16_k8", "hydralora_r2"}),
    "v2_macro3_200": ("outputs/pipeline_runs/moe_study_final_benchmark_panel_v2_recovery_20260724/macro3_200", 405, {"moelora_tr32_e16_k8", "hydralora_r2"}),
    "v2_macro45_96": ("outputs/pipeline_runs/moe_study_final_benchmark_panel_v2_recovery_20260724/macro45_96", 246, {"moelora_tr64_e32_k8", "mola_tr16_e8_k6"}),
    "v2_macro45_200": ("outputs/pipeline_runs/moe_study_final_benchmark_panel_v2_recovery_20260724/macro45_200", 405, {"moelora_tr64_e32_k8", "mola_tr16_e8_k6"}),
    "v2_mtllora5e5_96": ("outputs/pipeline_runs/moe_study_final_benchmark_panel_v2_recovery_20260724/mtllora5e5_96", 246, {"mtllora_lr5e5_r8_nup16"}),
    "v2_mtllora5e5_200": ("outputs/pipeline_runs/moe_study_final_benchmark_panel_v2_recovery_20260724/mtllora5e5_200", 405, {"mtllora_lr5e5_r8_nup16"}),
}


def inspect(name: str, root_rel: str, expected_tasks: int, expected_runs: set[str]) -> dict:
    root = ROOT / root_rel
    summary_path = root / "benchmarks" / "benchmark_summary.json"
    reports = sorted(root.glob("*.xlsx"))
    result = {"name": name, "root": root_rel, "status": "pending", "errors": []}
    if not summary_path.is_file():
        result["errors"].append("missing benchmark_summary.json")
        return result
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["errors"].append(f"invalid benchmark summary: {exc}")
        return result
    result["task_count"] = summary.get("num_selected_tasks")
    runs = {str(row.get("run_id")): row for row in summary.get("runs", []) if isinstance(row, dict)}
    if summary.get("num_selected_tasks") != expected_tasks:
        result["errors"].append(f"task count={summary.get('num_selected_tasks')}, expected={expected_tasks}")
    if set(runs) != expected_runs:
        result["errors"].append(f"runs={sorted(runs)}, expected={sorted(expected_runs)}")
    for run_id, row in runs.items():
        if row.get("num_tasks_seen") != expected_tasks or row.get("num_tasks_expected") != expected_tasks:
            result["errors"].append(f"{run_id}: incomplete task count")
        if row.get("missing_tasks") or row.get("extra_tasks_seen") or not row.get("strict_completeness"):
            result["errors"].append(f"{run_id}: non-strict benchmark completeness")
        if not bool(row.get("adapter_provenance", {}).get("valid")):
            result["errors"].append(f"{run_id}: invalid adapter provenance")
    if not reports:
        result["errors"].append("missing report workbook")
        return result
    report = reports[-1]
    result["report"] = str(report.relative_to(ROOT))
    try:
        with zipfile.ZipFile(report) as archive:
            if archive.testzip() is not None:
                result["errors"].append("corrupt XLSX member")
            if "xl/workbook.xml" not in archive.namelist():
                result["errors"].append("XLSX has no workbook.xml")
    except Exception as exc:
        result["errors"].append(f"invalid XLSX: {exc}")
    result["status"] = "complete" if not result["errors"] else "incomplete"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="Return zero while printing pending/incomplete contracts.")
    args = parser.parse_args()
    rows = [inspect(name, *contract) for name, contract in CONTRACTS.items()]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0 if args.allow_incomplete or all(row["status"] == "complete" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
