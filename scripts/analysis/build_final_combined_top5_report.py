#!/usr/bin/env python3
"""Validate and build one canonical five-run final-study workbook."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline_tools.reporting.sources import (
    iter_eval_rows,
    load_baseline_rows,
    load_baseline_rows_from_csv,
    load_baseline_rows_from_lm_eval_json,
    load_resource_map,
    resolve_baseline_lm_eval_path,
)
from pipeline_tools.reporting.workbook import (
    _merge_baseline_rows_with_precedence,
    build_workbook,
)


V1 = {
    "macro3_96": "outputs/pipeline_runs/moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1",
    "macro3_200": "outputs/pipeline_runs/moe_study_final_macro3_noctua2_a100_4gpu_200_langs_72h_v1",
    "macro45_96": "outputs/pipeline_runs/moe_study_final_macro45_noctua2_a100_4gpu_96_langs_34p5h_v1",
    "macro45_200": "outputs/pipeline_runs/moe_study_final_macro45_noctua2_a100_4gpu_200_langs_72h_v1",
    "mtllora5e5_96": "outputs/pipeline_runs/moe_study_final_mtllora_lr5e5_noctua2_a100_4gpu_96_langs_34p5h_v1",
    "mtllora5e5_200": "outputs/pipeline_runs/moe_study_final_mtllora_lr5e5_noctua2_a100_4gpu_200_langs_72h_v1",
}
V2 = "outputs/pipeline_runs/moe_study_final_benchmark_panel_v2_recovery_20260724"
ENGLISH = "outputs/pipeline_runs/moe_study_final_english_eval_20260725"

ENGLISH_TASKS = {
    "v1": {"belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_eng_Latn"},
    "v2": {"belebele_eng_Latn", "sib200_eng_Latn", "taxi1500_v2_eng_Latn"},
}
ALL_ENGLISH_TASKS = set().union(*ENGLISH_TASKS.values())

EXPECTED_RUNS = {
    "macro3": {"moelora_tr32_e16_k8", "hydralora_r2"},
    "macro45": {"moelora_tr64_e32_k8", "mola_tr16_e8_k6"},
    "mtllora5e5": {"mtllora_lr5e5_r8_nup16"},
}
ALL_RUNS = set().union(*EXPECTED_RUNS.values())

PANEL_SPECS = {
    "v1_96": {
        "sources": {
            "macro3": V1["macro3_96"],
            "macro45": V1["macro45_96"],
            "mtllora5e5": V1["mtllora5e5_96"],
        },
        "train_sources": {
            "macro3": V1["macro3_96"],
            "macro45": V1["macro45_96"],
            "mtllora5e5": V1["mtllora5e5_96"],
        },
        "source_tasks": 183,
        "source_breakdown": {"Belebele": 34, "FLORES": 92, "SIB200": 34, "Taxi1500": 23},
        "tasks": 186,
        "breakdown": {"Belebele": 35, "FLORES": 92, "SIB200": 35, "Taxi1500": 24},
        "english_panel": "v1",
        "english_size": "96",
    },
    "v1_200": {
        "sources": {
            "macro3": V1["macro3_200"],
            "macro45": V1["macro45_200"],
            "mtllora5e5": V1["mtllora5e5_200"],
        },
        "train_sources": {
            "macro3": V1["macro3_200"],
            "macro45": V1["macro45_200"],
            "mtllora5e5": V1["mtllora5e5_200"],
        },
        "source_tasks": 306,
        "source_breakdown": {"Belebele": 57, "FLORES": 156, "SIB200": 57, "Taxi1500": 36},
        "tasks": 309,
        "breakdown": {"Belebele": 58, "FLORES": 156, "SIB200": 58, "Taxi1500": 37},
        "english_panel": "v1",
        "english_size": "200",
    },
    "v2_96": {
        "sources": {
            "macro3": f"{V2}/macro3_96",
            "macro45": f"{V2}/macro45_96",
            "mtllora5e5": f"{V2}/mtllora5e5_96",
        },
        "train_sources": {
            "macro3": V1["macro3_96"],
            "macro45": V1["macro45_96"],
            "mtllora5e5": V1["mtllora5e5_96"],
        },
        "source_tasks": 246,
        "source_breakdown": {"Belebele": 39, "FLORES": 100, "SIB200": 50, "Taxi1500": 57},
        "tasks": 249,
        "breakdown": {"Belebele": 40, "FLORES": 100, "SIB200": 51, "Taxi1500": 58},
        "english_panel": "v2",
        "english_size": "96",
    },
    "v2_200": {
        "sources": {
            "macro3": f"{V2}/macro3_200",
            "macro45": f"{V2}/macro45_200",
            "mtllora5e5": f"{V2}/mtllora5e5_200",
        },
        "train_sources": {
            "macro3": V1["macro3_200"],
            "macro45": V1["macro45_200"],
            "mtllora5e5": V1["mtllora5e5_200"],
        },
        "source_tasks": 405,
        "source_breakdown": {"Belebele": 64, "FLORES": 166, "SIB200": 83, "Taxi1500": 92},
        "tasks": 408,
        "breakdown": {"Belebele": 65, "FLORES": 166, "SIB200": 84, "Taxi1500": 93},
        "english_panel": "v2",
        "english_size": "200",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing required CSV: {path}")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_tasks(root: Path, run_id: str) -> set[str]:
    candidates = [
        root / "benchmarks" / f"selected_tasks.{run_id}.txt",
        root / "benchmarks" / "selected_tasks.txt",
    ]
    for path in candidates:
        if path.is_file():
            return {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
    raise ValueError(f"No selected-task file for {run_id} below {root}")


def safe_link(source: Path, target: Path) -> None:
    if not source.exists():
        raise ValueError(f"Missing link source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=True)


def validate_panel(panel: str, groups: list[str] | None = None) -> dict[str, Any]:
    spec = PANEL_SPECS[panel]
    selected_groups = groups or list(EXPECTED_RUNS)
    unknown_groups = set(selected_groups) - set(EXPECTED_RUNS)
    if not selected_groups or unknown_groups:
        raise ValueError(f"Invalid source groups: {sorted(unknown_groups)}")
    expected_all_runs = set().union(*(EXPECTED_RUNS[group] for group in selected_groups))
    expected_tasks = int(spec["tasks"])
    expected_breakdown = dict(spec["breakdown"])
    source_tasks = int(spec["source_tasks"])
    source_breakdown = dict(spec["source_breakdown"])
    panel_english_tasks = ENGLISH_TASKS[str(spec["english_panel"])]
    summary_rows: list[dict[str, str]] = []
    benchmark_runs: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    task_sets: dict[str, set[str]] = {}

    for group in selected_groups:
        root_value = dict(spec["sources"])[group]
        root = ROOT / str(root_value)
        train_root = ROOT / str(dict(spec["train_sources"])[group])
        expected_group_runs = EXPECTED_RUNS[group]
        if not root.is_dir():
            raise ValueError(f"{panel}/{group}: source root is not ready: {root}")

        csv_rows = [row for row in read_csv(root / "summary.csv") if row.get("run_id") in expected_group_runs]
        csv_ids = [str(row.get("run_id", "")) for row in csv_rows]
        if len(csv_ids) != len(set(csv_ids)) or set(csv_ids) != expected_group_runs:
            raise ValueError(
                f"{panel}/{group}: summary.csv expected {sorted(expected_group_runs)}, got {sorted(csv_ids)}"
            )

        benchmark_summary = read_json(root / "benchmarks" / "benchmark_summary.json")
        if int(benchmark_summary.get("num_selected_tasks", 0)) != source_tasks:
            raise ValueError(
                f"{panel}/{group}: expected {source_tasks} source tasks, "
                f"got {benchmark_summary.get('num_selected_tasks')}"
            )
        if dict(benchmark_summary.get("expected_task_breakdown", {})) != source_breakdown:
            raise ValueError(
                f"{panel}/{group}: unexpected task breakdown "
                f"{benchmark_summary.get('expected_task_breakdown')}"
            )

        runs_by_id = {
            str(row.get("run_id", "")): row
            for row in benchmark_summary.get("runs", [])
            if isinstance(row, dict) and str(row.get("run_id", "")) in expected_group_runs
        }
        if set(runs_by_id) != expected_group_runs:
            raise ValueError(
                f"{panel}/{group}: benchmark summary expected {sorted(expected_group_runs)}, "
                f"got {sorted(runs_by_id)}"
            )

        eval_rows = [
            row
            for row in iter_eval_rows(root / "benchmarks")
            if str(row.get("run_id", "")) in expected_group_runs
        ]
        for run_id in sorted(expected_group_runs):
            run = runs_by_id[run_id]
            if (
                int(run.get("num_tasks_expected", 0)) != source_tasks
                or int(run.get("num_tasks_seen", 0)) != source_tasks
                or run.get("missing_tasks")
                or run.get("extra_tasks_seen")
                or not bool(run.get("strict_completeness"))
            ):
                raise ValueError(f"{panel}/{group}/{run_id}: incomplete benchmark summary: {run}")
            provenance = run.get("adapter_provenance", {})
            if isinstance(provenance, dict) and provenance and not bool(provenance.get("valid")):
                raise ValueError(f"{panel}/{group}/{run_id}: invalid adapter provenance")

            expected_set = selected_tasks(root, run_id)
            observed_rows = [row for row in eval_rows if str(row.get("run_id", "")) == run_id]
            observed_set = {str(row.get("task", "")) for row in observed_rows}
            if len(expected_set) != source_tasks or observed_set != expected_set:
                raise ValueError(
                    f"{panel}/{group}/{run_id}: raw task mismatch "
                    f"(selected={len(expected_set)}, observed={len(observed_set)}, "
                    f"missing={sorted(expected_set - observed_set)[:5]}, "
                    f"extra={sorted(observed_set - expected_set)[:5]})"
                )
            if any(row.get("value") is None for row in observed_rows):
                raise ValueError(f"{panel}/{group}/{run_id}: non-finite benchmark value")
            if not (train_root / "runs" / run_id).is_dir():
                raise ValueError(f"{panel}/{group}/{run_id}: training artifacts are missing")

            task_sets[run_id] = expected_set
            augmented_run = dict(run)
            augmented_run.update(
                {
                    "num_tasks_expected": expected_tasks,
                    "num_tasks_seen": expected_tasks,
                    "expected_task_breakdown": expected_breakdown,
                    "seen_task_breakdown": expected_breakdown,
                    "english_evaluation_root": str(
                        (ROOT / ENGLISH / str(spec["english_size"]) / "benchmarks" / run_id).resolve()
                    ),
                }
            )
            benchmark_runs.append(augmented_run)

        summary_rows.extend(csv_rows)
        source_records.append(
            {
                "group": group,
                "benchmark_root": str(root.resolve()),
                "training_root": str(train_root.resolve()),
                "runs": sorted(expected_group_runs),
            }
        )

    if (
        {str(row.get("run_id", "")) for row in summary_rows} != expected_all_runs
        or len(summary_rows) != len(expected_all_runs)
    ):
        raise ValueError(
            f"{panel}: combined summary must contain exactly the expected runs "
            f"{sorted(expected_all_runs)}"
        )

    english_root = ROOT / ENGLISH / str(spec["english_size"]) / "benchmarks"
    english_rows_all = iter_eval_rows(english_root)
    english_rows: list[dict[str, Any]] = []
    for run_id in sorted(expected_all_runs):
        run_rows = [
            row for row in english_rows_all
            if str(row.get("run_id", "")) == run_id
        ]
        observed_english = {str(row.get("task", "")) for row in run_rows}
        if observed_english != ALL_ENGLISH_TASKS:
            raise ValueError(
                f"{panel}/{run_id}: English task mismatch "
                f"(missing={sorted(ALL_ENGLISH_TASKS-observed_english)}, "
                f"extra={sorted(observed_english-ALL_ENGLISH_TASKS)})"
            )
        if any(row.get("value") is None for row in run_rows):
            raise ValueError(f"{panel}/{run_id}: non-finite English benchmark value")
        english_rows.extend(
            row for row in run_rows if str(row.get("task", "")) in panel_english_tasks
        )
        task_sets[run_id].update(panel_english_tasks)

    source_records.append(
        {
            "group": "english_extension",
            "benchmark_root": str(english_root.resolve()),
            "runs": sorted(expected_all_runs),
            "tasks_used": sorted(panel_english_tasks),
        }
    )
    reference_tasks = next(iter(task_sets.values()))
    if len(reference_tasks) != expected_tasks or any(
        tasks != reference_tasks for tasks in task_sets.values()
    ):
        raise ValueError(f"{panel}: the five configurations do not share the same task panel")

    return {
        "panel": panel,
        "status": "validated",
        "groups": selected_groups,
        "num_runs": len(expected_all_runs),
        "num_tasks_per_run": expected_tasks,
        "expected_task_breakdown": expected_breakdown,
        "runs": sorted(expected_all_runs),
        "summary_rows": summary_rows,
        "benchmark_runs": benchmark_runs,
        "english_rows": english_rows,
        "sources": source_records,
    }


def baseline_rows() -> list[dict[str, Any]]:
    model = "meta-llama/Llama-3.1-8B"
    baseline_json = resolve_baseline_lm_eval_path(ROOT / "outputs/baselines/llama31_8b_full_eval")
    baseline_csvs = [
        ROOT / "configs/baselines/llama31_8b_full_eval/l-3-1-8b_full_baseline_metrics_summary.csv",
        ROOT / "configs/baselines/llama31_8b_sib_taxi_eval/l-3-1-8b_sib_taxi_baseline_metrics_summary.csv",
    ]
    return _merge_baseline_rows_with_precedence(
        (0, load_baseline_rows(ROOT / "outputs/baselines", model)),
        (1, load_baseline_rows_from_lm_eval_json(baseline_json, model) if baseline_json else []),
        (2, [row for path in baseline_csvs for row in load_baseline_rows_from_csv(path)]),
    )


def build(
    panel: str,
    validation: dict[str, Any],
    output_root: Path,
    overwrite: bool,
    interim: bool,
) -> Path:
    if output_root.exists() and not overwrite:
        raise ValueError(f"Output already exists; pass --overwrite to replace safely: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "runs").mkdir(parents=True)
    (staging / "benchmarks").mkdir()

    spec = PANEL_SPECS[panel]
    for group in validation["groups"]:
        run_ids = EXPECTED_RUNS[group]
        benchmark_root = ROOT / str(dict(spec["sources"])[group])
        train_root = ROOT / str(dict(spec["train_sources"])[group])
        for run_id in run_ids:
            safe_link(train_root / "runs" / run_id, staging / "runs" / run_id)
            safe_link(benchmark_root / "benchmarks" / run_id, staging / "benchmarks" / run_id)
    safe_link(
        ROOT / ENGLISH / str(spec["english_size"]) / "benchmarks",
        staging / "english_benchmarks",
    )

    summary_rows = list(validation["summary_rows"])
    fieldnames: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (staging / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    benchmark_summary = {
        "report_type": f"combined_top5_{panel}",
        "source_reports": validation["sources"],
        "runs": validation["benchmark_runs"],
        "num_selected_tasks": validation["num_tasks_per_run"],
        "expected_task_breakdown": validation["expected_task_breakdown"],
        "strict_completeness": True,
    }
    benchmark_summary_path = staging / "benchmarks" / "benchmark_summary.json"
    benchmark_summary_path.write_text(json.dumps(benchmark_summary, indent=2) + "\n", encoding="utf-8")

    macro3_source = ROOT / str(dict(spec["sources"])["macro3"])
    macro3_train = ROOT / str(dict(spec["train_sources"])["macro3"])
    profile_path = macro3_source / "state" / "resolved_profile.json"
    profile = read_json(profile_path) if profile_path.is_file() else {}
    profile.update(
        {
            "report_type": f"combined_top5_{panel}",
            "report_status": (
                "INTERIM: benchmark scores passed strict coverage validation; final-loss jobs are pending"
                if interim
                else "FINAL: five configurations passed strict benchmark coverage validation"
            ),
            "included_runs": validation["runs"],
            "sources": validation["sources"],
        }
    )
    profile_path_out = staging / "combined_profile.json"
    profile_path_out.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    eval_rows = [*iter_eval_rows(staging / "benchmarks"), *validation["english_rows"]]
    for run_id in validation["runs"]:
        if len({str(row.get("task", "")) for row in eval_rows if row.get("run_id") == run_id}) != int(
            validation["num_tasks_per_run"]
        ):
            raise ValueError(f"{panel}/{run_id}: linked raw benchmark coverage changed during assembly")

    workbook_name = f"moe_study_final_combined_top5_{panel}{'_interim' if interim else ''}.xlsx"
    workbook_path = staging / workbook_name
    build_workbook(
        summary_rows=summary_rows,
        eval_rows=eval_rows,
        baseline_rows=baseline_rows(),
        resource_map=load_resource_map(
            ROOT / "result_analysis/l2v_nested_samples/fineweb2-language-distribution.csv"
        ),
        profile_payload=profile,
        runs_root=staging / "runs",
        sample_manifest_path=macro3_train / "sampled" / "manifest.json",
        tokenized_dir=macro3_train / "tokenized",
        benchmark_summary_path=benchmark_summary_path,
        repo_root=ROOT,
        profile_name=f"moe_study_final_combined_top5_{panel}",
        logs_root=None,
        out_path=workbook_path,
    )
    with zipfile.ZipFile(workbook_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None or "xl/workbook.xml" not in archive.namelist():
            raise ValueError(f"Invalid workbook archive: {corrupt_member or 'missing workbook.xml'}")

    digest = hashlib.sha256(workbook_path.read_bytes()).hexdigest()
    manifest = {
        key: value
        for key, value in validation.items()
        if key not in {"summary_rows", "benchmark_runs", "english_rows"}
    }
    manifest.update(
        {
            "status": "interim" if interim else "complete",
            "workbook": str((output_root / workbook_name).resolve()),
            "workbook_sha256": digest,
        }
    )
    (staging / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if output_root.exists():
        backup = output_root.with_name(
            f"{output_root.name}.backup.{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )
        output_root.rename(backup)
    staging.rename(output_root)
    return output_root / workbook_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, choices=sorted(PANEL_SPECS))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--interim", action="store_true")
    parser.add_argument(
        "--groups",
        help="Comma-separated source groups for an interim workbook (macro3,macro45,mtllora5e5)",
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    groups = [value.strip() for value in args.groups.split(",") if value.strip()] if args.groups else None
    if groups and not args.interim and not args.validate_only:
        raise ValueError("--groups is only allowed with --interim or --validate-only")
    validation = validate_panel(args.panel, groups)
    printable = {
        key: value
        for key, value in validation.items()
        if key not in {"summary_rows", "benchmark_runs", "english_rows"}
    }
    if args.validate_only:
        print(json.dumps(printable, indent=2))
        return

    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / f"outputs/pipeline_runs/moe_study_final_combined_top5_{args.panel}"
    )
    workbook = build(args.panel, validation, output_root, args.overwrite, args.interim)
    print(
        json.dumps(
            {
                **printable,
                "status": "interim" if args.interim else "complete",
                "workbook": str(workbook),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
