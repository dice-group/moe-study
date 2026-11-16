#!/usr/bin/env python3
"""Validate and submit the final English benchmark-extension DAG."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/pipeline_runs/moe_study_final_english_eval_20260725"
TASKS_FILE = ROOT / "configs/benchmark_panels/english/final_english_tasks.txt"
BAD_GPU_NODES = "n2gpu1212,n2gpu1217,n2gpu1223,n2gpu1227"
BAD_CPU_NODES = "n2cn1086"
RUN_SOURCES = {
    "96": {
        "moelora_tr32_e16_k8": "moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1",
        "hydralora_r2": "moe_study_final_macro3_noctua2_a100_4gpu_96_langs_34p5h_v1",
        "moelora_tr64_e32_k8": "moe_study_final_macro45_noctua2_a100_4gpu_96_langs_34p5h_v1",
        "mola_tr16_e8_k6": "moe_study_final_macro45_noctua2_a100_4gpu_96_langs_34p5h_v1",
        "mtllora_lr5e5_r8_nup16": "moe_study_final_mtllora_lr5e5_noctua2_a100_4gpu_96_langs_34p5h_v1",
    },
    "200": {
        "moelora_tr32_e16_k8": "moe_study_final_macro3_noctua2_a100_4gpu_200_langs_72h_v1",
        "hydralora_r2": "moe_study_final_macro3_noctua2_a100_4gpu_200_langs_72h_v1",
        "moelora_tr64_e32_k8": "moe_study_final_macro45_noctua2_a100_4gpu_200_langs_72h_v1",
        "mola_tr16_e8_k6": "moe_study_final_macro45_noctua2_a100_4gpu_200_langs_72h_v1",
        "mtllora_lr5e5_r8_nup16": "moe_study_final_mtllora_lr5e5_noctua2_a100_4gpu_200_langs_72h_v1",
    },
}


def adapter_checkpoint(size: str, run_id: str, source: str) -> Path:
    run_root = ROOT / "outputs/pipeline_runs" / source / "runs" / run_id
    candidates = [
        path for path in run_root.glob("checkpoint-*_adapter")
        if (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
        and (path / ".adapter_direct_save_complete").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{size}/{run_id}: expected one complete adapter checkpoint, got {len(candidates)}"
        )
    return candidates[0].resolve()


def static_validation() -> dict[str, dict[str, str]]:
    tasks = {
        line.strip()
        for line in TASKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    expected = {
        "belebele_eng_Latn",
        "sib200_eng_Latn",
        "taxi1500_eng_Latn",
        "taxi1500_v2_eng_Latn",
    }
    if tasks != expected:
        raise ValueError(f"English task file mismatch: {sorted(tasks)}")
    task_root = (
        ROOT
        / "scripts/eval/latest_checkpoint/lm-evaluation-harness/lm_eval/tasks"
    )
    configs = [
        task_root / "belebele/belebele_eng_Latn.yaml",
        task_root / "sib200/configs/sib200_eng_Latn.yaml",
        task_root / "taxi1500/configs/taxi1500_eng_Latn.yaml",
        task_root / "taxi1500/configs/taxi1500_v2_eng_Latn.yaml",
    ]
    missing = [str(path) for path in configs if not path.is_file()]
    if missing:
        raise ValueError(f"Missing English task configs: {missing}")

    checkpoints: dict[str, dict[str, str]] = {}
    for size, sources in RUN_SOURCES.items():
        checkpoints[size] = {
            run_id: str(adapter_checkpoint(size, run_id, source))
            for run_id, source in sources.items()
        }
    return checkpoints


def submit_job(args: list[str]) -> str:
    result = subprocess.run(
        ["sbatch", "--parsable", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError(f"Unexpected sbatch output: {result.stdout!r}")
    return job_id


def write_manifest(payload: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "submission_manifest.json"
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def submit(checkpoints: dict[str, dict[str, str]]) -> None:
    manifest_path = OUTPUT / "submission_manifest.json"
    if manifest_path.exists():
        raise ValueError(f"Submission manifest already exists: {manifest_path}")
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "purpose": "Add standalone English Belebele, SIB200, and Taxi1500 evaluations",
        "tasks_file": str(TASKS_FILE.resolve()),
        "checkpoints": checkpoints,
        "bad_gpu_nodes_excluded": BAD_GPU_NODES.split(","),
        "jobs": jobs,
    }
    write_manifest(manifest)

    eval_jobs: list[str] = []
    for size in ("96", "200"):
        for run_id, checkpoint in checkpoints[size].items():
            output_dir = OUTPUT / size / "benchmarks" / run_id
            job_id = submit_job(
                [
                    f"--job-name=eng{size}_{run_id[:16]}",
                    f"--exclude={BAD_GPU_NODES}",
                    f"--output={logs}/english_{size}_{run_id}.%j.log",
                    (
                        "--export=ALL,"
                        f"RUN_ID={run_id},CHECKPOINT={checkpoint},"
                        f"TASKS_FILE={TASKS_FILE.resolve()},OUTPUT_DIR={output_dir}"
                    ),
                    "scripts/slurm/run_final_english_checkpoint_20260725.sh",
                ]
            )
            eval_jobs.append(job_id)
            jobs.append(
                {
                    "job_id": job_id,
                    "stage": "adapter_evaluation",
                    "size": int(size),
                    "run_id": run_id,
                    "checkpoint": checkpoint,
                    "output_dir": str(output_dir),
                }
            )
            write_manifest(manifest)

    baseline_dir = OUTPUT / "baseline_raw"
    baseline_job = submit_job(
        [
            "--job-name=eng_baseline",
            f"--exclude={BAD_GPU_NODES}",
            f"--output={logs}/english_baseline.%j.log",
            (
                "--export=ALL,"
                f"TASKS_FILE={TASKS_FILE.resolve()},SHARD=english,OUTPUT_DIR={baseline_dir}"
            ),
            "scripts/slurm/run_missing_v2_baseline_shard.sh",
        ]
    )
    eval_jobs.append(baseline_job)
    jobs.append(
        {
            "job_id": baseline_job,
            "stage": "baseline_evaluation",
            "output_dir": str(baseline_dir),
        }
    )
    write_manifest(manifest)

    consolidate = submit_job(
        [
            "--job-name=eng_consolidate",
            f"--exclude={BAD_CPU_NODES}",
            f"--dependency=afterok:{':'.join(eval_jobs)}",
            f"--output={logs}/english_consolidate.%j.log",
            "scripts/slurm/consolidate_final_english_evaluations.sh",
        ]
    )
    jobs.append(
        {
            "job_id": consolidate,
            "stage": "consolidate",
            "dependency": f"afterok:{':'.join(eval_jobs)}",
        }
    )
    write_manifest(manifest)

    report_jobs: list[str] = []
    for panel in ("v1_96", "v1_200", "v2_96", "v2_200"):
        job_id = submit_job(
            [
                f"--job-name=eng_report_{panel}",
                f"--exclude={BAD_CPU_NODES}",
                f"--dependency=afterok:{consolidate}",
                f"--output={logs}/english_report_{panel}.%j.log",
                f"--export=ALL,PANEL={panel},GROUPS=",
                "scripts/slurm/build_final_combined_top5_report.sh",
            ]
        )
        report_jobs.append(job_id)
        jobs.append(
            {
                "job_id": job_id,
                "stage": "report",
                "panel": panel,
                "dependency": f"afterok:{consolidate}",
            }
        )
        write_manifest(manifest)

    validation = submit_job(
        [
            "--job-name=eng_validate",
            f"--exclude={BAD_CPU_NODES}",
            f"--dependency=afterok:{':'.join(report_jobs)}",
            f"--output={logs}/english_validate.%j.log",
            "scripts/slurm/validate_final_english_workbooks.sh",
        ]
    )
    jobs.append(
        {
            "job_id": validation,
            "stage": "validate_workbooks",
            "dependency": f"afterok:{':'.join(report_jobs)}",
        }
    )
    manifest["terminal_job_id"] = validation
    write_manifest(manifest)
    print(json.dumps({"submitted_jobs": len(jobs), "terminal_job_id": validation}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.validate_only == args.submit:
        parser.error("choose exactly one of --validate-only or --submit")
    checkpoints = static_validation()
    if args.validate_only:
        print(json.dumps({"status": "validated", "checkpoints": checkpoints}, indent=2))
    else:
        submit(checkpoints)


if __name__ == "__main__":
    main()
