#!/usr/bin/env python3
"""Validate and submit the complete final LoRA 96/200-language job graph."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/scratch/hpc-prf-merlin/joel/conda/envs/merlin/bin/python")
CAMPAIGN_ROOT = ROOT / "outputs/pipeline_runs/moe_study_final_lora_r2_r4_shared"
RUN_IDS = ("lora_r2", "lora_r4")
SHARDS = ("belebele", "flores_en_x", "flores_x_en", "sib200", "taxi1500")
BAD_CPU_NODES = "n2cn0167,n2cn1086"
PROFILES = {
    "moe_study_final_lora_r2_r4_noctua2_a100_4gpu_96_langs_34p5h_v1": {
        "size": 96,
        "panel": "v2_96_plus_english",
    },
    "moe_study_final_lora_r2_r4_noctua2_a100_4gpu_200_langs_72h_v1": {
        "size": 200,
        "panel": "v2_200_plus_english",
    },
}

sys.path.insert(0, str(ROOT))
from pipeline.config_loader import load_pipeline_config  # noqa: E402
from pipeline.runtime import _resolve_profile  # noqa: E402


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def submit_sbatch(args: list[str], *, dry_run: bool, label: str) -> str:
    command = ["sbatch", "--parsable", *args]
    if dry_run:
        job_id = f"dry_{re.sub(r'[^A-Za-z0-9_]+', '_', label)}"
        print(f"[DRY_RUN SBATCH] {shlex.join(command)}")
        return job_id
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError(f"Unexpected sbatch output: {result.stdout!r}")
    return job_id


def pipeline_chain(profile_name: str, *, dry_run: bool) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_profile(load_pipeline_config(ROOT / "configs/pipeline"), profile_name)
    output_root = ROOT / str(resolved["output_root"])
    graph_root = output_root / "logs/chain"
    before = set(graph_root.glob("job_graph_*.json")) if graph_root.is_dir() else set()
    command = [
        str(PYTHON),
        "-m",
        "pipeline.run",
        "--config",
        "configs/pipeline",
        "--profile",
        profile_name,
        "--start-at",
        "train",
        "--stop-after",
        "final_eval_loss",
        "--submit-chain",
    ]
    if dry_run:
        command.append("--dry-run")
    print(f"[PIPELINE] {shlex.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)
    after = set(graph_root.glob("job_graph_*.json"))
    created = sorted(after - before)
    if len(created) != 1:
        raise ValueError(
            f"{profile_name}: expected one new pipeline graph, found {len(created)}"
        )
    graph_path = created[0]
    return graph_path, json.loads(graph_path.read_text(encoding="utf-8"))


def executor_exclude(profile: dict[str, Any], stage: str) -> str:
    executor = profile.get("stage_executors", {}).get(stage, {})
    values = [
        str(flag).split("=", 1)[1]
        for flag in executor.get("extra_sbatch_flags", [])
        if str(flag).startswith("--exclude=")
    ]
    if len(values) != 1:
        raise ValueError(f"{stage}: expected one --exclude flag, found {values}")
    return values[0]


def job_by_run(graph: dict[str, Any], stage: str) -> dict[str, str]:
    rows = {
        str(job["run_id"]): str(job["job_id"])
        for job in graph.get("jobs", [])
        if job.get("stage") == stage
    }
    if set(rows) != set(RUN_IDS):
        raise ValueError(f"{stage}: unexpected graph run IDs {sorted(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    dry_run = bool(args.dry_run)

    validator = [
        str(PYTHON),
        "scripts/analysis/validate_final_lora_submission.py",
        "--require-no-training-artifacts",
    ]
    subprocess.run(validator, cwd=ROOT, check=True)

    manifest_path = CAMPAIGN_ROOT / (
        "submission_manifest.dry_run.json" if dry_run else "submission_manifest.json"
    )
    if manifest_path.exists() and not dry_run:
        raise ValueError(f"Submission manifest already exists: {manifest_path}")

    config = load_pipeline_config(ROOT / "configs/pipeline")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "purpose": "Compute-matched LoRA r2/r4 training, final loss, and v2-plus-English benchmarks",
        "dry_run": dry_run,
        "profiles": {},
        "jobs": [],
        "status": "submitting",
    }
    write_manifest(manifest_path, manifest)

    try:
        for profile_name, spec in PROFILES.items():
            profile = _resolve_profile(config, profile_name)
            output_root = (ROOT / str(profile["output_root"])).resolve()
            tasks_file = (ROOT / str(profile["benchmark"]["tasks_file"])).resolve()
            benchmark_root = (ROOT / str(profile["benchmark"]["output_dir"])).resolve()
            shard_root = output_root / "benchmark_shards"
            logs = output_root / "logs/final_lora"
            logs.mkdir(parents=True, exist_ok=True)

            graph_path, graph = pipeline_chain(profile_name, dry_run=dry_run)
            acceptance_jobs = job_by_run(graph, "train_acceptance")
            final_eval_jobs = job_by_run(graph, "final_eval_loss")
            gpu_exclude = executor_exclude(profile, "benchmark")
            profile_jobs: list[dict[str, Any]] = []

            for graph_job in graph.get("jobs", []):
                record = {
                    "source": "pipeline_chain",
                    "profile": profile_name,
                    **graph_job,
                }
                manifest["jobs"].append(record)
                profile_jobs.append(record)
            write_manifest(manifest_path, manifest)

            consolidation_jobs: dict[str, str] = {}
            for run_id in RUN_IDS:
                shard_jobs: list[str] = []
                for shard in SHARDS:
                    output_dir = shard_root / run_id / shard
                    job_id = submit_sbatch(
                        [
                            f"--job-name=lb{spec['size']}_{run_id[-2:]}_{shard[:4]}",
                            f"--exclude={gpu_exclude}",
                            f"--dependency=afterok:{acceptance_jobs[run_id]}",
                            f"--output={logs}/benchmark_{run_id}_{shard}.%j.log",
                            (
                                "--export=ALL,"
                                f"RUN_ROOT={output_root},RUN_ID={run_id},"
                                f"TASKS_FILE={tasks_file},SHARD={shard},OUTPUT_DIR={output_dir}"
                            ),
                            "scripts/slurm/run_final_lora_benchmark_shard.sh",
                        ],
                        dry_run=dry_run,
                        label=f"{spec['size']}_{run_id}_{shard}",
                    )
                    shard_jobs.append(job_id)
                    record = {
                        "source": "custom",
                        "profile": profile_name,
                        "stage": "benchmark_shard",
                        "run_id": run_id,
                        "shard": shard,
                        "job_id": job_id,
                        "dependency": f"afterok:{acceptance_jobs[run_id]}",
                        "output_dir": str(output_dir),
                    }
                    manifest["jobs"].append(record)
                    profile_jobs.append(record)
                    write_manifest(manifest_path, manifest)

                consolidation = submit_sbatch(
                    [
                        f"--job-name=lbc{spec['size']}_{run_id}",
                        f"--exclude={BAD_CPU_NODES}",
                        f"--dependency=afterok:{':'.join(shard_jobs)}",
                        f"--output={logs}/consolidate_{run_id}.%j.log",
                        (
                            "--export=ALL,"
                            f"RUN_ROOT={output_root},RUN_ID={run_id},"
                            f"SHARD_ROOT={shard_root / run_id},"
                            f"BENCHMARK_ROOT={benchmark_root},TASKS_FILE={tasks_file},"
                            f"PANEL={spec['panel']}"
                        ),
                        "scripts/slurm/consolidate_final_lora_benchmark_shards.sh",
                    ],
                    dry_run=dry_run,
                    label=f"{spec['size']}_{run_id}_consolidate",
                )
                consolidation_jobs[run_id] = consolidation
                record = {
                    "source": "custom",
                    "profile": profile_name,
                    "stage": "benchmark_consolidate",
                    "run_id": run_id,
                    "job_id": consolidation,
                    "dependency": f"afterok:{':'.join(shard_jobs)}",
                }
                manifest["jobs"].append(record)
                profile_jobs.append(record)
                write_manifest(manifest_path, manifest)

            report_dependencies = [
                *final_eval_jobs.values(),
                *consolidation_jobs.values(),
            ]
            report_job = submit_sbatch(
                [
                    f"--job-name=lora{spec['size']}_report",
                    f"--exclude={BAD_CPU_NODES}",
                    f"--dependency=afterok:{':'.join(report_dependencies)}",
                    f"--output={logs}/report.%j.log",
                    f"--export=ALL,PROFILE={profile_name}",
                    "scripts/slurm/build_and_validate_final_lora_report.sh",
                ],
                dry_run=dry_run,
                label=f"{spec['size']}_report",
            )
            report_record = {
                "source": "custom",
                "profile": profile_name,
                "stage": "report_and_validate",
                "run_id": "all",
                "job_id": report_job,
                "dependency": f"afterok:{':'.join(report_dependencies)}",
            }
            manifest["jobs"].append(report_record)
            profile_jobs.append(report_record)
            manifest["profiles"][profile_name] = {
                "size": spec["size"],
                "output_root": str(output_root),
                "pipeline_graph": str(graph_path.resolve()),
                "terminal_job_id": report_job,
                "jobs": len(profile_jobs),
            }
            write_manifest(manifest_path, manifest)

        manifest["status"] = "dry_run_complete" if dry_run else "submitted"
        manifest["terminal_job_ids"] = [
            data["terminal_job_id"] for data in manifest["profiles"].values()
        ]
        write_manifest(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "jobs": len(manifest["jobs"]),
                    "terminal_job_ids": manifest["terminal_job_ids"],
                    "manifest": str(manifest_path.resolve()),
                },
                indent=2,
            )
        )
        return 0
    except Exception:
        manifest["status"] = "submission_failed"
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
