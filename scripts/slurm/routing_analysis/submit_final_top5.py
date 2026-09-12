#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "96": REPO_ROOT / "configs/routing_analysis/final_top5_96_langs/initial_1000seq.json",
    "200": REPO_ROOT / "configs/routing_analysis/final_top5_200_langs/initial_1000seq.json",
}
STATIC_METHODS = {"MOELORA", "MTLLORA"}


def _method(checkpoint: Path) -> str:
    payload = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    return str(payload["peft_type"]).upper()


def _submit(command: list[str], *, dry_run: bool) -> str:
    print("[CMD]", " ".join(command))
    if dry_run:
        return f"dry-{abs(hash(tuple(command))) % 1000000}"
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from: {completed.stdout!r}")
    return job_id


def _sbatch_base(*, job_name: str, log_path: Path, time: str, gpu: bool = True) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        "--partition=gpu" if gpu else "--partition=normal",
        f"--job-name={job_name}",
        f"--time={time}",
        "--cpus-per-task=8",
        "--mem=64G",
        f"--output={log_path}",
        f"--error={log_path}",
    ]
    if gpu:
        command.append("--gres=gpu:a100:1")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit the final 96/200 top-5 routing-analysis job graph.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-canaries", action="store_true")
    args = parser.parse_args()

    logs = REPO_ROOT / "outputs/routing_analysis/slurm_logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker = REPO_ROOT / "scripts/slurm/routing_analysis/run_one.sh"
    postprocess = REPO_ROOT / "scripts/slurm/routing_analysis/run_postprocess.sh"
    job_graph: dict[str, object] = {"canaries": {}, "full": {}, "postprocess": None}
    full_jobs: list[str] = []

    for panel, config_path in CONFIGS.items():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for adapter in config["adapters"]:
            run_id = str(adapter["run_id"])
            checkpoint = Path(adapter["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = REPO_ROOT / checkpoint
            method = _method(checkpoint)
            key = f"{panel}:{run_id}"
            dependency = ""
            if not args.skip_canaries:
                canary_log = logs / f"canary_{panel}_{run_id}_%j.log"
                canary_cmd = _sbatch_base(
                    job_name=f"rt-c{panel}-{run_id}"[:128],
                    log_path=canary_log,
                    time="02:00:00",
                )
                canary_cmd.extend([str(worker), str(config_path), run_id, "canary"])
                canary_id = _submit(canary_cmd, dry_run=args.dry_run)
                job_graph["canaries"][key] = canary_id
                dependency = f"--dependency=afterok:{canary_id}"

            walltime = "02:00:00" if method in STATIC_METHODS else ("24:00:00" if panel == "96" else "48:00:00")
            full_log = logs / f"full_{panel}_{run_id}_%j.log"
            full_cmd = _sbatch_base(
                job_name=f"rt-f{panel}-{run_id}"[:128],
                log_path=full_log,
                time=walltime,
            )
            if dependency:
                full_cmd.append(dependency)
            full_cmd.extend([str(worker), str(config_path), run_id, "full"])
            full_id = _submit(full_cmd, dry_run=args.dry_run)
            job_graph["full"][key] = full_id
            full_jobs.append(full_id)

    post_log = logs / "postprocess_%j.log"
    post_cmd = _sbatch_base(
        job_name="routing-final-top5-post",
        log_path=post_log,
        time="06:00:00",
        gpu=False,
    )
    post_cmd.append(f"--dependency=afterok:{':'.join(full_jobs)}")
    post_cmd.append(str(postprocess))
    post_id = _submit(post_cmd, dry_run=args.dry_run)
    job_graph["postprocess"] = post_id
    job_graph["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    job_graph["dry_run"] = args.dry_run

    manifest = logs / ("submission_manifest.dry_run.json" if args.dry_run else "submission_manifest.json")
    manifest.write_text(json.dumps(job_graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] wrote submission graph to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
