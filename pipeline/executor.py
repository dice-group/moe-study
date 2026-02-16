#!/usr/bin/env python3
"""Executor abstraction for pipeline command execution.

Provides a protocol for running shell commands either locally (subprocess)
or via Slurm (sbatch + polling).  Stages never know which backend is active —
they just call ``executor.run()`` or ``executor.run_batch()`` with command lists.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

LOGGER = logging.getLogger("proof_pipeline")

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Executor(Protocol):
    """Minimal contract every executor backend must satisfy."""

    def run(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
    ) -> None:
        """Execute *cmd* and stream output to *log_path*."""
        ...

    def run_batch(
        self,
        jobs: Sequence[tuple[list[str], dict[str, str], Path]],
        dry_run: bool,
    ) -> None:
        """Execute multiple independent jobs.

        Each element is ``(cmd, env, log_path)``.
        ``LocalExecutor`` runs them sequentially; ``SlurmExecutor`` submits
        all as separate sbatch jobs and waits for all to finish.
        """
        ...


# ---------------------------------------------------------------------------
# Local executor (current behaviour, subprocess.Popen)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LocalExecutor:
    """Run commands as local subprocesses — the default."""

    def run(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
    ) -> None:
        started_at = _utc_stamp()
        started_perf = time.perf_counter()
        LOGGER.info("[CMD] %s", " ".join(cmd))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[START_UTC] {started_at}\n")
                fh.write(f"[DRY_RUN CMD] {' '.join(cmd)}\n")
                fh.write("[END] dry_run=true exit_code=0 duration_seconds=0.000\n")
            return

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[START_UTC] {started_at}\n")
            fh.write(f"\n[CMD] {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                LOGGER.info("%s", line.rstrip("\n"))
                fh.write(line)
            code = proc.wait()
            duration = time.perf_counter() - started_perf
            fh.write(f"[END] dry_run=false exit_code={code} duration_seconds={duration:.3f}\n")
            if code != 0:
                raise subprocess.CalledProcessError(code, cmd)

    def run_batch(
        self,
        jobs: Sequence[tuple[list[str], dict[str, str], Path]],
        dry_run: bool,
    ) -> None:
        """Run jobs sequentially on a local machine."""
        for cmd, env, log_path in jobs:
            self.run(cmd, env, dry_run, log_path)


# ---------------------------------------------------------------------------
# Slurm executor (sbatch + poll)
# ---------------------------------------------------------------------------

# Defaults applied when the profile sets ``executor.type = "slurm"`` but omits
# specific fields.
_SLURM_DEFAULTS: dict[str, Any] = {
    "partition": "gpu",
    "gres": "gpu:h100:1",
    "mem": "160G",
    "time": "24:00:00",
    "cpus_per_task": 4,
    "modules": [
        "toolchain/foss/2024a",
        "system/CUDA/12.6.0",
        "lib/NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0",
    ],
    "conda_base": "/opt/software/pc2/EB-SW/software/Miniforge3/25.3.0-3",
    "conda_env": "merlin",
    "poll_interval_seconds": 30,
    "extra_sbatch_flags": [],
}


@dataclass(slots=True)
class SlurmExecutor:
    """Submit commands via ``sbatch`` and poll ``sacct`` until they finish."""

    partition: str = _SLURM_DEFAULTS["partition"]
    gres: str = _SLURM_DEFAULTS["gres"]
    mem: str = _SLURM_DEFAULTS["mem"]
    time_limit: str = _SLURM_DEFAULTS["time"]
    cpus_per_task: int = _SLURM_DEFAULTS["cpus_per_task"]
    modules: list[str] = field(default_factory=lambda: list(_SLURM_DEFAULTS["modules"]))
    conda_base: str = _SLURM_DEFAULTS["conda_base"]
    conda_env: str = _SLURM_DEFAULTS["conda_env"]
    poll_interval: int = _SLURM_DEFAULTS["poll_interval_seconds"]
    extra_sbatch_flags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def run(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
    ) -> None:
        job_ids = self._submit_jobs([(cmd, env, log_path)], dry_run)
        if job_ids:
            self._wait_for_jobs(job_ids, dry_run)

    def run_batch(
        self,
        jobs: Sequence[tuple[list[str], dict[str, str], Path]],
        dry_run: bool,
    ) -> None:
        """Submit all jobs via sbatch, then wait for all to finish."""
        job_ids = self._submit_jobs(jobs, dry_run)
        if job_ids:
            self._wait_for_jobs(job_ids, dry_run)

    # ------------------------------------------------------------------
    # sbatch submission
    # ------------------------------------------------------------------

    def _submit_jobs(
        self,
        jobs: Sequence[tuple[list[str], dict[str, str], Path]],
        dry_run: bool,
    ) -> list[tuple[str, Path]]:
        """Submit each job and return list of (job_id, log_path) tuples."""
        submitted: list[tuple[str, Path]] = []
        for cmd, env, log_path in jobs:
            job_id = self._sbatch_submit(cmd, env, dry_run, log_path)
            if job_id:
                submitted.append((job_id, log_path))
        return submitted

    def _sbatch_submit(
        self,
        cmd: list[str],
        env: dict[str, str],
        dry_run: bool,
        log_path: Path,
    ) -> str | None:
        """Write a job script and submit it.  Returns the Slurm job ID."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = _utc_stamp()

        if dry_run:
            LOGGER.info("[DRY_RUN SBATCH] %s", " ".join(cmd))
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[START_UTC] {started_at}\n")
                fh.write(f"[DRY_RUN SBATCH] {' '.join(cmd)}\n")
                fh.write("[END] dry_run=true exit_code=0 duration_seconds=0.000\n")
            return None

        job_name = _job_name_from_log(log_path)
        slurm_log = log_path.with_suffix(".slurm.log")
        script = self._build_job_script(cmd, env, job_name, slurm_log)

        # Write the job script to a temp file next to the log.
        script_path = log_path.with_suffix(".sbatch.sh")
        script_path.write_text(script, encoding="utf-8")

        LOGGER.info("[SBATCH] Submitting %s", script_path)
        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip().split(";")[0]
        LOGGER.info("[SBATCH] Submitted job %s for %s", job_id, job_name)

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[START_UTC] {started_at}\n")
            fh.write(f"[SBATCH] job_id={job_id} script={script_path}\n")
            fh.write(f"[CMD] {' '.join(cmd)}\n")

        return job_id

    def _build_job_script(
        self,
        cmd: list[str],
        env: dict[str, str],
        job_name: str,
        slurm_log: Path,
    ) -> str:
        """Produce a complete sbatch shell script."""
        sbatch_lines = [
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --gres={self.gres}",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --time={self.time_limit}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --output={slurm_log}",
        ]
        for flag in self.extra_sbatch_flags:
            sbatch_lines.append(f"#SBATCH {flag}")

        module_lines = ""
        if self.modules:
            module_lines = "module purge\nmodule load " + " ".join(self.modules)

        # Export key env vars the pipeline needs.
        env_exports = []
        for key in sorted(env):
            # Only export vars the pipeline explicitly sets.
            if key.startswith(("PYTHONPATH", "PROOF_PIPELINE_", "WANDB_", "CUDA_", "HF_")):
                env_exports.append(f'export {key}="{env[key]}"')

        conda_activate = textwrap.dedent(f"""\
            CONDA_BASE="{self.conda_base}"
            source "${{CONDA_BASE}}/etc/profile.d/conda.sh"
            conda activate {self.conda_env}
        """)

        cmd_str = " ".join(_shell_quote(c) for c in cmd)

        script = textwrap.dedent(f"""\
            #!/bin/bash
            {chr(10).join(sbatch_lines)}

            set -euo pipefail

            {module_lines}

            {conda_activate}
            {chr(10).join(env_exports)}

            export PYTHONUNBUFFERED=1

            echo "[SLURM_START] $(date -u +%Y-%m-%dT%H:%M:%SZ) job=$SLURM_JOB_ID"
            {cmd_str}
            echo "[SLURM_END] $(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=$?"
        """)
        return script

    # ------------------------------------------------------------------
    # polling / waiting
    # ------------------------------------------------------------------

    def _wait_for_jobs(
        self,
        jobs: list[tuple[str, Path]],
        dry_run: bool,
    ) -> None:
        """Poll sacct until all jobs have finished.  Raises on failure."""
        if dry_run or not jobs:
            return

        pending = dict(jobs)  # {job_id: log_path}
        LOGGER.info("[SLURM] Waiting for %d job(s): %s", len(pending), ", ".join(pending))

        while pending:
            time.sleep(self.poll_interval)
            finished = self._poll_jobs(list(pending.keys()))
            for job_id, (state, exit_code) in finished.items():
                log_path = pending.pop(job_id)
                slurm_log = log_path.with_suffix(".slurm.log")
                duration_str = "unknown"

                # Stream Slurm log content back to pipeline log.
                if slurm_log.exists():
                    content = slurm_log.read_text(encoding="utf-8", errors="replace")
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(f"\n--- slurm output (job {job_id}) ---\n")
                        fh.write(content)
                    for line in content.splitlines():
                        LOGGER.info("[SLURM:%s] %s", job_id, line.rstrip())

                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"[END] slurm_job={job_id} state={state} "
                        f"exit_code={exit_code} duration_seconds={duration_str}\n"
                    )

                if state not in ("COMPLETED",):
                    raise subprocess.CalledProcessError(
                        int(exit_code) if exit_code.isdigit() else 1,
                        [f"slurm-job-{job_id}"],
                    )
                LOGGER.info("[SLURM] Job %s finished: %s", job_id, state)

            if pending:
                LOGGER.info(
                    "[SLURM] Still waiting for %d job(s): %s",
                    len(pending),
                    ", ".join(pending),
                )

    @staticmethod
    def _poll_jobs(job_ids: list[str]) -> dict[str, tuple[str, str]]:
        """Check sacct for finished jobs.  Returns {job_id: (state, exit_code)}."""
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(job_ids),
                "--format=JobID,State,ExitCode",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
        )
        finished: dict[str, tuple[str, str]] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            jid, state, exit_code = parts[0], parts[1], parts[2]
            # Skip sub-steps (e.g. "12345.batch").
            if "." in jid:
                continue
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                finished[jid] = (state, exit_code.split(":")[0])
        return finished


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_executor(
    profile: dict[str, Any],
    stage_cfg: dict[str, Any] | None = None,
) -> LocalExecutor | SlurmExecutor:
    """Instantiate the right executor from profile config.

    Reads ``profile["executor"]`` for the base config.  If *stage_cfg*
    contains an ``"executor"`` key, those fields override the profile-level
    settings (e.g. a stage can request more GPUs or memory).

    Returns ``LocalExecutor`` when executor type is ``"local"`` or absent.
    """
    base_cfg: dict[str, Any] = dict(profile.get("executor", {}))
    exec_type = base_cfg.pop("type", "local")

    if exec_type == "local":
        return LocalExecutor()

    if exec_type != "slurm":
        raise ValueError(f"Unknown executor type: {exec_type!r}  (expected 'local' or 'slurm')")

    # Merge stage-level overrides.
    if stage_cfg:
        stage_exec = stage_cfg.get("executor", {})
        base_cfg.update(stage_exec)

    return SlurmExecutor(
        partition=base_cfg.get("partition", _SLURM_DEFAULTS["partition"]),
        gres=base_cfg.get("gres", _SLURM_DEFAULTS["gres"]),
        mem=base_cfg.get("mem", _SLURM_DEFAULTS["mem"]),
        time_limit=base_cfg.get("time", _SLURM_DEFAULTS["time"]),
        cpus_per_task=int(base_cfg.get("cpus_per_task", _SLURM_DEFAULTS["cpus_per_task"])),
        modules=list(base_cfg.get("modules", _SLURM_DEFAULTS["modules"])),
        conda_base=str(base_cfg.get("conda_base", _SLURM_DEFAULTS["conda_base"])),
        conda_env=str(base_cfg.get("conda_env", _SLURM_DEFAULTS["conda_env"])),
        poll_interval=int(base_cfg.get("poll_interval_seconds", _SLURM_DEFAULTS["poll_interval_seconds"])),
        extra_sbatch_flags=list(base_cfg.get("extra_sbatch_flags", [])),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _job_name_from_log(log_path: Path) -> str:
    """Derive a short Slurm job name from the log file path."""
    name = log_path.stem
    # Truncate to 64 chars (Slurm limit is 128).
    return name[:64]


def _shell_quote(s: str) -> str:
    """Simple shell quoting — wrap in single quotes if it contains specials."""
    if re.match(r"^[a-zA-Z0-9_./:=@,-]+$", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"
