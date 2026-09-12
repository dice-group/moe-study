#!/usr/bin/env python3
"""Capture controller-spooled Slurm scripts for an existing pipeline job graph.

The pipeline job graph records the local script path used at submission time.
Local files can later be overwritten when two jobs share a generated basename,
while Slurm retains the exact script that it accepted.  This utility preserves
those authoritative controller copies under unique job-ID-based names and
writes a companion manifest containing their hashes and live Slurm records.

The source manifest and Slurm jobs are read-only.  Snapshot files and the
companion manifest are written atomically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _safe_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._")
    if not text:
        raise ValueError(f"unsafe empty filename component derived from {value!r}")
    return text


def _atomic_write(path: Path, payload: bytes, *, require_absent: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if require_absent and path.exists():
        raise FileExistsError(f"refusing to replace existing provenance artifact: {path}")

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_immutable_snapshot(path: Path, payload: bytes) -> None:
    """Create *path*, or verify that an existing snapshot is byte-identical."""
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(f"existing spool snapshot differs from controller copy: {path}")
        return
    _atomic_write(path, payload)


def _run_scontrol(scontrol: str, *args: str) -> str:
    result = subprocess.run(
        [scontrol, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic").strip()
        raise RuntimeError(f"{' '.join([scontrol, *args])} failed ({result.returncode}): {detail}")
    return result.stdout


def _normalize_dependency(value: object) -> str:
    dependency = str(value or "").replace("(unfulfilled)", "").replace("(fulfilled)", "")
    if not dependency:
        return ""

    # Slurm's JSON output expands ``afterok:1:2`` into the equivalent
    # ``afterok:1,afterok:2``.  Compare canonical condition/job pairs rather
    # than these two serialization forms.
    conditions: list[str] = []
    for clause in dependency.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if ":" not in clause:
            conditions.append(clause)
            continue
        condition, raw_ids = clause.split(":", 1)
        job_ids = [job_id for job_id in raw_ids.split(":") if job_id]
        conditions.extend(f"{condition}:{job_id}" for job_id in job_ids)
    return ",".join(sorted(conditions))


def _load_manifest(path: Path, expected_job_count: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job graph must be a JSON object")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("job graph must contain a non-empty jobs list")
    if expected_job_count is not None and len(jobs) != expected_job_count:
        raise ValueError(f"job graph contains {len(jobs)} jobs, expected {expected_job_count}")

    required = {"job_id", "stage", "run_id", "script_path", "dependency"}
    job_ids: list[str] = []
    stage_runs: list[tuple[str, str]] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{index}] is not an object")
        missing = sorted(required - set(job))
        if missing:
            raise ValueError(f"jobs[{index}] is missing required fields: {missing}")
        job_id = str(job["job_id"]).strip()
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
            raise ValueError(f"jobs[{index}] has unsupported Slurm job ID: {job_id!r}")
        job_ids.append(job_id)
        stage_runs.append((str(job["stage"]), str(job["run_id"])))

    duplicate_ids = sorted(job_id for job_id, count in Counter(job_ids).items() if count != 1)
    if duplicate_ids:
        raise ValueError(f"job graph contains duplicate job IDs: {duplicate_ids}")
    duplicate_stage_runs = sorted(item for item, count in Counter(stage_runs).items() if count != 1)
    if duplicate_stage_runs:
        raise ValueError(f"job graph contains duplicate stage/run pairs: {duplicate_stage_runs}")
    return payload, jobs


def _capture_job(
    source_job: dict[str, Any],
    *,
    snapshot_dir: Path,
    source_path_groups: dict[str, list[str]],
    manifest_code_signature: str,
    scontrol: str,
) -> dict[str, Any]:
    job_id = str(source_job["job_id"])
    stage = str(source_job["stage"])
    run_id = str(source_job["run_id"])
    snapshot_name = ".".join(
        (
            _safe_component(job_id),
            _safe_component(stage),
            _safe_component(run_id),
            "sbatch.sh",
        )
    )
    snapshot_path = (snapshot_dir / snapshot_name).resolve()

    script_text = _run_scontrol(scontrol, "write", "batch_script", job_id, "-")
    script_bytes = script_text.encode("utf-8")
    if not script_text.startswith("#!/") or "#SBATCH --job-name=" not in script_text:
        raise RuntimeError(f"controller returned an invalid batch script for job {job_id}")

    live_payload = json.loads(_run_scontrol(scontrol, "show", "job", job_id, "--json"))
    live_jobs = live_payload.get("jobs", []) if isinstance(live_payload, dict) else []
    if not isinstance(live_jobs, list) or len(live_jobs) != 1:
        raise RuntimeError(f"expected exactly one live Slurm record for job {job_id}, got {len(live_jobs)}")
    live_record = live_jobs[0]
    if str(live_record.get("job_id", "")) != job_id:
        raise RuntimeError(
            f"live Slurm record ID {live_record.get('job_id')!r} does not match manifest job {job_id}"
        )

    live_dependency = _normalize_dependency(live_record.get("dependency"))
    manifest_dependency = _normalize_dependency(source_job.get("dependency"))
    if live_dependency != manifest_dependency:
        raise RuntimeError(
            f"job {job_id} dependency differs: live={live_dependency!r}, manifest={manifest_dependency!r}"
        )

    live_name = str(live_record.get("name", ""))
    if f"#SBATCH --job-name={live_name}\n" not in script_text:
        raise RuntimeError(f"job {job_id} spooled script does not contain live job name {live_name!r}")
    if manifest_code_signature:
        expected_signature = f"export PROOF_PIPELINE_CODE_SIG={manifest_code_signature}\n"
        if expected_signature not in script_text:
            raise RuntimeError(f"job {job_id} spooled script is missing code signature {manifest_code_signature}")

    _write_immutable_snapshot(snapshot_path, script_bytes)
    if _sha256_file(snapshot_path) != _sha256_bytes(script_bytes):
        raise RuntimeError(f"snapshot hash verification failed for job {job_id}: {snapshot_path}")

    source_script_path = str(Path(str(source_job["script_path"])).resolve())
    collision_job_ids = source_path_groups[source_script_path]
    return {
        **source_job,
        "job_id": job_id,
        "source_script_path": source_script_path,
        "source_script_path_collided": len(collision_job_ids) > 1,
        "source_script_path_job_ids": collision_job_ids,
        "spooled_script_path": str(snapshot_path),
        "spooled_script_sha256": _sha256_bytes(script_bytes),
        "spooled_script_bytes": len(script_bytes),
        "live_slurm_record": live_record,
    }


def snapshot_job_graph(
    *,
    manifest_path: Path,
    snapshot_dir: Path,
    companion_path: Path,
    expected_job_count: int | None,
    scontrol: str,
    workers: int,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    companion_path = companion_path.resolve()
    snapshot_dir = snapshot_dir.resolve()
    if companion_path == manifest_path:
        raise ValueError("companion path must differ from the source manifest")
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")

    source_hash_before = _sha256_file(manifest_path)
    manifest, jobs = _load_manifest(manifest_path, expected_job_count)
    source_path_groups_raw: dict[str, list[str]] = defaultdict(list)
    for job in jobs:
        source_path = str(Path(str(job["script_path"])).resolve())
        source_path_groups_raw[source_path].append(str(job["job_id"]))
    source_path_groups = {path: sorted(ids) for path, ids in source_path_groups_raw.items()}

    def capture_one(job: dict[str, Any]) -> dict[str, Any]:
        return _capture_job(
            job,
            snapshot_dir=snapshot_dir,
            source_path_groups=source_path_groups,
            manifest_code_signature=str(manifest.get("code_signature", "")),
            scontrol=scontrol,
        )
    # Preserve source-manifest order while bounding load on slurmctld.
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        captured = list(pool.map(capture_one, jobs))

    captured_ids = [item["job_id"] for item in captured]
    manifest_ids = [str(job["job_id"]) for job in jobs]
    snapshot_paths = [item["spooled_script_path"] for item in captured]
    if captured_ids != manifest_ids:
        raise RuntimeError("captured job IDs/order differ from source manifest")
    if len(snapshot_paths) != len(set(snapshot_paths)):
        raise RuntimeError("spool snapshot paths are not unique")
    if not all(Path(path).is_file() for path in snapshot_paths):
        raise RuntimeError("one or more spool snapshot files are missing")

    source_hash_after = _sha256_file(manifest_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source manifest changed during snapshot capture")

    collision_groups = [
        {"source_script_path": path, "job_ids": ids}
        for path, ids in sorted(source_path_groups.items())
        if len(ids) > 1
    ]
    state_counts = Counter(
        state
        for item in captured
        for state in item["live_slurm_record"].get("job_state", [])
    )
    captured_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    companion = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "slurm_controller_spool_snapshot",
        "captured_utc": captured_utc,
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": source_hash_before,
            "schema_version": manifest.get("schema_version"),
            "profile": manifest.get("profile"),
            "code_signature": manifest.get("code_signature"),
        },
        "snapshot_directory": str(snapshot_dir),
        "jobs": captured,
        "audit": {
            "expected_job_count": expected_job_count if expected_job_count is not None else len(jobs),
            "source_job_count": len(jobs),
            "captured_job_count": len(captured),
            "unique_job_id_count": len(set(captured_ids)),
            "unique_snapshot_path_count": len(set(snapshot_paths)),
            "all_jobs_accounted_for": len(captured) == len(jobs) == len(set(captured_ids)),
            "source_manifest_preserved": source_hash_after == source_hash_before,
            "source_script_path_collision_count": len(collision_groups),
            "source_script_path_collisions": collision_groups,
            "live_state_counts": dict(sorted(state_counts.items())),
            "capture_workers": min(workers, len(jobs)),
        },
    }
    if not companion["audit"]["all_jobs_accounted_for"]:
        raise RuntimeError("not all manifest jobs were captured")

    companion_bytes = (json.dumps(companion, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(companion_path, companion_bytes, require_absent=True)
    if json.loads(companion_path.read_text(encoding="utf-8"))["audit"]["captured_job_count"] != len(jobs):
        raise RuntimeError("companion manifest post-write verification failed")
    return companion


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Existing pipeline job_graph JSON")
    parser.add_argument("--snapshot-dir", type=Path, help="Directory for unique controller-spooled scripts")
    parser.add_argument("--companion-json", type=Path, help="Atomic companion manifest output")
    parser.add_argument("--expected-job-count", type=int, help="Fail unless the graph contains this many jobs")
    parser.add_argument("--scontrol", default="scontrol", help="scontrol executable (default: scontrol)")
    parser.add_argument("--workers", type=int, default=8, help="Maximum concurrent controller reads (default: 8)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = args.manifest.resolve()
    snapshot_dir = args.snapshot_dir or manifest.with_name(f"{manifest.stem}.spooled_scripts")
    companion = args.companion_json or manifest.with_name(f"{manifest.stem}.spool_verified.json")
    result = snapshot_job_graph(
        manifest_path=manifest,
        snapshot_dir=snapshot_dir,
        companion_path=companion,
        expected_job_count=args.expected_job_count,
        scontrol=str(args.scontrol),
        workers=args.workers,
    )
    audit = result["audit"]
    print(f"source_manifest={result['source_manifest']['path']}")
    print(f"source_manifest_sha256={result['source_manifest']['sha256']}")
    print(f"snapshot_directory={result['snapshot_directory']}")
    print(f"companion_json={companion.resolve()}")
    print(f"captured_jobs={audit['captured_job_count']}")
    print(f"source_script_path_collisions={audit['source_script_path_collision_count']}")
    print("snapshot_audit=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
