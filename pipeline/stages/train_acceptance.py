#!/usr/bin/env python3
"""Validate one completed training attempt before downstream evaluation."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from .runtime import _dump_json
from ..context import PipelineContext
from ..training import adapter_checkpoint_provenance


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)_adapter$")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing required file: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"could not read JSON {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected a JSON object in {path}"
    return payload, None


def _nonfinite_paths(value: Any, *, prefix: str = "$") -> list[str]:
    bad: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        bad.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            bad.extend(_nonfinite_paths(item, prefix=f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            bad.extend(_nonfinite_paths(item, prefix=f"{prefix}[{index}]"))
    return bad


def _positive_global_step(payload: dict[str, Any] | None) -> tuple[int | None, str | None]:
    raw = payload.get("global_step") if payload is not None else None
    if isinstance(raw, bool):
        return None, f"trainer_state global_step must be a positive integer (got {raw!r})"
    try:
        step = int(raw)
    except (TypeError, ValueError):
        return None, f"trainer_state global_step must be a positive integer (got {raw!r})"
    if isinstance(raw, float) and not raw.is_integer():
        return None, f"trainer_state global_step must be a positive integer (got {raw!r})"
    if step <= 0:
        return None, f"trainer_state global_step must be positive (got {step})"
    return step, None


def _numeric_train_runtime(payload: dict[str, Any] | None) -> tuple[float | None, str | None]:
    raw = payload.get("train_runtime") if payload is not None else None
    if isinstance(raw, bool):
        return None, f"train_results train_runtime must be finite and numeric (got {raw!r})"
    try:
        runtime = float(raw)
    except (TypeError, ValueError):
        return None, f"train_results train_runtime must be finite and numeric (got {raw!r})"
    if not math.isfinite(runtime) or runtime <= 0:
        return None, f"train_results train_runtime must be finite and positive (got {raw!r})"
    return runtime, None


def _finite_train_loss(payload: dict[str, Any] | None) -> tuple[float | None, str | None]:
    raw = payload.get("train_loss") if payload is not None else None
    if isinstance(raw, bool):
        return None, f"train_results train_loss must be finite and numeric (got {raw!r})"
    try:
        loss = float(raw)
    except (TypeError, ValueError):
        return None, f"train_results train_loss must be finite and numeric (got {raw!r})"
    if not math.isfinite(loss):
        return None, f"train_results train_loss must be finite (got {raw!r})"
    return loss, None


def _resolve_adapter_dir(run_dir: Path, global_step: int | None) -> Path | None:
    if global_step is not None:
        expected = run_dir / f"checkpoint-{global_step}_adapter"
        if expected.is_dir():
            return expected

    candidates: list[tuple[int, Path]] = []
    if run_dir.is_dir():
        for child in run_dir.iterdir():
            if not child.is_dir():
                continue
            match = _CHECKPOINT_RE.match(child.name)
            if match:
                candidates.append((int(match.group(1)), child))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    if (run_dir / "adapter_config.json").is_file():
        return run_dir
    return None


def _scan_logs(
    paths: list[Path],
    patterns: list[str],
    required_patterns: list[str] | None = None,
) -> dict[str, Any]:
    required_patterns = required_patterns or []
    compiled: list[tuple[str, re.Pattern[str]]] = []
    required_compiled: list[tuple[str, re.Pattern[str]]] = []
    invalid_patterns: list[str] = []
    for pattern in patterns:
        try:
            compiled.append((pattern, re.compile(pattern, flags=re.IGNORECASE)))
        except re.error as exc:
            invalid_patterns.append(f"{pattern!r}: {exc}")
    for pattern in required_patterns:
        try:
            required_compiled.append((pattern, re.compile(pattern, flags=re.IGNORECASE)))
        except re.error as exc:
            invalid_patterns.append(f"required {pattern!r}: {exc}")

    matches: list[dict[str, Any]] = []
    required_seen = {pattern: False for pattern in required_patterns}
    scanned: list[str] = []
    missing: list[str] = []
    for path in paths:
        if not path.is_file():
            missing.append(str(path))
            continue
        scanned.append(str(path))
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    for source, regex in compiled:
                        if regex.search(line):
                            if len(matches) < 50:
                                matches.append(
                                    {
                                        "path": str(path),
                                        "line": line_number,
                                        "pattern": source,
                                        "text": line.strip()[:500],
                                    }
                                )
                            break
                    for source, regex in required_compiled:
                        if not required_seen[source] and regex.search(line):
                            required_seen[source] = True
        except OSError as exc:
            missing.append(f"{path}: {exc}")

    missing_required = [pattern for pattern, seen in required_seen.items() if not seen]
    return {
        # Log rejection is fail-closed: configured patterns are useful only if
        # the exact training log was available and readable.
        "pass": (
            not invalid_patterns
            and not matches
            and not missing
            and not missing_required
            and (not patterns and not required_patterns or bool(scanned))
        ),
        "scanned_paths": scanned,
        "missing_paths": missing,
        "patterns": patterns,
        "required_patterns": required_patterns,
        "missing_required_patterns": missing_required,
        "invalid_patterns": invalid_patterns,
        "matches": matches,
    }


def _query_slurm_accounting(job_id: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobIDRaw,State,ExitCode",
                "--parsable2",
                "--noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except Exception as exc:
        return {"available": False, "job_id": job_id, "error": str(exc)}

    if result.returncode != 0:
        return {
            "available": False,
            "job_id": job_id,
            "error": (result.stderr or result.stdout or "sacct failed").strip(),
        }

    for raw_line in result.stdout.splitlines():
        fields = raw_line.strip().split("|")
        if len(fields) < 3 or fields[0] != job_id:
            continue
        state = fields[1].strip().rstrip("+")
        return {
            "available": True,
            "job_id": job_id,
            "state": state,
            "exit_code": fields[2].strip(),
        }
    return {"available": False, "job_id": job_id, "error": "sacct returned no parent-job record"}


def _discover_log_paths(*, logs_root: Path, run_id: str, explicit_paths: list[Path]) -> list[Path]:
    if explicit_paths:
        return list(dict.fromkeys(explicit_paths))

    candidates = list((logs_root / "train_runs" / run_id).glob("*.train.log"))
    candidates.extend((logs_root / "chain" / "slurm").glob(f"chain_train.{run_id}.*.slurm.log"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return []
    newest_mtime = max(path.stat().st_mtime for path in candidates if path.is_file())
    # A direct and a Slurm log from the same launch can differ slightly in mtime.
    return sorted(path for path in candidates if path.is_file() and path.stat().st_mtime >= newest_mtime - 300)


def evaluate_training_run(
    *,
    run_id: str,
    run_dir: Path,
    config: dict[str, Any],
    log_paths: list[Path],
    train_job_id: str,
    slurm_query: Callable[[str], dict[str, Any]] = _query_slurm_accounting,
) -> dict[str, Any]:
    """Return a complete, serializable acceptance decision for one run."""

    reasons: list[str] = []
    checks: dict[str, Any] = {}

    results_path = run_dir / "train_results.json"
    state_path = run_dir / "trainer_state.json"
    train_results, results_error = _read_json(results_path)
    trainer_state, state_error = _read_json(state_path)
    checks["train_results"] = {"pass": results_error is None, "path": str(results_path), "error": results_error}
    checks["trainer_state"] = {"pass": state_error is None, "path": str(state_path), "error": state_error}
    if results_error:
        reasons.append(results_error)
    if state_error:
        reasons.append(state_error)

    nonfinite: list[str] = []
    if train_results is not None:
        nonfinite.extend(f"train_results{path[1:]}" for path in _nonfinite_paths(train_results))
    if trainer_state is not None:
        nonfinite.extend(f"trainer_state{path[1:]}" for path in _nonfinite_paths(trainer_state))
    checks["finite_values"] = {"pass": not nonfinite, "nonfinite_paths": nonfinite}
    if nonfinite:
        reasons.append(f"non-finite numeric values found: {nonfinite[:10]}")

    global_step, step_error = _positive_global_step(trainer_state)
    minimum_step_raw = config.get("minimum_global_step", 1)
    try:
        minimum_step = int(minimum_step_raw)
        if (
            isinstance(minimum_step_raw, bool)
            or (isinstance(minimum_step_raw, float) and not minimum_step_raw.is_integer())
            or minimum_step < 1
        ):
            raise ValueError
    except (TypeError, ValueError):
        minimum_step = None
        step_error = "train_acceptance minimum_global_step must be a positive integer"
    if step_error is None and global_step is not None and minimum_step is not None and global_step < minimum_step:
        step_error = f"trainer_state global_step {global_step} is below required minimum {minimum_step}"
    checks["global_step"] = {
        "pass": step_error is None,
        "value": global_step,
        "minimum": minimum_step,
        "error": step_error,
    }
    if step_error:
        reasons.append(step_error)

    train_loss, loss_error = _finite_train_loss(train_results)
    checks["train_loss"] = {"pass": loss_error is None, "value": train_loss, "error": loss_error}
    if loss_error:
        reasons.append(loss_error)

    runtime, runtime_error = _numeric_train_runtime(train_results)
    target_raw = config.get("target_runtime_seconds")
    tolerance_raw = config.get("runtime_tolerance_seconds", 0)
    try:
        target = float(target_raw)
        tolerance = float(tolerance_raw)
        if not math.isfinite(target) or target <= 0 or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError
        config_error = None
    except (TypeError, ValueError):
        target = None
        tolerance = None
        config_error = (
            "train_acceptance target_runtime_seconds must be positive and "
            "runtime_tolerance_seconds must be non-negative"
        )

    runtime_pass = runtime_error is None and config_error is None
    deviation: float | None = None
    if runtime_pass and runtime is not None and target is not None and tolerance is not None:
        deviation = runtime - target
        runtime_pass = abs(deviation) <= tolerance
        if not runtime_pass:
            runtime_error = (
                f"train_runtime {runtime:.3f}s is outside target {target:.3f}s "
                f"± {tolerance:.3f}s"
            )
    runtime_error = runtime_error or config_error
    checks["runtime"] = {
        "pass": runtime_pass,
        "value_seconds": runtime,
        "target_seconds": target,
        "tolerance_seconds": tolerance,
        "deviation_seconds": deviation,
        "error": runtime_error,
    }
    if runtime_error:
        reasons.append(runtime_error)

    require_marker = bool(config.get("require_direct_save_marker", False))
    adapter_dir = _resolve_adapter_dir(run_dir, global_step)
    if adapter_dir is None:
        checkpoint_check: dict[str, Any] = {
            "pass": False,
            "adapter_path": "",
            "error": "no adapter checkpoint found",
        }
    else:
        expected_step = global_step if _CHECKPOINT_RE.match(adapter_dir.name) else None
        provenance = adapter_checkpoint_provenance(
            adapter_dir,
            expected_step=expected_step,
            require_direct_save_marker=require_marker,
        )
        nonempty = bool((provenance.get("adapter_weight_size") or 0) > 0)
        checkpoint_check = {
            **provenance,
            "pass": bool(provenance.get("valid")) and nonempty,
            "nonempty": nonempty,
        }
        if not nonempty:
            checkpoint_check.setdefault("errors", []).append("adapter weight file is empty")
    checks["checkpoint"] = checkpoint_check
    if not checkpoint_check["pass"]:
        errors = checkpoint_check.get("errors") or [checkpoint_check.get("error", "invalid checkpoint")]
        reasons.append(f"checkpoint rejected: {errors}")

    patterns_raw = config.get("reject_log_patterns", [])
    if not isinstance(patterns_raw, list):
        patterns_raw = [patterns_raw]
    patterns = [str(pattern) for pattern in patterns_raw if str(pattern).strip()]
    required_patterns_raw = config.get("required_log_patterns", [])
    if not isinstance(required_patterns_raw, list):
        required_patterns_raw = [required_patterns_raw]
    required_patterns = [str(pattern) for pattern in required_patterns_raw if str(pattern).strip()]
    log_check = _scan_logs(log_paths, patterns, required_patterns)
    checks["logs"] = log_check
    if not log_check["pass"]:
        reasons.append(
            "training log is missing a required signal/save marker, contains a rejection pattern, "
            "or could not be validated"
        )

    require_slurm = bool(config.get("require_slurm_completed", False))
    if require_slurm:
        if not train_job_id:
            slurm = {"pass": False, "available": False, "error": "training job ID was not provided"}
        else:
            slurm = slurm_query(train_job_id)
            slurm["pass"] = bool(
                slurm.get("available")
                and slurm.get("state") == "COMPLETED"
                and str(slurm.get("exit_code", "")).split(":", 1)[0] == "0"
            )
        if not slurm["pass"]:
            reasons.append(f"Slurm training job was not COMPLETED with exit code 0: {slurm}")
    else:
        slurm = {"pass": True, "required": False, "job_id": train_job_id}
    checks["slurm"] = slurm

    return {
        "schema_version": 1,
        "stage": "train_acceptance",
        "checked_at_utc": _utc_now(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "accepted": not reasons,
        "reasons": reasons,
        "checks": checks,
    }


def stage_train_acceptance(ctx: PipelineContext) -> None:
    if "train_acceptance" not in ctx.active_stages:
        return
    if not bool(ctx.train_acceptance_cfg.get("enabled", False)):
        return

    runs = ctx.train_cfg.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise SystemExit("train_acceptance: no training runs configured in profile.training.runs")

    explicit_log_raw = os.environ.get("PROOF_PIPELINE_TRAIN_LOG_PATH", "").strip()
    explicit_logs = [Path(item) for item in explicit_log_raw.split(os.pathsep) if item.strip()]
    train_job_id = os.environ.get("PROOF_PIPELINE_TRAIN_JOB_ID", "").strip()
    failures: list[str] = []

    for run in runs:
        run_id = str(run["id"])
        output_path = ctx.train_acceptance_root / run_id / "acceptance.json"
        if ctx.args.dry_run:
            ctx.logger.info(
                "[DRY_RUN] train_acceptance run=%s output=%s train_job_id=%s",
                run_id,
                output_path,
                train_job_id or "<missing>",
            )
            continue

        try:
            log_paths = _discover_log_paths(
                logs_root=ctx.logs_root,
                run_id=run_id,
                explicit_paths=explicit_logs,
            )
            decision = evaluate_training_run(
                run_id=run_id,
                run_dir=ctx.train_root / run_id,
                config=ctx.train_acceptance_cfg,
                log_paths=log_paths,
                train_job_id=train_job_id,
            )
        except (Exception, SystemExit) as exc:
            decision = {
                "schema_version": 1,
                "stage": "train_acceptance",
                "checked_at_utc": _utc_now(),
                "run_id": run_id,
                "run_dir": str(ctx.train_root / run_id),
                "accepted": False,
                "reasons": [f"unexpected acceptance error: {type(exc).__name__}: {exc}"],
                "checks": {},
            }
        _dump_json(output_path, decision)
        ctx.logger.info("[TRAIN_ACCEPTANCE] run=%s accepted=%s output=%s", run_id, decision["accepted"], output_path)
        if not decision["accepted"]:
            failures.append(run_id)

    if failures:
        raise SystemExit(f"train_acceptance rejected run(s): {failures}")
