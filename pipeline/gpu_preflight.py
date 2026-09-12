#!/usr/bin/env python3
"""Fail-fast GPU health checks used by local and Slurm pipeline executors."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Sequence


def _run(command: Sequence[str], *, timeout_seconds: float) -> str:
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=True,
            timeout=max(1.0, timeout_seconds),
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"GPU preflight command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"GPU preflight timed out: {' '.join(command)}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise SystemExit(f"GPU preflight command failed ({exc.returncode}): {' '.join(command)}\n{detail}") from exc
    return result.stdout


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SystemExit("GPU preflight exceeded its timeout")
    return remaining


def _visible_torch_gpu_names() -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("GPU preflight requires PyTorch") from exc
    try:
        count = torch.cuda.device_count()
        return [torch.cuda.get_device_name(index) for index in range(count)]
    except Exception as exc:
        raise SystemExit(f"GPU preflight could not inspect CUDA devices: {exc}") from exc


def _allocated_nvidia_rows(*, deadline: float, expected_gpu_count: int) -> list[list[str]]:
    fields = ["index", "uuid", "ecc.errors.uncorrected.volatile.total"]
    output = _run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        timeout_seconds=_remaining(deadline),
    )
    rows = [[column.strip() for column in line.split(",")] for line in output.splitlines() if line.strip()]
    if len(rows) == expected_gpu_count:
        return rows

    visible_raw = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
    visible = [token.strip() for token in visible_raw.split(",") if token.strip()]
    selected = [row for row in rows if len(row) >= 2 and (row[0] in visible or row[1] in visible)]
    if len(selected) != expected_gpu_count:
        raise SystemExit(
            "GPU preflight could not map allocated CUDA devices to nvidia-smi rows: "
            f"expected={expected_gpu_count} visible={visible} rows={len(rows)} selected={len(selected)}"
        )
    return selected


def _check_nvlink(*, nvidia_rows: list[list[str]], deadline: float) -> None:
    for row in nvidia_rows:
        device_index = row[0]
        output = _run(
            ["nvidia-smi", "nvlink", "--status", "-i", device_index],
            timeout_seconds=_remaining(deadline),
        )
        active_links = len(re.findall(r"\bLink\s+\d+:\s+[0-9.]+\s+GB/s", output, flags=re.IGNORECASE))
        inactive_links = len(re.findall(r"\bLink\s+\d+:\s+Inactive\b", output, flags=re.IGNORECASE))
        if active_links == 0:
            raise SystemExit(f"GPU preflight found no active NVLink links for GPU {device_index}")
        if inactive_links:
            raise SystemExit(
                f"GPU preflight found {inactive_links} inactive NVLink link(s) for GPU {device_index}"
            )


def _distributed_worker(expected_world_size: int) -> None:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected_world_size:
        raise SystemExit(f"GPU preflight world size mismatch: expected {expected_world_size}, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        value = torch.tensor([float(dist.get_rank() + 1)], device="cuda")
        dist.all_reduce(value)
        expected = float(world_size * (world_size + 1) // 2)
        if value.item() != expected:
            raise SystemExit(f"GPU preflight NCCL all-reduce mismatch: expected {expected}, got {value.item()}")
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def _check_torch_distributed(*, expected_gpu_count: int, deadline: float) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={expected_gpu_count}",
        "--module",
        "pipeline.gpu_preflight",
        "--distributed-worker",
        "--expected-world-size",
        str(expected_gpu_count),
    ]
    _run(command, timeout_seconds=_remaining(deadline))


def run_preflight(
    *,
    required_gpu_count: int,
    required_gpu_name: str,
    check_ecc: bool,
    check_nvlink: bool,
    check_torch_distributed: bool,
    timeout_seconds: int,
) -> None:
    if required_gpu_count <= 0:
        raise SystemExit("GPU preflight required_gpu_count must be positive")
    if timeout_seconds <= 0:
        raise SystemExit("GPU preflight timeout_seconds must be positive")

    deadline = time.monotonic() + timeout_seconds
    gpu_names = _visible_torch_gpu_names()
    if len(gpu_names) != required_gpu_count:
        raise SystemExit(f"GPU preflight expected {required_gpu_count} GPU(s), found {len(gpu_names)}")

    required_name = required_gpu_name.strip().lower()
    for gpu_name in gpu_names:
        if required_name and required_name not in gpu_name.lower():
            raise SystemExit(f"GPU preflight expected GPU name containing {required_gpu_name!r}, found {gpu_name!r}")

    nvidia_rows: list[list[str]] = []
    if check_ecc or check_nvlink:
        nvidia_rows = _allocated_nvidia_rows(deadline=deadline, expected_gpu_count=required_gpu_count)
    if check_ecc:
        for row in nvidia_rows:
            try:
                uncorrected_errors = int(row[2])
            except (IndexError, ValueError) as exc:
                raise SystemExit(f"GPU preflight could not parse ECC count from row: {row}") from exc
            if uncorrected_errors != 0:
                raise SystemExit(
                    f"GPU preflight found {uncorrected_errors} volatile uncorrected ECC error(s) on GPU {row[0]}"
                )

    if check_nvlink:
        _check_nvlink(nvidia_rows=nvidia_rows, deadline=deadline)
    if check_torch_distributed:
        _check_torch_distributed(expected_gpu_count=required_gpu_count, deadline=deadline)
    print(
        "[GPU_PREFLIGHT_OK] "
        f"gpus={required_gpu_count} name={required_gpu_name or 'any'} "
        f"ecc={check_ecc} nvlink={check_nvlink} distributed={check_torch_distributed}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-gpu-count", type=int, default=1)
    parser.add_argument("--required-gpu-name", default="")
    parser.add_argument("--check-ecc", action="store_true")
    parser.add_argument("--check-nvlink", action="store_true")
    parser.add_argument("--check-torch-distributed", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--distributed-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-world-size", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.distributed_worker:
        _distributed_worker(args.expected_world_size)
        return
    run_preflight(
        required_gpu_count=args.required_gpu_count,
        required_gpu_name=args.required_gpu_name,
        check_ecc=args.check_ecc,
        check_nvlink=args.check_nvlink,
        check_torch_distributed=args.check_torch_distributed,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
