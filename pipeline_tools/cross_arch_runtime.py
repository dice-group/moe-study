"""Build the same FlashAttention release for cluster glibc, without changing Merlin."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request

from pipeline_tools.cross_arch_study import PYTHON, campaign_root, model_root, sha256, read_json, write_json, utc_now, sbatch, runtime_profile


def build():
    root = campaign_root() / "runtime/flash_attn_2_8_3_sm80"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "build_ready.json").exists():
        raise ValueError("Runtime build already has a receipt; inspect before retrying")
    env = dict(os.environ)
    # Conda currently contains CUDA 13 nvcc: never infer the compiler from PATH.
    software = Path(runtime_profile()["software_root"])
    cuda = software / "CUDA/12.6.0"
    gcc = software / "GCCcore/13.3.0/bin"
    ninja = software / "Ninja/1.12.1-GCCcore-13.3.0/bin"
    env["LD_LIBRARY_PATH"] = str(gcc.parent / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    for binary in (cuda / "bin/nvcc", gcc / "g++", ninja / "ninja"):
        if not binary.is_file(): raise ValueError(f"Required build tool is absent: {binary}")
        subprocess.run([str(binary), "--version"], check=True, env=env)
    temporary = root / "tmp"
    temporary.mkdir(exist_ok=True)
    env.update(FLASH_ATTENTION_FORCE_BUILD="TRUE", FLASH_ATTN_CUDA_ARCHS="80", TORCH_CUDA_ARCH_LIST="8.0",
               MAX_JOBS="4", NVCC_THREADS="2", CUDA_HOME=str(cuda), TMPDIR=str(temporary),
               CC=str(gcc / "gcc"), CXX=str(gcc / "g++"),
               PATH=os.pathsep.join([str(ninja), str(cuda / "bin"), str(gcc), env["PATH"]]))
    with urllib.request.urlopen("https://pypi.org/pypi/flash-attn/2.8.3/json", timeout=60) as handle:
        metadata = json.load(handle)
    sources = [item for item in metadata["urls"] if item["packagetype"] == "sdist"]
    if len(sources) != 1: raise ValueError("Ambiguous source release")
    source = sources[0]
    archive = root / source["filename"]
    urllib.request.urlretrieve(source["url"], archive)
    if sha256(archive) != source["digests"]["sha256"]:
        raise ValueError("Source archive checksum mismatch")
    wheels = root / "wheels"
    wheels.mkdir(exist_ok=True)
    subprocess.run([str(PYTHON), "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir",
                    "--wheel-dir", str(wheels), str(archive)], env=env, check=True, timeout=7000)
    built = list(wheels.glob("flash_attn-2.8.3-*.whl"))
    if len(built) != 1: raise ValueError("Expected exactly one freshly built wheel")
    overlay = root / "site-packages"
    subprocess.run([str(PYTHON), "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(overlay), str(built[0])], check=True, env=env)
    probe_env = dict(env, PYTHONPATH=str(overlay) + os.pathsep + env.get("PYTHONPATH", ""))
    subprocess.run([str(PYTHON), "-c", "import torch,flash_attn,flash_attn_2_cuda; print(flash_attn.__file__, flash_attn_2_cuda.__file__)"], check=True, env=probe_env)
    write_json(root / "build_ready.json", {"pass": True, "version": "2.8.3", "cuda_arch": "80", "glibc": platform.libc_ver(),
        "source_url": source["url"], "source_sha256": sha256(archive), "wheel_sha256": sha256(built[0]),
        "files": {str(p.relative_to(overlay)): sha256(p) for p in overlay.glob("*.so")}, "utc": utc_now()})


def submit(retry=False):
    root = campaign_root()
    receipt = root / "runtime_jobs.json"
    history = read_json(receipt) if receipt.exists() else None
    if history and not retry: raise ValueError("Runtime repair already recorded; use explicit --retry only after inspection")
    if history:
        for job_id in (history["build"], history["backend_recheck"]):
            state = subprocess.check_output(["squeue", "-h", "-j", job_id, "-o", "%T"], text=True).strip()
            if state == "PENDING": subprocess.run(["scancel", job_id], check=True)
            elif state: raise ValueError(f"Do not duplicate active runtime job {job_id}: {state}")
    previous = read_json(root / "backend_jobs.json")
    write_json(model_root("gemma2_9b") / "backend_before_runtime_repair.json", read_json(model_root("gemma2_9b") / "backend_ready.json"))
    label = "xa_flashattn_build_" + utc_now().replace(":", "").split(".")[0] if retry else "xa_flashattn_build"
    job = sbatch([str(PYTHON), "-m", "pipeline_tools.cross_arch_runtime", "build"], label, hours="02:00:00", submit=True)
    write_json(receipt, {"build": job, "status": "submitting"})
    # Even a failed build may use the already-proven eager fallback; record why.
    probe = sbatch([str(PYTHON), "-m", "pipeline_tools.cross_arch_verify", "backend", "--model", "gemma2_9b"],
                    "xa_backend_gemma_recheck", gpu=True, gpu_count=1, hours="00:30:00", dependency=f"afterany:{job}", submit=True)
    previous.setdefault("history", []).extend(j for j in previous["jobs"] if j["model"] == "gemma2_9b")
    previous["jobs"] = [j for j in previous["jobs"] if j["model"] != "gemma2_9b"] + [{"model": "gemma2_9b", "job_id": probe}]
    write_json(root / "backend_jobs.json", previous)
    write_json(receipt, {"build": job, "backend_recheck": probe, "status": "submitted", "previous_attempt": history})
    print(json.dumps(read_json(receipt)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "submit"])
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    if args.action == "build": build()
    else: submit(args.retry)


if __name__ == "__main__":
    main()
