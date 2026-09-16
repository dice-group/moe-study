"""Optional, job-local byte-identical model copy for repeated GPU diagnostics."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
from pathlib import Path
import shutil

from pipeline_tools.cross_arch_study import contract, model_root, read_json, write_json, sha256


def cache_root(model):
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or model not in contract()["models"]:
        raise ValueError("Model cache requires a known model and a Slurm allocation")
    return Path(f"/dev/shm/moe-study-model-{job}-{model}")


def local_model_path(model):
    original = model_root(model) / "model"
    if not os.environ.get("SLURM_JOB_ID"):
        return original
    root = cache_root(model)
    if not (root / "ready.json").exists():
        return original
    ready = read_json(root / "ready.json")
    assets_path = model_root(model) / "assets.json"
    if (root.stat().st_uid != os.getuid() or not ready.get("pass")
            or ready["assets_sha256"] != sha256(assets_path)
            or ready["job_id"] != os.environ["SLURM_JOB_ID"]):
        raise ValueError("Stale or foreign model cache")
    assets = read_json(assets_path)
    for name, info in assets["files"].items():
        path = root / "model" / name
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Incomplete model cache: {name}")
    return root / "model"


def stage(model):
    root = cache_root(model)
    if (root / "ready.json").exists():
        print(local_model_path(model), flush=True)
        return
    root.mkdir(mode=0o700, exist_ok=False)
    destination = root / "model"
    destination.mkdir()
    source = model_root(model) / "model"
    assets_path = model_root(model) / "assets.json"
    assets = read_json(assets_path)
    if not assets["pass"] or assets["revision"] != contract()["models"][model]["revision"]:
        raise ValueError("Unvalidated model assets")

    def copy_verified(item):
        name, info = item
        if Path(name).name != name:
            raise ValueError("Unexpected model asset path")
        target = destination / name
        shutil.copyfile(source / name, target)
        if target.stat().st_size != info["bytes"] or sha256(target) != info["sha256"]:
            raise ValueError(f"Model cache byte verification failed: {name}")
        target.chmod(0o444)
        print(f"[MODEL_CACHE] verified {model}/{name}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy_verified, assets["files"].items()))
    write_json(root / "ready.json", {"pass": True, "job_id": os.environ["SLURM_JOB_ID"],
        "assets_sha256": sha256(assets_path), "canonical_base": str(source)})
    print(local_model_path(model), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(contract()["models"]))
    stage(parser.parse_args().model)
