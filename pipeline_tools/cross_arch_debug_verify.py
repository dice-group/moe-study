"""Reload and five-task smoke checks of one successful diagnostic checkpoint."""
import argparse
import os
import subprocess

from pipeline_tools.cross_arch_study import PYTHON, ROOT, campaign_root, model_root, read_json, write_json, utc_now, checkpoint_weights, sha256
from pipeline_tools.cross_arch_pilot import benchmark_command, smoke_panel, primary_values, code_signature, BENCHMARK_CWD


def verify(model, run_id, training_label, label):
    import re
    for value in (training_label, label, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Invalid diagnostic identifier")
    prior = read_json(model_root(model) / "debug" / training_label / run_id / "diagnostic.json")
    if not prior["pass"] or not prior["debug_only"]:
        raise ValueError("A successful training/checkpoint diagnostic is required")
    from pathlib import Path
    checkpoint = Path(prior["checkpoint"])
    out = model_root(model) / "debug" / label / run_id
    out.mkdir(parents=True, exist_ok=False)
    result = {"pass": False, "debug_only": True, "model": model, "run_id": run_id,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint_weights(checkpoint)),
              "code_signature": code_signature(), "started_utc": utc_now()}
    write_json(out / "validation.json", result)
    env = dict(os.environ)
    env["PROOF_PIPELINE_CROSS_ARCH_MODEL"] = model
    # Preserve the GPU mask assigned by srun: cuda:0 is the allocated GPU.
    backend = read_json(model_root(model) / "backend_ready.json")["selected_backend"]
    env["LM_EVAL_ATTN_IMPL"] = {"disabled": "eager", "sdpa": "sdpa", "fa2": "flash_attention_2"}[backend]
    try:
        with (out / "reload.log").open("w") as log:
            subprocess.run([str(PYTHON), "-m", "pipeline_tools.cross_arch_verify", "adapter", "--model", model,
                "--checkpoint", str(checkpoint)], cwd=ROOT, env=env, check=True,
                stdout=log, stderr=subprocess.STDOUT, timeout=900)
        result["reload"] = read_json(checkpoint.parent / "reload_check.json")
        tasks = smoke_panel((campaign_root() / "inputs/tasks.txt").read_text().splitlines())
        smoke = out / "benchmark_smoke"
        smoke.mkdir()
        tasks_path = smoke / "tasks.txt"
        tasks_path.write_text("\n".join(tasks) + "\n")
        with (out / "benchmark.log").open("w") as log:
            subprocess.run(benchmark_command(model, run_id, checkpoint, tasks_path, smoke, limit=2),
                cwd=BENCHMARK_CWD, env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        result["benchmark_smoke"] = primary_values(smoke, tasks)
        result["pass"] = bool(result["reload"]["pass"])
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    result["finished_utc"] = utc_now()
    write_json(out / "validation.json", result)
    print({k: result.get(k) for k in ("model", "run_id", "pass", "reason")}, flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "run-id", "training-label", "label"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    verify(args.model, args.run_id, args.training_label, args.label)
