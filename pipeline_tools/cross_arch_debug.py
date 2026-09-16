"""Per-configuration diagnostics inside an existing four-A100 allocation.

Short diagnostics never write promotion gates. Full 120-step calibration and
its independent confirmation remain required before production submission.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import socket
import subprocess

from pipeline_tools.cross_arch_study import (
    ROOT, PYTHON, campaign_root, model_root, contract, read_json, write_json,
    runtime_pythonpath, utc_now, sha256,
)
from pipeline_tools.cross_arch_pilot import (
    load_profile, find_run, code_signature, benchmark_command, smoke_panel, primary_values, BENCHMARK_CWD,
)


def attempt(model, run_id, label, batch=None, steps=3, reload=True, enable_gc=False, non_reentrant_gc=False):
    print(f"[DEBUG] Preparing {model}/{run_id} ({label})", flush=True)
    from pipeline.training import build_training_cmd, adapter_checkpoint_provenance
    from pipeline.executor import LocalExecutor
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("Invalid attempt label")
    if steps < 3 or steps > 120:
        raise ValueError("Diagnostics require 3-120 steps, not a shortened LR schedule")
    profile = load_profile(model)
    run = find_run(profile, run_id)
    if batch is not None:
        if batch not in (1, 2, 3, 4, 6, 12):
            raise ValueError("Batch must divide global batch 48 on four GPUs")
        run["train"].update(batch_size=batch, grad_accum=12 // batch)
    if enable_gc:
        run["train"].update(disable_gradient_checkpointing=False)
    if non_reentrant_gc:
        run["train"].update(disable_gradient_checkpointing=False, use_reentrant_gc=False)
    run["train"].update(logging_steps=1, flash_attn=read_json(model_root(model) / "backend_ready.json")["selected_backend"])
    out = model_root(model) / "debug" / label / run_id
    if out.exists():
        raise ValueError(f"Use a new label; diagnostic evidence is preserved: {out}")
    out.mkdir(parents=True)
    env = dict(os.environ)
    from pipeline_tools.cross_arch_study import training_runtime_environment
    env.update(training_runtime_environment(model, run_id))
    env.update(PYTHONPATH=runtime_pythonpath(model), PROOF_PIPELINE_CROSS_ARCH_STOP_STEPS=str(steps),
               PROOF_PIPELINE_PREFLIGHT_COMPLETE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false")
    cmd, env = build_training_cmd(repo_root=ROOT, env=env, output_dir=out, run_cfg=run,
        common=profile["training"], language_map_path=campaign_root() / "inputs/language_map.json",
        tokenized_path=model_root(model) / "tokenized", embedding_training_cfg={"enabled": False},
        run_name=f"{model}/{run_id}/debug/{label}")
    # Same Accelerate launch parser/implementation, without importing unused
    # CLI subcommands (model-size estimation, TPU conversion, etc.).
    cmd[:2] = [str(PYTHON), "-m", "accelerate.commands.launch"]
    # Keep the verified job-scoped RAM copy across retries in this allocation.
    # It lives in /dev/shm, and is never used by another Slurm job.
    env["PROOF_PIPELINE_TOKENIZED_CLEANUP"] = "0"
    print(f"[DEBUG] Hashing source for {model}/{run_id}", flush=True)
    result = {"pass": False, "debug_only": True, "model": model, "run_id": run_id,
              "reload_and_benchmark_requested": reload,
              "steps": steps, "job_id": os.environ["SLURM_JOB_ID"], "command": cmd,
              "run_config": run, "code_signature": code_signature(), "started_utc": utc_now()}
    write_json(out / "diagnostic.json", result)
    try:
        print(f"[DEBUG] Training {model}/{run_id}; log: {out / 'train.log'}", flush=True)
        LocalExecutor().run(cmd, env, False, out / "train.log")
        telemetry = read_json(out / "campaign_training.json")
        trained = read_json(out / "train_results.json")
        checkpoint = out / f"checkpoint-{steps}_adapter"
        provenance = adapter_checkpoint_provenance(checkpoint, expected_step=steps, require_direct_save_marker=True)
        if (telemetry["global_step"] != steps or telemetry["gradient_checks"] < 1
                or not telemetry["signal_checkpoint"] or not provenance["valid"]
                or not math.isfinite(trained["train_loss"])):
            raise ValueError("Forward/backward/optimizer/signal/checkpoint diagnostic failed")
        result.update(telemetry=telemetry, checkpoint=str(checkpoint), train_loss=trained["train_loss"])
        if reload:
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["LM_EVAL_ATTN_IMPL"] = {"disabled": "eager", "sdpa": "sdpa", "fa2": "flash_attention_2"}[run["train"]["flash_attn"]]
            with (out / "reload.log").open("w") as log:
                subprocess.run([str(PYTHON), "-m", "pipeline_tools.cross_arch_verify", "adapter",
                    "--model", model, "--checkpoint", str(checkpoint)], cwd=ROOT, env=env,
                    check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            tasks = smoke_panel((campaign_root() / "inputs/tasks.txt").read_text().splitlines())
            smoke = out / "benchmark_smoke"
            smoke.mkdir()
            (smoke / "tasks.txt").write_text("\n".join(tasks) + "\n")
            with (out / "benchmark.log").open("w") as log:
                subprocess.run(benchmark_command(model, run_id, checkpoint, smoke / "tasks.txt", smoke, limit=2),
                    cwd=BENCHMARK_CWD, env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            result["reload"] = read_json(out / "reload_check.json")
            result["benchmark_smoke"] = primary_values(smoke, tasks)
        result["pass"] = True
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    result["finished_utc"] = utc_now()
    write_json(out / "diagnostic.json", result)
    print({k: result.get(k) for k in ("model", "run_id", "pass", "reason", "train_loss")}, flush=True)
    return result


def main():
    import faulthandler
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    faulthandler.dump_traceback_later(180, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(contract()["models"]))
    parser.add_argument("--run-id", choices=contract()["run_ids"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--enable-gc", action="store_true", help="Enable checkpointing with the configured implementation; scientific settings stay fixed")
    parser.add_argument("--non-reentrant-gc", action="store_true", help="Explicitly change checkpoint implementation as a separate systems probe")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use an existing salloc/srun allocation")
    # Reuse a successful hardware/runtime preflight only within this exact live
    # allocation. Every production allocation still performs its own preflight.
    preflight_path = model_root(args.model) / "debug" / f"preflight_{os.environ['SLURM_JOB_ID']}.json"
    expected = {"job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
                "preflight_sha256": sha256(ROOT / "pipeline/gpu_preflight.py")}
    prior = read_json(preflight_path) if preflight_path.exists() else {}
    if not prior.get("pass") or any(prior.get(k) != v for k, v in expected.items()):
        check = subprocess.run([str(PYTHON), "-m", "pipeline.gpu_preflight", "--required-gpu-count", "4",
            "--required-gpu-name", "A100", "--check-ecc", "--check-nvlink", "--check-torch-distributed"],
            cwd=ROOT, check=True, timeout=330, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(check.stdout, flush=True)
        write_json(preflight_path, {**expected, "pass": True, "output": check.stdout, "checked_utc": utc_now()})
    else:
        print(f"[DEBUG] Reusing preflight for current allocation {expected['job_id']}", flush=True)
    runs = [args.run_id] if args.run_id else ["lora_r2", *[r for r in contract()["run_ids"] if r != "lora_r2"]]
    results = [attempt(args.model, run, args.label, args.batch, args.steps, not args.training_only, args.enable_gc, args.non_reentrant_gc) for run in runs]
    faulthandler.cancel_dump_traceback_later()
    if not all(r["pass"] for r in results):
        raise SystemExit("Some diagnostics failed; inspect individual logs. Production remains held.")


if __name__ == "__main__":
    main()
