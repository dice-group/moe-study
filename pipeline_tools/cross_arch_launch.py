"""Fail-closed production submission after each real four-A100 pilot passes.

`advance --submit` submits only validated configurations, never a failed pilot.
`train_acceptance` is an afterany audit; evaluations require its successful exit.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess

from pipeline_tools.cross_arch_study import (
    ROOT, PYTHON, CONDA_BASE, CONTRACT, campaign_root, model_root, contract,
    read_json, write_json, sha256, capacity_check, sbatch, utc_now, runtime_pythonpath,
    runtime_profile, training_runtime_environment, attempt_logs,
)
from pipeline_tools.cross_arch_pilot import code_signature, load_profile, find_run


def validate_selected_run(original, selected, backend):
    """Only measured systems knobs may differ from the frozen scientific run."""
    expected, observed = copy.deepcopy(original), copy.deepcopy(selected)
    for key in ("batch_size", "grad_accum", "disable_gradient_checkpointing", "use_reentrant_gc", "flash_attn", "logging_steps"):
        expected["train"].pop(key, None)
        observed["train"].pop(key, None)
    if expected != observed:
        raise ValueError("Pilot changed scientific settings outside the permitted systems knobs")
    train = selected["train"]
    if train["batch_size"] not in (1, 2, 3, 4, 6, 12) or 4 * train["batch_size"] * train["grad_accum"] != 48:
        raise ValueError("Microbatch or effective batch differs from the contract")
    if train["flash_attn"] != backend:
        raise ValueError("Unvalidated attention backend")


def validate_gate(model, run_id):
    root, out = campaign_root(), model_root(model)
    if read_json(root / "manifest.json")["contract_sha256"] != sha256(CONTRACT):
        raise ValueError("Frozen study contract changed")
    gate = read_json(out / "gates" / f"{run_id}.json")
    if not gate.get("pass") or gate.get("model") != model or gate.get("run_id") != run_id:
        raise ValueError(f"No successful pilot gate for {model}/{run_id}")
    expected_files = {
        "pipeline_config_sha256": root / "pipeline_config.json",
        "data_sha256": out / "data_ready.json",
        "backend_sha256": out / "backend_ready.json",
    }
    if gate.get("code_signature") != code_signature() or any(gate.get(key) != sha256(path) for key, path in expected_files.items()):
        raise ValueError("Pilot evidence is stale: campaign code/config/data/backend changed")
    backend, data = read_json(out / "backend_ready.json"), read_json(out / "data_ready.json")
    if not backend["pass"] or not data["pass"] or not data["document_assignment_equal"]:
        raise ValueError("Backend/data validation failed")
    if backend["revision"] != contract()["models"][model]["revision"]:
        raise ValueError("Wrong model revision")
    if backend["selected_backend"] == "fa2" and backend.get("flash_attention_sha256") != sha256(backend["flash_attention_binary"]):
        raise ValueError("Validated FlashAttention binary changed")
    if not gate["capacity"]["pass"] or not capacity_check(data["train_sequences"], max(gate["confirmed_steps_per_second"]))["pass"]:
        raise ValueError("Dataset exhaustion risk; user decision required, no repetition permitted")
    if not gate["reload"]["pass"] or len(gate["benchmark_smoke"]) != 5:
        raise ValueError("Adapter reload or benchmark smoke test failed")
    validate_selected_run(find_run(load_profile(model), run_id), gate["selected_run"], backend["selected_backend"])
    return gate


def clean_environment(model, run_id):
    if os.environ.get("HALA_MAX_STEPS_OVERRIDE") or os.environ.get("PROOF_PIPELINE_CROSS_ARCH_STOP_STEPS"):
        raise ValueError("Pilot/max-step override present in production environment")
    env = dict(os.environ)
    env.update(training_runtime_environment(model, run_id))
    env.update(PYTHONPATH=runtime_pythonpath(model),
               PROOF_PIPELINE_REPO_ROOT=str(ROOT), PROOF_PIPELINE_PYTHON=str(PYTHON),
               PROOF_PIPELINE_PRESERVE_ENV="1", WANDB_MODE="disabled",
               HF_HOME=runtime_profile()["hf_home"],
               HF_DATASETS_CACHE=str(Path(runtime_profile()["hf_home"]) / "datasets"),
               PROOF_PIPELINE_CROSS_ARCH_MODEL=model, PROOF_PIPELINE_CROSS_ARCH_RUN=run_id,
               PROOF_PIPELINE_CROSS_ARCH_PRODUCTION="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return env


def production_run(model, run_id):
    gate = validate_gate(model, run_id)
    run = copy.deepcopy(gate["selected_run"])
    run["train"]["logging_steps"] = find_run(load_profile(model), run_id)["train"]["logging_steps"]
    return run


def advance(model=None, run_id=None, submit=False):
    """Idempotent submission; do not resubmit failed or partially recorded jobs."""
    from pipeline_tools.cross_arch_node import CrossArchSlurmExecutor, excluded_nodes
    from pipeline.training import build_training_cmd
    root = campaign_root()
    with (root / ".production_submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = root / "production_jobs.json"
        record = read_json(receipt) if receipt.exists() else {"jobs": [], "created_utc": utc_now()}
        for m in ([model] if model else contract()["models"]):
            for rid in ([run_id] if run_id else contract()["run_ids"]):
                if any(j["model"] == m and j["run_id"] == rid for j in record["jobs"]):
                    print(f"Already recorded: {m}/{rid}; no duplicate submission")
                    continue
                gate_path = model_root(m) / "gates" / f"{rid}.json"
                if not gate_path.exists() or not read_json(gate_path).get("pass"):
                    print(f"HELD: {m}/{rid}: pilot not passed")
                    continue
                run, profile, out = production_run(m, rid), load_profile(m), model_root(m)
                destination = out / "runs" / rid
                if destination.exists() and any(destination.iterdir()):
                    raise ValueError(f"Existing production output requires inspection: {destination}")
                env = clean_environment(m, rid)
                cmd, env = build_training_cmd(repo_root=ROOT, env=env, output_dir=destination,
                    run_cfg=run, common=profile["training"], language_map_path=root / "inputs/language_map.json",
                    tokenized_path=out / "tokenized", embedding_training_cfg={"enabled": False}, run_name=f"{m}/{rid}")
                cmd[0] = str(PYTHON.parent / "accelerate")
                runtime = runtime_profile()
                executor = CrossArchSlurmExecutor(partition=runtime["gpu_partition"], gres=f"gpu:{runtime['gpu_type']}:4", **runtime["resources"]["train"],
                    time_limit=contract()["slurm_time"], modules=runtime["modules"], conda_base=CONDA_BASE, conda_env=str(PYTHON.parent.parent),
                    extra_sbatch_flags=["--nodes=1", "--ntasks-per-node=1", "--signal=B:USR1@1200", "--kill-on-invalid-dep=yes",
                                        "--exclude=" + ",".join(excluded_nodes())],
                    preflight={"enabled": True, "required_gpu_count": 4, "required_gpu_name": "A100",
                               "timeout_seconds": 300, "check_ecc": True, "check_nvlink": True, "check_torch_distributed": True})
                log = attempt_logs(out) / f"production_{rid}.log"
                slurm_log = log.with_suffix(".slurm.log")
                if submit:
                    destination.mkdir(parents=True, exist_ok=True)
                    write_json(destination / "frozen_run.json", run)
                train_id = executor.submit(cmd, env, not submit, log, job_name=f"xa96_{m}_{rid}", slurm_output=slurm_log)
                job = {"model": m, "run_id": rid, "train": train_id, "train_log": str(slurm_log), "status": "submitting"}
                if submit:
                    record["jobs"].append(job)
                    write_json(receipt, record)
                accept_id = sbatch([str(PYTHON), "-m", "pipeline_tools.cross_arch_launch", "accept", "--model", m, "--run-id", rid],
                    f"xa_accept_{m}_{rid}", dependency=f"afterany:{train_id}" if submit else None, submit=submit,
                    hours="00:20:00", extra_env={"PROOF_PIPELINE_TRAIN_JOB_ID": str(train_id), "PROOF_PIPELINE_TRAIN_LOG_PATH": str(slurm_log)})
                job.update(acceptance=accept_id, status="submitted")
                if submit:
                    write_json(receipt, record)
                    from pipeline_tools.cross_arch_eval import submit_evaluation
                    from pipeline_tools.cross_arch_report import submit_ready_reports
                    job["evaluation"] = submit_evaluation(m, rid, f"afterok:{accept_id}", submit=True)
                    write_json(receipt, record)
                    submit_ready_reports(submit=True)
                print(json.dumps(job), flush=True)


def accept(model, run_id):
    from pipeline.stages.train_acceptance import evaluate_training_run
    validate_gate(model, run_id)
    out = model_root(model)
    destination = out / "runs" / run_id
    decision = evaluate_training_run(run_id=run_id, run_dir=destination, config=load_profile(model)["train_acceptance"],
        log_paths=[Path(os.environ["PROOF_PIPELINE_TRAIN_LOG_PATH"])], train_job_id=os.environ["PROOF_PIPELINE_TRAIN_JOB_ID"])
    telemetry = read_json(destination / "campaign_training.json")
    if telemetry.get("pilot") or not telemetry["signal_checkpoint"] or telemetry["epoch"] >= 1 or telemetry["gradient_checks"] < 2:
        decision["accepted"] = False
        decision["reasons"].append("Campaign telemetry: pilot, missing gradient/signal checks, or exhausted epoch")
    write_json(out / "acceptance" / run_id / "acceptance.json", decision)
    print(json.dumps(decision, indent=2), flush=True)
    if not decision["accepted"]:
        raise ValueError("Training acceptance failed; evaluations remain held")


def arm(submit=False):
    """Queue short gate-checkers; they may launch production only after a pilot."""
    root = campaign_root()
    path = root / "promotion_jobs.json"
    if submit and path.exists():
        raise ValueError("Promotion watchers already recorded; inspect before retrying")
    pilots = read_json(root / "pilot_jobs.json")["jobs"]
    record = {"jobs": []}
    for pilot in pilots:
        model, run_id = pilot["model"], pilot["run_id"]
        job_id = sbatch([str(PYTHON), "-m", "pipeline_tools.cross_arch_launch", "advance",
                        "--model", model, "--run-id", run_id, "--submit"],
                       f"xa_promote_{model}_{run_id}", hours="00:20:00", submit=submit,
                       dependency=f"afterok:{pilot['job_id']}")
        record["jobs"].append({"model": model, "run_id": run_id, "job_id": job_id})
        if submit: write_json(path, record)
    # The native baselines are evaluations, not additional training runs.
    from pipeline_tools.cross_arch_eval import submit_evaluation
    for model in contract()["models"]:
        first = next(j["job_id"] for j in pilots if j["model"] == model and j["run_id"] == "lora_r2")
        submit_evaluation(model, "baseline", f"afterok:{first}", submit=submit)
    if submit:
        from pipeline_tools.cross_arch_report import submit_ready_reports
        submit_ready_reports(submit=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["advance", "accept", "arm"])
    parser.add_argument("--model", choices=list(contract()["models"]))
    parser.add_argument("--run-id", choices=contract()["run_ids"])
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.action == "advance": advance(args.model, args.run_id, args.submit)
    elif args.action == "arm": arm(args.submit)
    else:
        if not args.model or not args.run_id: parser.error("--model and --run-id required")
        accept(args.model, args.run_id)


if __name__ == "__main__":
    main()
