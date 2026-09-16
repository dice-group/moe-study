"""New architecture workbooks and common-seven ranking correlations.

Never modifies the submitted Llama workbooks or analysis notebook.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
from itertools import combinations
import os
from pathlib import Path

import pandas as pd

from pipeline_tools.cross_arch_study import ROOT, campaign_root, model_root, contract, read_json, write_json, sha256, sbatch, PYTHON
from pipeline_tools.cross_arch_pilot import load_profile

BENCHMARKS = ["Belebele", "FLORES", "SIB200", "Taxi1500"]
LLAMA_WORKBOOK = ROOT / "result_analysis/reports/moe_study/moe_study_final_96_languages_2026-07-25.xlsx"


def primary_data(raw, run_ids=None):
    mask = ((raw.benchmark.eq("Belebele") & raw.metric.eq("acc_norm"))
            | (raw.benchmark.eq("FLORES") & raw.metric.str.startswith("bits_per_byte "))
            | (raw.benchmark.isin(["SIB200", "Taxi1500"]) & raw.metric.eq("acc")))
    data = raw[mask].copy()
    if run_ids is not None:
        data = data[data.run_id.isin(run_ids)].copy()
    data["value"] = pd.to_numeric(data.value, errors="raise")
    data["score"] = data.value.where(data.better.ne("lower"), -data.value)
    data["score_family"] = data.benchmark
    flores = data.benchmark.eq("FLORES")
    data.loc[flores, "score_family"] = "FLORES " + data.loc[flores, "metric"].str.extract(r"(en->x|x->en)", expand=False)
    return data


def macro_ranks(data):
    """Rank first; FLORES directions 50/50, then benchmarks 25% each."""
    import numpy as np
    pivot = data.pivot(index=["score_family", "task", "language", "metric"], columns="run_id", values="score")
    if pivot.empty or not np.isfinite(pivot.to_numpy()).all():
        raise ValueError("Missing or nonfinite paired benchmark panel")
    unit_ranks = pivot.rank(axis=1, ascending=False, method="average")
    family_ranks = unit_ranks.groupby(level="score_family").mean()
    family_ranks.loc["FLORES"] = family_ranks.loc[["FLORES en->x", "FLORES x->en"]].mean()
    benchmark_ranks = family_ranks.loc[BENCHMARKS].T
    result = benchmark_ranks.copy()
    result["macro_avg_rank"] = result.mean(axis=1)
    return result.sort_values("macro_avg_rank"), benchmark_ranks


def check_panel(data, run_ids, tasks):
    if set(data.run_id) != set(run_ids):
        raise ValueError("Final comparison is missing a configuration or includes an unexpected one")
    for run_id, rows in data.groupby("run_id"):
        if len(rows) != len(tasks) or set(rows.task) != set(tasks):
            raise ValueError(f"Incomplete/duplicated primary panel for {run_id}")


def build_report(model):
    from pipeline_tools.reporting.workbook import build_workbook
    from pipeline_tools.reporting.sources import iter_eval_rows, load_baseline_rows_from_lm_eval_json, load_resource_map
    from pipeline_tools.reporting.artifacts import collect_benchmark_raw_rows
    from openpyxl import load_workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    out, root, profile = model_root(model), campaign_root(), load_profile(model)
    tasks = (root / "inputs/tasks.txt").read_text().splitlines()
    summary, provenance, benchmark_summary = [], [], []
    for run in profile["training"]["runs"]:
        run_id, folder = run["id"], out / "runs" / run["id"]
        accepted = read_json(out / "acceptance" / run_id / "acceptance.json")
        complete = read_json(out / "benchmarks" / run_id / "complete.json")
        if not accepted["accepted"] or not complete["pass"] or complete["results_sha256"] != sha256(out / "benchmarks" / run_id / "no_language_ids.json"):
            raise ValueError(f"Missing final accepted results for {run_id}")
        telemetry, trained = read_json(folder / "campaign_training.json"), read_json(folder / "train_results.json")
        loss = read_json(folder / "final_eval_loss/complete.json")
        if not loss["pass"] or loss["adapter_sha256"] != complete["adapter_sha256"]:
            raise ValueError("Benchmark and validation loss refer to different checkpoints")
        summary.append({"run_id": run_id, "kind": run["kind"], "output_dir": str(folder), "status": "accepted",
                        "train_loss": trained["train_loss"], "train_runtime": trained["train_runtime"], "eval_loss": loss["eval_loss"],
                        "trainable_params": telemetry["trainable_parameters"], "seed": 42})
        provenance.append({"run_id": run_id, "accepted": True, "primary_units": len(tasks),
                           "adapter_sha256": complete["adapter_sha256"], "train_runtime": trained["train_runtime"],
                           "trainable_parameters": telemetry["trainable_parameters"]})
        benchmark_summary.append({"run_id": run_id, "evaluated_checkpoint": loss["checkpoint"],
                                  "num_tasks_expected": len(tasks), "num_tasks_seen": len(tasks), "missing_tasks": []})
    baseline = out / "baseline"
    baseline_receipt = read_json(baseline / "complete.json")
    if not baseline_receipt["pass"] or baseline_receipt["results_sha256"] != sha256(baseline / "no_language_ids.json"):
        raise ValueError("Native baseline missing or invalid")
    profile["training"]["runs"] = [read_json(out / "runs" / rid / "frozen_run.json") for rid in contract()["run_ids"]]
    eval_rows = iter_eval_rows(out / "benchmarks")
    baseline_rows = load_baseline_rows_from_lm_eval_json(baseline / "no_language_ids.json", contract()["models"][model]["repo_id"])
    resource_map = load_resource_map(ROOT / "result_analysis/l2v_nested_samples/fineweb2-language-distribution.csv")
    raw = pd.DataFrame(collect_benchmark_raw_rows(eval_rows, baseline_rows, summary, resource_map))
    data = primary_data(raw)
    baseline_ids = set(data.loc[data.is_baseline, "run_id"])
    if len(baseline_ids) != 1:
        raise ValueError("Exactly one model-specific baseline is required")
    check_panel(data, [*contract()["run_ids"], *baseline_ids], tasks)
    ranking, per_benchmark = macro_ranks(data)
    write_json(out / "benchmarks/benchmark_summary.json", {"strict_completeness": True, "runs": benchmark_summary,
        "tasks_file": str(root / "inputs/tasks.txt"), "num_selected_tasks": len(tasks)})
    date = dt.datetime.now().astimezone().date().isoformat()
    destination = ROOT / "result_analysis/reports/moe_study" / f"moe_study_final_{model}_96_languages_{date}.xlsx"
    temporary = out / "report.pending.xlsx"
    build_workbook(summary_rows=summary, eval_rows=eval_rows, baseline_rows=baseline_rows, resource_map=resource_map,
        profile_payload=profile, runs_root=out / "runs", sample_manifest_path=root / "inputs/sample_manifest.json",
        tokenized_dir=out / "tokenized", benchmark_summary_path=out / "benchmarks/benchmark_summary.json",
        repo_root=ROOT, profile_name=model, logs_root=out / "logs", out_path=temporary)
    wb = load_workbook(temporary)
    for row in wb["experiment_meta"].iter_rows():
        if row[0].value == "report_path": row[1].value = str(destination)
    backend = read_json(out / "backend_ready.json")
    for key, value in {"campaign_id": contract()["campaign_id"],
                       "model_revision": contract()["models"][model]["revision"],
                       "validated_attention_backend": backend["selected_backend"],
                       "benchmark_language_id_mode": "no_ids",
                       "nominal_training_languages": 96, "gpus_per_training": 4,
                       "trainer_budget_seconds": 122400}.items():
        wb["experiment_meta"].append([key, value])
    documents = read_json(out / "data_ready.json")
    document_rows = [{"language": language, "split": split, "documents": count,
                      "sequences": documents["sequences"][split][language],
                      "effective_tokens": documents["effective_tokens"][split][language]}
                     for split, counts in documents["documents"].items() for language, count in counts.items()]
    for name, frame in [("macro_ranks", ranking.reset_index()), ("benchmark_ranks", per_benchmark.reset_index()),
                        ("training_documents", pd.DataFrame(document_rows)), ("campaign_validation", pd.DataFrame(provenance))]:
        ws = wb.create_sheet(name)
        for row in dataframe_to_rows(frame, index=False, header=True): ws.append(row)
        ws.freeze_panes = "A2"
    wb.save(temporary)
    wb.close()
    # Round-trip through the actual workbook parser before publishing it.
    check_panel(primary_data(pd.read_excel(temporary, sheet_name="benchmark_raw")), data.run_id.unique(), tasks)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    write_json(out / "report_ready.json", {"pass": True, "path": str(destination), "sha256": sha256(destination), "run_ids": sorted(data.run_id.unique())})
    print(destination, flush=True)


def correlations():
    from scipy.stats import spearmanr, kendalltau
    root = campaign_root()
    paths = {"llama31_8b": LLAMA_WORKBOOK}
    for model in contract()["models"]:
        ready = read_json(model_root(model) / "report_ready.json")
        if not ready["pass"] or ready["sha256"] != sha256(ready["path"]):
            raise ValueError("A final workbook changed or is not ready")
        paths[model] = Path(ready["path"])
    tasks = (root / "inputs/tasks.txt").read_text().splitlines()
    rankings = {}
    for model, path in paths.items():
        data = primary_data(pd.read_excel(path, sheet_name="benchmark_raw"), contract()["run_ids"])
        check_panel(data, contract()["run_ids"], tasks)
        rankings[model] = macro_ranks(data)[0]  # Recompute all ranks WITHOUT any baseline.
    rows = []
    for a, b in combinations(rankings, 2):
        for column in [*BENCHMARKS, "macro_avg_rank"]:
            x = rankings[a].reindex(contract()["run_ids"])[column]
            y = rankings[b].reindex(contract()["run_ids"])[column]
            rows.append({"model_a": a, "model_b": b, "benchmark": column, "n_configs": len(x),
                         "spearman_rho": spearmanr(x, y).statistic, "kendall_tau_b": kendalltau(x, y).statistic})
    target = root / "correlations"
    target.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(target / "rank_correlations.csv", index=False)
    for model, ranking in rankings.items(): ranking.to_csv(target / f"{model}_common_seven_ranks.csv")
    write_json(target / "provenance.json", {"workbooks": {key: {"path": str(p), "sha256": sha256(p)} for key, p in paths.items()},
        "pool": contract()["run_ids"], "interpretation": "Descriptive correlations across seven fixed configurations, one training seed; not independent replications or causal architectural effects."})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def pending_collectors(sources):
    """Do not depend on finished jobs that Slurm may already have purged.

    Reuse only complete, hash-verified artifacts; unfinished collectors remain
    afterok dependencies. The report independently validates all inputs again.
    """
    from pipeline_tools.cross_arch_pilot import primary_values
    panel = (campaign_root() / "inputs/tasks.txt").read_text().splitlines()
    dependencies = []
    for job in sources:
        run_id, model = job["run_id"], job["model"]
        folder = model_root(model) / ("baseline" if run_id == "baseline" else "benchmarks/" + run_id)
        receipt = folder / "complete.json"
        if not receipt.exists():
            dependencies.append(job["collect"])
            continue
        ready = read_json(receipt)
        if (not ready.get("pass") or ready.get("model") != model or ready.get("run_id") != run_id
                or ready.get("tasks") != panel
                or ready.get("results_sha256") != sha256(folder / "no_language_ids.json")):
            raise ValueError(f"Invalid completed collector receipt: {receipt}")
        primary_values(folder, panel)
    return list(dict.fromkeys(dependencies))


def submit_ready_reports(submit=False):
    root = campaign_root()
    with (root / ".report_submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluation = read_json(root / "evaluation_jobs.json")["jobs"]
        path = root / "report_jobs.json"
        record = read_json(path) if path.exists() else {"jobs": []}
        for model in contract()["models"]:
            if any(j.get("model") == model for j in record["jobs"]): continue
            sources = [j for j in evaluation if j["model"] == model and j.get("status") == "submitted"]
            if {j["run_id"] for j in sources} != {"baseline", *contract()["run_ids"]}: continue
            dependencies = pending_collectors(sources)
            job = sbatch([str(PYTHON), "-m", "pipeline_tools.cross_arch_report", "report", "--model", model],
                "xa_report_" + model, hours="01:00:00", submit=submit,
                dependency="afterok:" + ":".join(dependencies) if dependencies else None)
            if submit:
                record["jobs"].append({"stage": "report", "model": model, "job_id": job})
                write_json(path, record)
        reports = [j["job_id"] for j in record["jobs"] if j["stage"] == "report"]
        if len(reports) == 2 and not any(j["stage"] == "correlations" for j in record["jobs"]):
            job = sbatch([str(PYTHON), "-m", "pipeline_tools.cross_arch_report", "correlations"], "xa_correlations",
                hours="00:30:00", submit=submit, dependency="afterok:" + ":".join(reports))
            if submit:
                record["jobs"].append({"stage": "correlations", "job_id": job})
                write_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["report", "correlations", "submit"])
    parser.add_argument("--model", choices=list(contract()["models"]))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.action == "report":
        if not args.model: parser.error("--model required")
        build_report(args.model)
    elif args.action == "correlations": correlations()
    else: submit_ready_reports(args.submit)


if __name__ == "__main__":
    main()
