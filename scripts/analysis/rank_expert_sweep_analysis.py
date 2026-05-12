#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config_loader import deep_merge, load_pipeline_config  # noqa: E402
from pipeline.runtime import _resolve_profile  # noqa: E402


DEFAULT_PROFILE = "moe_study_rank_expert_sweep_otus_cluster_10_langs"
RANK_RE = re.compile(r"_r(\d+)$")
TOTAL_RANK_RE = re.compile(r"_tr(\d+)_")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle) if row.get("run_id")}


def _settings_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_metadata(run: dict[str, Any], common_lora: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run["id"])
    kind = str(run["kind"])
    params = run.get("params", {}) if isinstance(run.get("params"), dict) else {}
    lora = deep_merge(common_lora, run.get("lora", {}))
    configured_rank = _as_int(lora.get("rank"))
    alpha = _as_int(lora.get("alpha"))

    total_match = TOTAL_RANK_RE.search(run_id)
    rank_match = RANK_RE.search(run_id)
    total_rank = _as_int(total_match.group(1)) if total_match else None
    rank_label = None
    rank_semantics = "lora_rank"

    if total_rank is not None:
        rank_label = f"tr{total_rank}"
        rank_semantics = "total_rank"
    elif rank_match:
        rank_label = f"r{rank_match.group(1)}"
        total_rank = configured_rank
    elif kind == "movlora":
        rank_label = "rankless"
        configured_rank = None
        alpha = None
        rank_semantics = "rankless_ia3"

    num_experts = (
        params.get("moelora_num_experts")
        or params.get("adamole_num_experts")
        or params.get("mola_num_experts")
        or params.get("movlora_num_experts")
        or params.get("hmora_num_experts")
        or params.get("cola_num_experts")
        or params.get("hydralora_num_experts")
    )
    top_k = (
        params.get("moelora_sparse_top_k")
        or params.get("mola_top_k")
        or params.get("movlora_top_k")
        or params.get("hmora_top_k")
        or params.get("cola_top_k")
        or params.get("hydralora_top_k")
    )

    return {
        "run_id": run_id,
        "kind": kind,
        "rank_label": rank_label,
        "total_rank": total_rank,
        "configured_lora_rank": configured_rank,
        "rank_semantics": rank_semantics,
        "lora_alpha": alpha,
        "num_experts": num_experts,
        "top_k": top_k,
    }


def _validate(profile: dict[str, Any], settings_rows: list[dict[str, str]] | None = None) -> list[str]:
    runs = profile["training"]["runs"]
    errors: list[str] = []
    expected_run_ids: set[str] | None = None
    if settings_rows:
        included = [row["run_id"] for row in settings_rows if row.get("status", "included") == "included"]
        infeasible = [row["run_id"] for row in settings_rows if row.get("status") == "infeasible_bs1"]
        expected_run_ids = set(included)
        if len(included) != len(expected_run_ids):
            errors.append("included run ids are not unique in settings")
        if len(infeasible) != len(set(infeasible)):
            errors.append("infeasible run ids are not unique in settings")
    elif not runs:
        errors.append("profile has no training runs")

    ids = [str(run.get("id", "")) for run in runs]
    if len(ids) != len(set(ids)):
        errors.append("run ids are not unique")
    if expected_run_ids is not None and set(ids) != expected_run_ids:
        missing = sorted(expected_run_ids - set(ids))
        unexpected = sorted(set(ids) - expected_run_ids)
        if missing:
            errors.append(f"calibrated profile missing included runs: {missing}")
        if unexpected:
            errors.append(f"calibrated profile includes non-included runs: {unexpected}")

    for run in runs:
        run_id = str(run["id"])
        kind = str(run["kind"])
        params = run.get("params", {}) if isinstance(run.get("params"), dict) else {}
        lora = deep_merge(profile["training"].get("lora", {}), run.get("lora", {}))
        rank = _as_int(lora.get("rank"))
        alpha = _as_int(lora.get("alpha"))

        if kind != "movlora" and rank is not None and alpha != 2 * rank:
            errors.append(f"{run_id}: expected lora alpha {2 * rank}, found {alpha}")

        if kind == "moelora":
            if params.get("moelora_num_experts") != 8:
                errors.append(f"{run_id}: moelora_num_experts must be 8")
            if params.get("moelora_sparse_top_k") != 2:
                errors.append(f"{run_id}: moelora_sparse_top_k must be 2")
            if rank is None or rank % 8 != 0:
                errors.append(f"{run_id}: MoE-LoRA lora.rank must be divisible by 8")

        if kind == "adamole":
            if params.get("adamole_num_experts") != 8:
                errors.append(f"{run_id}: adamole_num_experts must be 8")
            if float(params.get("adamole_max_threshold", -1.0)) != 0.0625:
                errors.append(f"{run_id}: adamole_max_threshold must be 0.0625")

        if kind == "mola":
            if params.get("mola_num_experts") != 8:
                errors.append(f"{run_id}: mola_num_experts must be 8")
            if params.get("mola_num_null_experts") != 16:
                errors.append(f"{run_id}: mola_num_null_experts must be 16")
            if params.get("mola_top_k") != 6:
                errors.append(f"{run_id}: mola_top_k must be 6")

        if kind == "hmora":
            if params.get("hmora_num_experts") != 8:
                errors.append(f"{run_id}: hmora_num_experts must be 8")
            if params.get("hmora_top_k") != 2:
                errors.append(f"{run_id}: hmora_top_k must be 2")

        if kind == "movlora":
            if params.get("movlora_num_experts") != 8:
                errors.append(f"{run_id}: movlora_num_experts must be 8")
            if params.get("movlora_top_k") != 0:
                errors.append(f"{run_id}: movlora_top_k must be 0")

        total_match = TOTAL_RANK_RE.search(run_id)
        if total_match and kind in {"adamole", "mola", "hmora"}:
            total_rank = int(total_match.group(1))
            if rank != total_rank // 8:
                errors.append(f"{run_id}: expected per-expert rank {total_rank // 8}, found {rank}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize the MoE-study rank/expert sweep.")
    parser.add_argument("--config", default="configs/pipeline", help="Pipeline config bundle path.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Pipeline profile name.")
    parser.add_argument("--summary-csv", default="", help="Optional summary.csv path override.")
    parser.add_argument("--settings-csv", default="", help="Optional calibrated settings CSV with infeasible run statuses.")
    parser.add_argument("--output-csv", default="", help="Optional analysis CSV output path.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the resolved run matrix without writing CSV.")
    args = parser.parse_args()

    config = load_pipeline_config(Path(args.config))
    profile = _resolve_profile(config, args.profile)
    settings_rows = _settings_rows(Path(args.settings_csv)) if args.settings_csv else []
    errors = _validate(profile, settings_rows)
    if errors:
        for error in errors:
            print(f"[rank_sweep_validation] {error}", file=sys.stderr)
        return 1
    if args.validate_only:
        print(f"[rank_sweep_validation] ok profile={args.profile} runs={len(profile['training']['runs'])}")
        return 0

    summary_path = Path(args.summary_csv) if args.summary_csv else Path(profile["summary"]["summary_csv"])
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else Path(profile["output_root"]) / "rank_expert_sweep_analysis.csv"
    )
    summaries = _summary_rows(summary_path)
    common_lora = profile["training"].get("lora", {})
    fields = [
        "run_id",
        "kind",
        "rank_label",
        "total_rank",
        "configured_lora_rank",
        "rank_semantics",
        "lora_alpha",
        "num_experts",
        "top_k",
        "trainable_params",
        "all_params",
        "trainable_pct",
        "train_runtime",
        "effective_tokens_per_sec",
        "train_loss",
    ]
    rows = []
    for run in profile["training"]["runs"]:
        metadata = _run_metadata(run, common_lora)
        summary = summaries.get(metadata["run_id"], {})
        rows.append(
            {
                **metadata,
                "trainable_params": summary.get("trainable_params"),
                "all_params": summary.get("all_params"),
                "trainable_pct": summary.get("trainable_pct"),
                "train_runtime": summary.get("train_runtime"),
                "effective_tokens_per_sec": summary.get("effective_tokens_per_sec"),
                "train_loss": summary.get("train_loss"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[rank_sweep_analysis] wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
