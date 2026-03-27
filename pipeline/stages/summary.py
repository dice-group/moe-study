from __future__ import annotations

import csv
import json
import re

from .runtime import (
    _dump_json,
    _latest_metric,
    _latest_metric_any_suffix,
    _load_json,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..context import PipelineContext


_TRAINABLE_RE = re.compile(
    r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)\s*\|\|\s*trainable%:\s*([0-9.]+)"
)


def _latest_metric_from_log_jsonl(path, metric: str):
    if not path.exists():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        value = payload.get(metric)
        if isinstance(value, (int, float)):
            latest = value
    return latest


def _load_trainable_stats(run_id: str, output_dir, logs_root) -> dict[str, object]:
    artifact_path = output_dir / "trainable_params.json"
    if artifact_path.exists():
        return _load_json(artifact_path)
    run_log_dir = logs_root / "train_runs" / run_id
    if not run_log_dir.exists():
        return {}
    candidates = sorted(run_log_dir.glob("*.train.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        last_match = None
        for line in candidate.read_text(encoding="utf-8").splitlines():
            match = _TRAINABLE_RE.search(line)
            if match:
                last_match = match
        if last_match is not None:
            return {
                "trainable_params": int(last_match.group(1).replace(",", "")),
                "all_params": int(last_match.group(2).replace(",", "")),
                "trainable_pct": float(last_match.group(3)),
            }
    return {}


def stage_summary(ctx: PipelineContext) -> None:
    if "summary" in ctx.active_stages:
        runs = ctx.train_cfg.get("runs", [])
        ctx.summary_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.gate_path.parent.mkdir(parents=True, exist_ok=True)
        extension_manifest = _load_json(ctx.extension_manifest_path) if ctx.extension_manifest_path.exists() else {}
        embedding_manifest = _load_json(ctx.embedding_init_manifest_path) if ctx.embedding_init_manifest_path.exists() else {}
        embedding_verify_summary = _load_json(ctx.embedding_verify_summary_path) if ctx.embedding_verify_summary_path.exists() else {}
        signature = _stage_signature(
            {
                "stage": "summary",
                "cfg": ctx.summary_cfg,
                "train_cfg": ctx.train_cfg,
                "extension_manifest": str(ctx.extension_manifest_path),
                "embedding_init_manifest": str(ctx.embedding_init_manifest_path),
                "embedding_verify_summary": str(ctx.embedding_verify_summary_path),
            }
        )
        expected = [ctx.summary_path, ctx.gate_path]
        if _should_skip_stage(
            state_path=ctx.state_path,
            stage="summary",
            signature=signature,
            expected_artifacts=expected,
            force_stages=ctx.force_stages,
            dry_run=ctx.args.dry_run,
        ):
            ctx.logger.info("[SKIP] summary stage (cache hit)")
        else:
            rows: list[dict[str, object]] = []
            for run in runs:
                run_id = str(run["id"])
                output_dir = ctx.train_root / run_id
                train_results = _load_json(output_dir / "train_results.json") if (output_dir / "train_results.json").exists() else {}
                trainer_state = _load_json(output_dir / "trainer_state.json") if (output_dir / "trainer_state.json").exists() else {}
                trainable_stats = _load_trainable_stats(run_id, output_dir, ctx.logs_root)
                history = trainer_state.get("log_history", [])
                eval_loss = _latest_metric(history, "eval_loss")
                if eval_loss is None:
                    eval_loss = _latest_metric_from_log_jsonl(output_dir / "trainer_log.jsonl", "eval_loss")
                rows.append(
                    {
                        "run_id": run_id,
                        "kind": run.get("kind"),
                        "train_loss": train_results.get("train_loss"),
                        "train_runtime": train_results.get("train_runtime"),
                        "effective_tokens_per_sec": train_results.get("effective_tokens_per_sec"),
                        "num_input_tokens_seen": train_results.get("num_input_tokens_seen"),
                        "trainable_params": trainable_stats.get("trainable_params"),
                        "all_params": trainable_stats.get("all_params"),
                        "trainable_pct": trainable_stats.get("trainable_pct"),
                        "eval_loss": eval_loss,
                        "language_prior_loss": _latest_metric(history, "language_prior_loss"),
                        "router_grad_present_frac": _latest_metric(history, "router_grad_present_frac"),
                        "expert_load_cv": _latest_metric(history, "expert_load_cv"),
                        "active_expert_frac": _latest_metric(history, "active_expert_frac"),
                        "active_token_pct": _latest_metric(history, "active_token_pct"),
                        "zero_routed_pct": _latest_metric(history, "zero_routed_pct"),
                        "experts_per_token": _latest_metric(history, "experts_per_token"),
                        "router_entropy": _latest_metric(history, "router_entropy"),
                        "router_max_prob": _latest_metric(history, "router_max_prob"),
                        "topk_density": _latest_metric(history, "topk_density"),
                        "lpr_target_hit_rate": _latest_metric_any_suffix(
                            history, ["language_target_hit_rate", "expert_target_hit_rate"]
                        ),
                        "lpr_target_token_frac": _latest_metric_any_suffix(
                            history, ["language_target_token_frac", "expert_target_token_frac"]
                        ),
                        "tokenizer_extension_enabled": bool(ctx.tokenizer_extension_cfg.get("enabled", False)),
                        "extended_tokenizer_dir": extension_manifest.get("extended_tokenizer_dir"),
                        "initialized_model_dir": embedding_manifest.get("initialized_model_dir"),
                        "vocab_growth": extension_manifest.get("vocab_growth"),
                        "embedding_verify_pass": embedding_verify_summary.get("pass"),
                    }
                )
            if not ctx.args.dry_run:
                with ctx.summary_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
            baseline_id = str(ctx.summary_cfg.get("baseline_id", "lora_baseline"))
            metric = str(ctx.summary_cfg.get("primary_metric", "eval_loss"))
            min_rel_impr = float(ctx.summary_cfg.get("min_relative_improvement", 0.0))
            baseline_row = next((row for row in rows if row["run_id"] == baseline_id), None)
            baseline_val = baseline_row.get(metric) if baseline_row else None
            comparisons: list[dict[str, object]] = []
            passes = []
            for row in rows:
                if row["run_id"] == baseline_id:
                    continue
                value = row.get(metric)
                rel = None
                improved = False
                if isinstance(value, (int, float)) and isinstance(baseline_val, (int, float)) and (baseline_val != 0):
                    rel = (baseline_val - value) / abs(baseline_val)
                    improved = rel >= min_rel_impr
                comparisons.append(
                    {
                        "run_id": row["run_id"],
                        "metric_value": value,
                        "baseline_value": baseline_val,
                        "relative_improvement": rel,
                        "passes_gate": improved,
                    }
                )
                if improved:
                    passes.append(row["run_id"])
            gate = {
                "profile": ctx.args.profile,
                "primary_metric": metric,
                "baseline_id": baseline_id,
                "baseline_value": baseline_val,
                "min_relative_improvement": min_rel_impr,
                "passing_runs": passes,
                "comparisons": comparisons,
            }
            if not ctx.args.dry_run:
                _dump_json(ctx.gate_path, gate)
                ctx.lineage_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                _dump_json(
                    ctx.lineage_manifest_path,
                    {
                        "tokenizer_extension_manifest": (
                            str(ctx.extension_manifest_path.resolve()) if ctx.extension_manifest_path.exists() else ""
                        ),
                        "embedding_init_manifest": (
                            str(ctx.embedding_init_manifest_path.resolve()) if ctx.embedding_init_manifest_path.exists() else ""
                        ),
                        "embedding_verify_summary": (
                            str(ctx.embedding_verify_summary_path.resolve())
                            if ctx.embedding_verify_summary_path.exists()
                            else ""
                        ),
                        "extended_tokenizer_dir": extension_manifest.get("extended_tokenizer_dir"),
                        "initialized_model_dir": embedding_manifest.get("initialized_model_dir"),
                        "vocab_growth": extension_manifest.get("vocab_growth"),
                        "embedding_verify_pass": embedding_verify_summary.get("pass"),
                    },
                )
                _save_stage_state(ctx.state_path, "summary", signature, [str(ctx.summary_path), str(ctx.gate_path)])
            ctx.logger.info("[OK] summary_csv=%s", ctx.summary_path)
            ctx.logger.info("[OK] gate_json=%s", ctx.gate_path)
