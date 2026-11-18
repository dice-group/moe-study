import json
import math
from pathlib import Path
from typing import Any

from .runtime import (
    _build_run_label,
    _load_json,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..runtime import _deep_merge
from ..context import PipelineContext
from ..training import adapter_checkpoint_provenance, build_final_eval_cmd, resolve_adapter_eval_path
from .per_run_summary import write_per_run_summary


def _resolve_eval_batch_size(ctx: PipelineContext, run: dict[str, Any]) -> int:
    raw = ctx.final_eval_loss_cfg.get("batch_size", 1)
    if isinstance(raw, str) and raw.strip().lower() in {"train", "training", "run", "per_run"}:
        train_cfg = _deep_merge(ctx.train_cfg.get("train", {}), run.get("train", {}))
        raw = train_cfg.get("batch_size", 1)

    try:
        batch_size = int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"final_eval_loss.batch_size must be a positive integer or 'train' (got {raw!r})") from exc
    if batch_size <= 0:
        raise SystemExit(f"final_eval_loss.batch_size must be positive (got {batch_size})")
    return batch_size


def _expected_max_steps(ctx: PipelineContext, run: dict[str, Any]) -> int | None:
    train_cfg = _deep_merge(ctx.train_cfg.get("train", {}), run.get("train", {}))
    raw = train_cfg.get("max_steps")
    if raw is None:
        return None
    try:
        steps = int(raw)
    except (TypeError, ValueError):
        return None
    return steps if steps > 0 else None


def _trainer_state_global_step(run_dir: Path, *, dry_run: bool) -> int | None:
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        if dry_run:
            return None
        raise SystemExit(
            f"final_eval_loss: checkpoint_step_source='trainer_state' but trainer_state.json is missing: {state_path}"
        )
    payload = _load_json(state_path)
    raw = payload.get("global_step") if isinstance(payload, dict) else None
    try:
        step = int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"final_eval_loss: trainer_state.json has invalid global_step for {run_dir}: {raw!r}"
        ) from exc
    if step <= 0:
        raise SystemExit(f"final_eval_loss: trainer_state.json global_step must be positive for {run_dir}: {step}")
    return step


def _expected_checkpoint_step(ctx: PipelineContext, run: dict[str, Any], run_dir: Path) -> int | None:
    source = str(ctx.final_eval_loss_cfg.get("checkpoint_step_source", "max_steps")).strip().lower()
    if source in {"", "max_steps", "config", "configured"}:
        return _expected_max_steps(ctx, run)
    if source in {"trainer_state", "state", "global_step"}:
        return _trainer_state_global_step(run_dir, dry_run=ctx.args.dry_run)
    raise SystemExit(
        "final_eval_loss.checkpoint_step_source must be one of "
        f"'max_steps' or 'trainer_state' (got {source!r})"
    )


def stage_final_eval_loss(ctx: PipelineContext) -> None:
    if "final_eval_loss" not in ctx.active_stages:
        return
    if not bool(ctx.final_eval_loss_cfg.get("enabled", False)):
        return

    runs = ctx.train_cfg.get("runs", [])
    if not runs:
        raise SystemExit("final_eval_loss: no training runs configured in profile.training.runs")

    output_subdir = str(ctx.final_eval_loss_cfg.get("output_subdir", "final_eval_loss")).strip() or "final_eval_loss"
    signature = _stage_signature(
        {
            "stage": "final_eval_loss",
            "cfg": ctx.final_eval_loss_cfg,
            "train_cfg": ctx.train_cfg,
            "tokenized_dir": str(ctx.tokenized_dir),
            "output_subdir": output_subdir,
        }
    )
    per_run_summary_root = ctx.final_eval_loss_root / "run_summaries"
    expected = [ctx.train_root / str(run["id"]) / output_subdir / "eval_results.json" for run in runs]
    expected.extend(per_run_summary_root / f"{run['id']}.json" for run in runs)
    stage_summary = ctx.final_eval_loss_root / "summary.json"
    expected.append(stage_summary)
    state_stage = (
        f"final_eval_loss:{runs[0]['id']}"
        if len(runs) == 1 and ctx.args.run_id
        else "final_eval_loss"
    )
    force_stages = set(ctx.force_stages)
    if "final_eval_loss" in force_stages:
        force_stages.add(state_stage)

    if _should_skip_stage(
        state_path=ctx.state_path,
        stage=state_stage,
        signature=signature,
        expected_artifacts=expected,
        force_stages=force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] final_eval_loss stage (cache hit)")
        return

    ctx.final_eval_loss_root.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run_id = str(run["id"])
        run_name_label = _build_run_label(
            base_run_label=ctx.base_run_label,
            run=run,
            default_num_experts=int(ctx.train_cfg.get("num_experts", 0)),
        )
        run_dir = ctx.train_root / run_id
        lineage_path = run_dir / "model_tokenizer_lineage.json"
        lineage = _load_json(lineage_path) if lineage_path.exists() else {}
        model_name_or_path = str(lineage.get("model_name_or_path") or ctx.train_cfg.get("model_name_or_path") or "")
        if not model_name_or_path:
            raise SystemExit(f"final_eval_loss: missing model_name_or_path for run {run_id}")

        expected_step = _expected_checkpoint_step(ctx, run, run_dir)
        strict_final_checkpoint = bool(ctx.final_eval_loss_cfg.get("strict_final_checkpoint", False))
        require_direct_save_marker = ctx.final_eval_loss_cfg.get("require_direct_save_marker")
        if require_direct_save_marker is None:
            require_direct_save_marker = strict_final_checkpoint
        if ctx.args.dry_run:
            try:
                adapter_path = resolve_adapter_eval_path(
                    run_dir,
                    expected_step=expected_step,
                    strict_final_checkpoint=strict_final_checkpoint,
                    require_direct_save_marker=bool(require_direct_save_marker),
                )
            except SystemExit:
                adapter_path = run_dir
        else:
            adapter_path = resolve_adapter_eval_path(
                run_dir,
                expected_step=expected_step,
                strict_final_checkpoint=strict_final_checkpoint,
                require_direct_save_marker=bool(require_direct_save_marker),
            )
        out_dir = run_dir / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_results_path = out_dir / "eval_results.json"
        already_complete = (
            "final_eval_loss" not in ctx.force_stages
            and (not ctx.args.dry_run)
            and eval_results_path.exists()
        )
        if already_complete:
            ctx.logger.info("[SKIP] final_eval_loss run %s (already complete)", run_id)
        else:
            cmd, run_env = build_final_eval_cmd(
                repo_root=ctx.repo_root,
                env=ctx.env,
                output_dir=out_dir,
                run_cfg=run,
                common=ctx.train_cfg,
                language_map_path=ctx.language_map_path,
                tokenized_path=ctx.tokenized_dir,
                embedding_training_cfg=ctx.embedding_training_cfg,
                run_name=run_name_label,
                model_name_or_path=model_name_or_path,
                adapter_name_or_path=str(adapter_path),
                per_device_eval_batch_size=_resolve_eval_batch_size(ctx, run),
                max_samples=int(ctx.final_eval_loss_cfg["limit"]) if ctx.final_eval_loss_cfg.get("limit") is not None else None,
                verify_adapter=not ctx.args.dry_run,
            )
            _run_stage_cmd(
                cmd=cmd,
                env=run_env,
                dry_run=ctx.args.dry_run,
                logs_root=ctx.logs_root,
                stage="final_eval_loss",
                suffix=run_name_label,
                executor=ctx.executor,
            )

        if not ctx.args.dry_run:
            if not eval_results_path.exists():
                raise SystemExit(f"final_eval_loss: missing eval_results.json for run {run_id}: {eval_results_path}")
            payload = json.loads(eval_results_path.read_text(encoding="utf-8"))
            eval_loss = payload.get("eval_loss")
            if not isinstance(eval_loss, (int, float)) or not math.isfinite(float(eval_loss)):
                raise SystemExit(
                    f"final_eval_loss: eval_loss missing/non-finite for run {run_id} in {eval_results_path}"
                )
            eval_row = {
                "run_id": run_id,
                "eval_loss": float(eval_loss),
                "adapter_name_or_path": str(adapter_path),
                "adapter_provenance": adapter_checkpoint_provenance(
                    adapter_path,
                    expected_step=expected_step,
                    require_direct_save_marker=bool(require_direct_save_marker),
                ),
                "model_name_or_path": model_name_or_path,
                "eval_results": str(eval_results_path),
            }
            write_per_run_summary(
                per_run_path=per_run_summary_root / f"{run_id}.json",
                aggregate_path=stage_summary,
                metadata={},
                run=eval_row,
            )

    if not ctx.args.dry_run:
        _save_stage_state(ctx.state_path, state_stage, signature, [str(path) for path in expected])
        ctx.logger.info("[OK] final_eval_loss_summary=%s", stage_summary)
