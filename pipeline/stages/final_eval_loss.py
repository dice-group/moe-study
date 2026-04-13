import json
import re
from pathlib import Path
from typing import Any

from .runtime import (
    _build_run_label,
    _dump_json,
    _load_json,
    _missing_modules,
    _run_stage_cmd,
    _save_stage_state,
    _should_skip_stage,
    _stage_signature,
)
from ..context import PipelineContext
from ..training import build_final_eval_cmd


_CHECKPOINT_ADAPTER_RE = re.compile(r"^checkpoint-(\d+)_adapter$")


def _resolve_adapter_for_eval(run_dir: Path) -> Path:
    """Pick the best adapter directory to evaluate.

    Preference order:
    1) latest 'checkpoint-*_adapter' (produced when we save latest SIGTERM JIT checkpointing a few seconds before job termination)
    2) run root, if it looks like a PEFT adapter directory
    """
    best_step = -1
    best_path: Path | None = None
    if run_dir.exists():
        for child in run_dir.iterdir():
            if not child.is_dir():
                continue
            match = _CHECKPOINT_ADAPTER_RE.match(child.name)
            if not match:
                continue
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_path = child

    if best_path is not None:
        return best_path

    # fallback to root adapter 'trainer.save_model()`
    if (run_dir / "adapter_config.json").exists():
        return run_dir

    raise SystemExit(
        f"final_eval_loss: could not resolve adapter to evaluate in {run_dir} "
        "(expected checkpoint-*_adapter or adapter_config.json)."
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
    expected = [ctx.train_root / str(run["id"]) / output_subdir / "eval_results.json" for run in runs]

    if _should_skip_stage(
        state_path=ctx.state_path,
        stage="final_eval_loss",
        signature=signature,
        expected_artifacts=expected,
        force_stages=ctx.force_stages,
        dry_run=ctx.args.dry_run,
    ):
        ctx.logger.info("[SKIP] final_eval_loss stage (cache hit)")
        return

    ctx.final_eval_loss_root.mkdir(parents=True, exist_ok=True)
    eval_rows: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run["id"])
        run_name_label = _build_run_label(
            base_run_label=ctx.base_run_label,
            run=run,
            default_num_experts=int(ctx.train_cfg.get("num_experts", 0)),
        )
        run_dir = ctx.train_root / run_id
        lineage_path = run_dir / "model_tokenizer_lineage.json"
        if not lineage_path.exists():
            raise SystemExit(
                f"final_eval_loss requires {lineage_path} (produced by train stage). "
                "Run train stage first."
            )
        lineage = _load_json(lineage_path)
        model_name_or_path = str(lineage.get("model_name_or_path") or ctx.train_cfg.get("model_name_or_path") or "")
        if not model_name_or_path:
            raise SystemExit(f"final_eval_loss: missing model_name_or_path for run {run_id}")

        adapter_path = _resolve_adapter_for_eval(run_dir)
        out_dir = run_dir / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_results_path = out_dir / "eval_results.json"
        if (
            "final_eval_loss" not in ctx.force_stages
            and (not ctx.args.dry_run)
            and eval_results_path.exists()
        ):
            ctx.logger.info("[SKIP] final_eval_loss run %s (already complete)", run_id)
            continue

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
            per_device_eval_batch_size=int(ctx.final_eval_loss_cfg.get("batch_size", 1)),
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
            if not isinstance(eval_loss, (int, float)):
                raise SystemExit(
                    f"final_eval_loss: eval_loss missing/non-numeric for run {run_id} in {eval_results_path}"
                )
            eval_rows.append(
                {
                    "run_id": run_id,
                    "eval_loss": float(eval_loss),
                    "adapter_name_or_path": str(adapter_path),
                    "model_name_or_path": model_name_or_path,
                    "eval_results": str(eval_results_path),
                }
            )

    if not ctx.args.dry_run:
        stage_summary = ctx.final_eval_loss_root / "summary.json"
        _dump_json(stage_summary, {"runs": eval_rows})
        _save_stage_state(ctx.state_path, "final_eval_loss", signature, [str(path) for path in expected])
        ctx.logger.info("[OK] final_eval_loss_summary=%s", stage_summary)

