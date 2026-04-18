from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _norm_targets(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return sorted({part.strip() for part in value.split(",") if part.strip()})
    if isinstance(value, list):
        return sorted({str(part).strip() for part in value if str(part).strip()})
    return []


def verify_adapter_lora_hparams(
    *,
    adapter_dir: Path,
    expected_rank: int,
    expected_alpha: int,
    expected_dropout: float,
    expected_targets: object,
) -> None:
    """Assert that adapter_config.json matches the LoRA hyperparams we intend to use.

    This is a safety check: some training/eval entrypoints reconstruct adapter modules from
    CLI args before loading weights. If the args don't match the checkpoint, eval can fail
    or be silently wrong.
    """
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return
    try:
        adapter_cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read adapter_config.json: {cfg_path} ({exc})") from exc

    exp_targets = _norm_targets(expected_targets)
    ckpt_rank = int(adapter_cfg.get("r", expected_rank))
    ckpt_alpha = int(adapter_cfg.get("lora_alpha", expected_alpha))
    ckpt_dropout = float(adapter_cfg.get("lora_dropout", expected_dropout))
    ckpt_targets = _norm_targets(adapter_cfg.get("target_modules"))

    mismatches: list[str] = []
    if ckpt_rank != int(expected_rank):
        mismatches.append(f"rank ckpt={ckpt_rank} cfg={expected_rank}")
    if ckpt_alpha != int(expected_alpha):
        mismatches.append(f"alpha ckpt={ckpt_alpha} cfg={expected_alpha}")
    if abs(ckpt_dropout - float(expected_dropout)) > 1e-6:
        mismatches.append(f"dropout ckpt={ckpt_dropout} cfg={expected_dropout}")
    if ckpt_targets and (ckpt_targets != exp_targets):
        mismatches.append(f"target_modules ckpt={ckpt_targets} cfg={exp_targets}")

    if mismatches:
        raise SystemExit(
            "LoRA hyperparams in config do not match adapter checkpoint "
            f"({cfg_path}): " + "; ".join(mismatches)
        )

