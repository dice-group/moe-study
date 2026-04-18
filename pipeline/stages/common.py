"""Common helpers for proof-pipeline stage handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_lm_eval_layout(repo_root: Path, logger: Any) -> None:
    candidates = [
        repo_root / "scripts" / "eval" / "latest_checkpoint" / "lm-evaluation-harness" / "lm_eval",
        repo_root / "scripts" / "eval" / "lm-evaluation-harness" / "lm_eval",
    ]
    local_lm_eval = next((path for path in candidates if path.exists()), None)
    compat_link = repo_root / "lm_eval"
    if local_lm_eval is None or compat_link.exists():
        return
    try:
        compat_link.symlink_to(local_lm_eval.relative_to(repo_root))
        logger.info("[OK] created compatibility symlink: %s -> %s", compat_link, local_lm_eval)
    except OSError as exc:
        logger.warning("[WARN] could not create compatibility symlink %s: %s", compat_link, exc)
