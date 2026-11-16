#!/usr/bin/env python3
"""Print the single complete direct-save adapter checkpoint for one run."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.training import adapter_checkpoint_provenance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = (args.run_root / "runs" / args.run_id).resolve()
    candidates = sorted(
        checkpoint
        for checkpoint in run_dir.glob("checkpoint-*_adapter")
        if (checkpoint / ".adapter_direct_save_complete").is_file()
        and (checkpoint / "adapter_config.json").is_file()
        and (checkpoint / "adapter_model.safetensors").is_file()
    )
    if len(candidates) != 1:
        raise SystemExit(
            f"{args.run_id}: expected one complete direct-save checkpoint, "
            f"found {len(candidates)} under {run_dir}"
        )

    checkpoint = candidates[0]
    match = re.fullmatch(r"checkpoint-(\d+)_adapter", checkpoint.name)
    if not match:
        raise SystemExit(f"Unexpected checkpoint name: {checkpoint.name}")
    provenance = adapter_checkpoint_provenance(
        checkpoint,
        expected_step=int(match.group(1)),
        require_direct_save_marker=True,
    )
    if not provenance.get("valid"):
        raise SystemExit(f"{args.run_id}: invalid adapter provenance: {provenance}")
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
