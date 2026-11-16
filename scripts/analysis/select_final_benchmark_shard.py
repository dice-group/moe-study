#!/usr/bin/env python3
"""Select one disjoint benchmark-family shard from a task list."""

from __future__ import annotations

import argparse
from pathlib import Path


def belongs(task: str, shard: str) -> bool:
    if shard == "belebele":
        return task.startswith("belebele_")
    if shard == "sib200":
        return task.startswith("sib200_")
    if shard == "taxi1500":
        return task.startswith("taxi1500_")
    if shard == "flores_en_x":
        return task.startswith("flores_en_perplexity_eng_Latn-")
    if shard == "flores_x_en":
        return task.startswith("flores_en_perplexity_") and not task.startswith(
            "flores_en_perplexity_eng_Latn-"
        )
    raise ValueError(shard)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument(
        "--shard",
        required=True,
        choices=["belebele", "flores_en_x", "flores_x_en", "sib200", "taxi1500"],
    )
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    if args.parts < 1 or not 0 <= args.part_index < args.parts:
        raise SystemExit(
            f"Invalid part selection: part_index={args.part_index}, parts={args.parts}"
        )

    tasks = [
        line.strip()
        for line in Path(args.tasks_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    family_tasks = [task for task in tasks if belongs(task, args.shard)]
    if not family_tasks or len(family_tasks) != len(set(family_tasks)):
        raise SystemExit(f"Invalid shard {args.shard}: {len(family_tasks)} tasks")
    selected = family_tasks[args.part_index :: args.parts]
    if not selected:
        raise SystemExit(
            f"Empty shard part: shard={args.shard}, part_index={args.part_index}, "
            f"parts={args.parts}"
        )

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(
        f"[OK] shard={args.shard} part={args.part_index + 1}/{args.parts} "
        f"tasks={len(selected)} family_tasks={len(family_tasks)} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
