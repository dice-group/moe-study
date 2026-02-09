#!/usr/bin/env python3
"""Create a masked-label clone of a tokenized dataset saved with HF `save_to_disk`.

The script keeps all columns untouched except `labels`, where positions with
`attention_mask == 0` are rewritten to `-100`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="Path to source dataset (load_from_disk).")
    parser.add_argument("--output-path", required=True, help="Path for masked dataset (save_to_disk).")
    parser.add_argument("--num-proc", type=int, default=32, help="Workers for datasets.map.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output path if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input_path)
    dst = Path(args.output_path)

    if not src.exists():
        raise FileNotFoundError(f"Input dataset not found: {src}")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {dst}. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    dataset = load_from_disk(str(src))
    missing = {"labels", "attention_mask"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    def mask_batch(batch):
        out = []
        for labels, mask in zip(batch["labels"], batch["attention_mask"]):
            out.append([tok if m == 1 else -100 for tok, m in zip(labels, mask)])
        batch["labels"] = out
        return batch

    masked = dataset.map(
        mask_batch,
        batched=True,
        num_proc=max(1, int(args.num_proc)),
        desc="Mask pad labels to -100",
    )
    masked.save_to_disk(str(dst))
    print(f"[INFO] Saved masked dataset to {dst}")
    print(f"[INFO] Rows: {len(masked)}")


if __name__ == "__main__":
    main()
