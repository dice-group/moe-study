#!/usr/bin/env python3
"""Determine how many English rows to add per subset."""

import csv
import statistics
from pathlib import Path

SAMPLE_SIZES_PATH = Path("/home/ubuntu/dice_repos/moe-study/data_dir/fw_samples/sample_sizes.tsv")
SUBSETS_DIR = Path("/home/ubuntu/dice_repos/moe-study/data_dir/llm_nested_samples")


def load_sample_sizes() -> dict:
    with SAMPLE_SIZES_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["subset"]: int(row["sample_size"]) for row in reader}


def english_rows_for_subset(csv_path: Path, sample_sizes: dict) -> tuple[str, int, list[str]]:
    sizes, missing = [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["subset"]
            if code.lower() == "english":
                continue
            size = sample_sizes.get(code)
            if size is None:
                missing.append(code)
                continue
            sizes.append(size)
    english_rows = int(round(statistics.median(sizes))) if sizes else 0
    return csv_path.name, english_rows, missing


def main() -> None:
    sample_sizes = load_sample_sizes()
    header = f"{'subset_file':35s} | {'english_rows':>12s} | missing_sample_sizes"
    print(header)
    print("-" * len(header))
    for csv_path in sorted(SUBSETS_DIR.glob("*.csv")):
        subset_file, english_rows, missing = english_rows_for_subset(csv_path, sample_sizes)
        missing_str = ",".join(missing) if missing else "-"
        print(f"{subset_file:35s} | {english_rows:12d} | {missing_str}")


if __name__ == "__main__":
    main()
