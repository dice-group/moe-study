#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets
import time
import gc

def merge_in_batches(root: Path, batch_size: int = 20):
    """
    Safely merge all tokenized subdatasets under `root`.
    Writes intermediate merged parts to disk in batches,
    then merges those into one final dataset.
    """
    out_final = root / "merged"
    out_final.mkdir(exist_ok=True, parents=True)

    shard_dirs = []
    for lang_dir in sorted(root.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name == "merged":
            continue
        for file_dir in lang_dir.glob("*/"):
            for shard_dir in file_dir.glob("*/"):
                if any(shard_dir.glob("data-*.arrow")):
                    shard_dirs.append(shard_dir)

    print(f"Found {len(shard_dirs)} tokenized shards to merge.")

    partial_paths = []
    log_file = out_final / "merge_log.txt"
    processed = set()

    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                processed.add(line.strip())
        print(f"Resuming from previous run: {len(processed)} batches already done.")

    # Batch merge loop
    for i in range(0, len(shard_dirs), batch_size):
        batch_dirs = shard_dirs[i:i+batch_size]
        batch_name = f"merged_part_{i//batch_size:03d}"
        out_path = root / batch_name

        if out_path.exists() or batch_name in processed:
            print(f"Skipping already merged batch {batch_name}")
            partial_paths.append(out_path)
            continue

        print(f"🔹 Merging batch {batch_name} ({len(batch_dirs)} shards)...")
        datasets = []
        for d in batch_dirs:
            try:
                ds = load_from_disk(str(d))
                datasets.append(ds)
            except Exception as e:
                print(f"Skipping {d}: {e}")

        if not datasets:
            print(f"No valid datasets in {batch_name}, skipping.")
            continue

        merged = concatenate_datasets(datasets)
        merged.save_to_disk(str(out_path))
        partial_paths.append(out_path)
        print(f"Saved {batch_name} ({len(merged)} examples)")
        del merged, datasets
        gc.collect()

        with open(log_file, "a") as f:
            f.write(batch_name + "\n")

        time.sleep(1)  # give filesystem a breath

    print(f"Merging {len(partial_paths)} partial batches into final dataset...")

    partial_datasets = [load_from_disk(str(p)) for p in partial_paths]
    final = concatenate_datasets(partial_datasets)
    final.save_to_disk(str(out_final))

    print(f"Final merged dataset saved to: {out_final}")
    print(f"Total shards: {len(shard_dirs)}, batches: {len(partial_paths)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Safe batch-wise dataset merger for Hugging Face Arrow datasets.")
    ap.add_argument("--root", type=Path, required=True, help="Root tokenized directory (contains all languages)")
    ap.add_argument("--batch-size", type=int, default=20, help="Number of subdatasets per merge batch")
    args = ap.parse_args()

    merge_in_batches(args.root, args.batch_size)
