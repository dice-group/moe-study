#!/usr/bin/env python3
import argparse, gzip, json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def tokenize_file(file_path: Path, tokenizer_name: str, out_dir: Path, num_proc: int = 8, shard_size: int = 100_000):
    """
    Tokenizes a single .jsonl.gz file and writes tokenized shards to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    print(f"Tokenizing {file_path}")
    ds = load_dataset("json", data_files=str(file_path), split="train", streaming=True)

    buffer = []
    shard_idx = 0
    for i, ex in enumerate(ds):
        text = ex.get("text", "")
        toks = tokenizer(text, add_special_tokens=False)
        buffer.append(toks)

        if len(buffer) >= shard_size:
            Dataset.from_list(buffer).save_to_disk(str(out_dir / f"shard_{shard_idx:04d}"))
            print(f"Saved shard {shard_idx} ({i:,} examples)", flush=True)
            buffer.clear()
            shard_idx += 1

    if buffer:
        Dataset.from_list(buffer).save_to_disk(str(out_dir / f"shard_{shard_idx:04d}"))
    print(f"finished {file_path.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--num-proc", type=int, default=8)
    args = ap.parse_args()

    # create subdir for this file
    lang_name = args.input_file.parent.name
    shard_name = args.input_file.stem
    out_subdir = args.out_dir / lang_name / shard_name

    tokenize_file(args.input_file, args.tokenizer, out_subdir, args.num_proc)

if __name__ == "__main__":
    main()
