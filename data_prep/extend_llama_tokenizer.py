import os
import gzip
import json
import argparse
import random
import numpy as np
import torch

from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def count_docs(file_list):
    """Count total docs across files"""
    total = 0
    for fpath in file_list:
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for _ in f:
                total += 1
    return total

def iter_jsonl_gz(file_list, max_docs=None):
    """Yield text from a list of .jsonl.gz files"""
    count = 0
    for fpath in file_list:
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        yield text
                        count += 1
                        if max_docs and count >= max_docs:
                            return
                except Exception:
                    continue


def collect_files(data_root, slurm_task_id, slurm_ntasks):
    """Split file list across SLURM tasks"""
    all_files = []
    for lang in sorted(os.listdir(data_root)):
        lang_dir = os.path.join(data_root, lang)
        if not os.path.isdir(lang_dir):
            continue
        for fname in os.listdir(lang_dir):
            if fname.endswith(".jsonl.gz"):
                all_files.append(os.path.join(lang_dir, fname))

    all_files.sort()
    shard = [f for i, f in enumerate(all_files) if i % slurm_ntasks == slurm_task_id]
    return shard


def collect_token_frequencies(files, tokenizer, base_vocab, out_file, max_docs, task_id=0, debug=False):
    """collect new tokens and their frequency"""
    if debug:
        print(f"[Task {task_id}][DEBUG] Shard files = {files}")

    if not files:
        print(f"[Task {task_id}] No files assigned!")
        return

    total = count_docs(files) if not max_docs else max_docs
    print(f"[Task {task_id}] Total docs to process: {total}")

    counter, processed = Counter(), 0
    for text in iter_jsonl_gz(files, max_docs=max_docs):
        if debug and processed < 3:  # show first 3 texts for inspection
            print(f"[Task {task_id}][DEBUG] Sample text: {text[:200]}...")

        tokens = tokenizer.tokenize(text)
        if debug and processed < 3:
            print(f"[Task {task_id}][DEBUG] Tokenized: {tokens[:20]}")

        for token in tokens:
            if token not in base_vocab:
                counter[token] += 1

        processed += 1
        if processed % 1000 == 0:
            pct = (processed / total) * 100
            print(f"[Task {task_id}] {processed}/{total} docs ({pct:.1f}%)", flush=True)

    Path(os.path.dirname(out_file)).mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(counter, f, ensure_ascii=False)

    print(f"[Task {task_id}] Collected {len(counter)} unique unknown tokens -> {out_file}")
    if debug and len(counter) == 0:
        print(f"[Task {task_id}][DEBUG] No OOV tokens found — possible causes: "
              f"(1) tokenizer already covers everything, "
              f"(2) filtering condition too strict, "
              f"(3) shard has no usable text.")

def merge_tokens(base_model, shard_files, output_dir, min_freq, target_vocab):
    """ Merge tokens and add the most frequent ones """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base_vocab = set(tokenizer.get_vocab().keys())

    merged_counter = Counter()
    for sf in shard_files:
        if not os.path.exists(sf):
            continue
        with open(sf, "r", encoding="utf-8") as f:
            shard_counts = json.load(f)
            merged_counter.update(shard_counts)
    print(f"[Merge] Total candidate tokens: {len(merged_counter)}")

    # filter by frequency
    filtered = {tok: cnt for tok, cnt in merged_counter.items() if cnt >= min_freq}
    print(f"[Merge] After min_freq={min_freq}: {len(filtered)} tokens")

    # sort by frequency
    sorted_tokens = sorted(filtered.items(), key=lambda x: -x[1])

    # cut to target vocab size
    if target_vocab is not None:
        sorted_tokens = sorted_tokens[:target_vocab]

    new_tokens = [tok for tok, _ in sorted_tokens if tok not in base_vocab]

    print(f"[Merge] Adding {len(new_tokens)} new tokens to tokenizer")
    tokenizer.add_tokens(new_tokens)
    tokenizer.save_pretrained(output_dir)
    print(f"[Merge] Extended tokenizer saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to input samples")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1b", help="Base model name")
    parser.add_argument("--max_docs", type=int, default=None, help="Limit docs per shard")
    parser.add_argument("--merge", action="store_true", help="Run merge step only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_freq", type=int, default=10, help="Min frequency to keep token")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--target_vocab", type=int, default=5000, help="Max vocab size to add")
    args = parser.parse_args()

    set_seed(args.seed)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    if not args.merge:
        shard_files = collect_files(args.data_root, task_id, ntasks)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_vocab = set(tokenizer.get_vocab().keys())

        out_file = os.path.join(args.output_dir, f"tokens_{task_id}.json")
        print(f"[Task {task_id}] Processing {len(shard_files)} files")
        collect_token_frequencies(shard_files, tokenizer, base_vocab, out_file, args.max_docs, task_id, debug=args.debug)
    else:
        shard_files = [os.path.join(args.output_dir, f"tokens_{i}.json") for i in range(ntasks)]
        merge_tokens(args.base_model, shard_files, args.output_dir,
                     min_freq=args.min_freq, target_vocab=args.target_vocab)

if __name__ == "__main__":
    main()
