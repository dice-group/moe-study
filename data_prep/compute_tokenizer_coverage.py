#!/usr/bin/env python3
import os
import gzip
import json
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer

def read_jsonl_gz(path, max_lines=None):
    texts = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                data = json.loads(line)
                text = data.get("text") or data.get("content") or ""
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def compute_metrics(args_tuple):
    fname, data_dir, tokenizer_name, max_lines = args_tuple
    
    from transformers import AutoTokenizer  # local import for process safety
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    lang_id = fname.replace(".jsonl.gz", "")
    path = os.path.join(data_dir, fname)
    texts = read_jsonl_gz(path, max_lines=max_lines)
    total_chars = total_words = total_tokens = unk_count = 0
    
    for t in texts:
        total_chars += len(t)
        words = t.split()
        total_words += len(words)
        tokens = tokenizer(t)["input_ids"]
        total_tokens += len(tokens)
        unk_count += tokens.count(tokenizer.unk_token_id)
        
    chars_per_token = total_chars / total_tokens if total_tokens else 0
    pieces_per_word = total_tokens / total_words if total_words else 0
    unk_rate = unk_count / total_tokens if total_tokens else 0
    
    return {
        "language": lang_id,
        "chars_per_token": chars_per_token,
        "pieces_per_word": pieces_per_word,
        "unknown_rate": unk_rate,
        "num_texts": len(texts),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max_lines", type=int, default=10000)
    parser.add_argument("--output_csv", default="tokenizer_coverage.csv")
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".jsonl.gz")])
    task_args = [(f, args.data_dir, args.tokenizer, args.max_lines) for f in files]

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(compute_metrics, task_args), total=len(task_args)))

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\n Saved metrics for {len(df)} languages to {args.output_csv}")

if __name__ == "__main__":
    main()
