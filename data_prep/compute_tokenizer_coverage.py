#!/usr/bin/env python3
import os
import gzip
import json
import argparse
import pandas as pd
from tqdm import tqdm
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

def compute_metrics(tokenizer, texts):
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
    return chars_per_token, pieces_per_word, unk_rate

def main():
    parser = argparse.ArgumentParser(description="Compute tokenizer coverage metrics per language file.")
    parser.add_argument("--data_dir", required=True, help="Directory with .jsonl.gz language files.")
    parser.add_argument("--tokenizer", required=True, help="HF model name or path (e.g., meta-llama/Meta-Llama-3-8B).")
    parser.add_argument("--max_lines", type=int, default=10000, help="Max lines to sample per language file.")
    parser.add_argument("--output_csv", default="tokenizer_coverage.csv", help="Output CSV path.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    results = []
    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".jsonl.gz")])
    for fname in tqdm(files, desc="Processing languages"):
        lang_id = fname.replace(".jsonl.gz", "")
        texts = read_jsonl_gz(os.path.join(args.data_dir, fname), max_lines=args.max_lines)
        if not texts:
            continue
        cpt, ppw, unk = compute_metrics(tokenizer, texts)
        results.append({
            "language": lang_id,
            "chars_per_token": cpt,
            "pieces_per_word": ppw,
            "unknown_rate": unk,
            "num_texts": len(texts)
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n Saved metrics for {len(df)} languages to {args.output_csv}")

if __name__ == "__main__":
    main()
