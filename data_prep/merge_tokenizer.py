import argparse
import gzip
import json
import glob
import os

from collections import Counter
from transformers import AutoTokenizer

def exhaustive_merge(Vi, Vt, tV):
    Tnew = set()
    updated = True
    while updated:
        updated = False
        for i in range(len(tV) - 1):
            prev, nxt = tV[i], tV[i+1]
            merged = prev + nxt
            if merged in Vt:
                tV = tV[:i] + [merged] + tV[i+2:]
                Tnew.add(merged)
                updated = True
                break
    return Tnew


def vocab_extend(Vi, Vt, docs, m, tokenizer_fn):
    V = set(Vi)
    F = Counter()
    T = set()
    for d in docs:
        tV = tokenizer_fn(d)
        Tnew = exhaustive_merge(Vi, Vt, tV)
        V |= Tnew
        T |= Tnew
        F.update(Tnew)
    Tpruned = {t for t in T if F[t] >= m}
    return set(Vi) | Tpruned


def read_jsonl_gz(path, text_key="text", limit=None):
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if text_key in record:
                yield record[text_key]
                count += 1
                if limit and count >= limit:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", required=True)
    parser.add_argument("--target_tokenizer", required=True)
    parser.add_argument("--data_dir", required=True, help="Directory of .jsonl.gz files")
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional doc limit per file")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer)
    target_tok = AutoTokenizer.from_pretrained(args.target_tokenizer)

    Vi = set(base_tok.get_vocab().keys())
    Vt = set(target_tok.get_vocab().keys())

    all_docs = []
    files = glob.glob(f"{args.data_dir}/*.jsonl.gz")
    print(f"Found {len(files)} files in {args.data_dir}")

    for f in files:
        print(f"Reading: {f}")
        docs = list(read_jsonl_gz(f, text_key="text", limit=args.limit))
        all_docs.extend(docs)

    Vfinal = vocab_extend(
        Vi=Vi,
        Vt=Vt,
        docs=all_docs,
        m=args.min_freq,
        tokenizer_fn=base_tok.tokenize
    )

    print(f"Original vocab size: {len(Vi)}")
    print(f"Extended vocab size: {len(Vfinal)}")
    new_tokens = list(Vfinal - Vi)
    print(f"Discovered {len(new_tokens)} new tokens")
    if new_tokens:
        print("Sample new tokens:", new_tokens[:50])
        
    base_tok.add_tokens(list(Vfinal - Vi))
    os.makedirs(args.output_dir, exist_ok=True)
    base_tok.save_pretrained(args.output_dir)
    print(f"Extended tokenizer saved to {args.output_dir}")

#TODO: max 250k vocab size
#TODO: how big shouold be the vocab size of the mutlilingual, self trained tokenizer then? i set to 300k for now, is this reasonable?