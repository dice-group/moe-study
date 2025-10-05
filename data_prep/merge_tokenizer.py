import argparse
import gzip
import json
import glob
import os
import time

from collections import Counter
from transformers import AutoTokenizer

def exhaustive_merge(Vi, Vt, tV):
    """
    Vi: llama vocab     Vt: target multilingual (NLBB) vocab     tV: sequence of tokens
    Returns merged token sequence
    """
    t = list(tV)
    T_new = set()
    updated = True

    while updated:
        updated = False
        for i in range(len(t) - 1):
            tmerged = t[i] + t[i+1]
            if tmerged in Vt:
                t[i:i+2] = [tmerged]
                T_new.add(tmerged)
                updated = True
                break
    return T_new


def vocab_extend(Vi, Vt, docs, m, tokenizer_fn):
    V = set(Vi)
    T = set()
    F = Counter()

    total_docs = len(docs)
    for idx, d in enumerate(docs, 1):
        tV = tokenizer_fn(d)
        T_new = exhaustive_merge(Vi, Vt, tV)
        V |= T_new
        T |= T_new
        for tok in T_new:
            F[tok] +=1
            
        if idx % 50 == 0 or idx == total_docs:     # show progress
            pct = 100 * idx / total_docs
            print(f"Processed {idx}/{total_docs} docs ({pct:.1f}%)")


    T_pruned = {t for t in T if F[t] > m}
    V_final = Vi | T_pruned
    return V_final, F


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
    parser.add_argument("--min_freq", type=int, default=3)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit per file")
    parser.add_argument("--freq_output", default=None, help="Optional path to save token frequencies")
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

    Vfinal, F = vocab_extend(
        Vi=Vi,
        Vt=Vt,
        docs=all_docs,
        m=args.min_freq,
        tokenizer_fn=base_tok.tokenize,
    )

    print(f"Original vocab size: {len(Vi)}")
    print(f"Extended vocab size: {len(Vfinal)}")
    new_tokens = list(Vfinal - Vi)
    print(f"Discovered {len(new_tokens)} new tokens")
    if new_tokens:
        print("Sample new tokens:", new_tokens[:50])

    # Save extended tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    base_tok.add_tokens(list(Vfinal - Vi))
    base_tok.save_pretrained(args.output_dir)
    print(f"Extended tokenizer saved to {args.output_dir}")

    # Save frequencies if requested
    if args.freq_output:
        print(f"Saving token frequencies to {args.freq_output}")
        with open(args.freq_output, "w", encoding="utf-8") as fout:
            for tok, freq in F.most_common():
                fout.write(f"{tok}\t{freq}\n")