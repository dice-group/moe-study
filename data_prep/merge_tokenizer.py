import argparse
import gzip
import json
import glob
import os
import time

from collections import Counter
from transformers import AutoTokenizer

def exhaustive_merge(Vt, tV):
    """
    Vi: llama vocab     Vt: target multilingual (NLBB) vocab     tV: sequence of tokens
    Returns set of new tokens (from Vt) that should be added to Vi
    """
    t = list(tV)
    updated = True
    while updated:
        updated = False
        i = 0
        while i < len(t) - 1:
            prev, nxt = t[i], t[i+1]
            merged = prev + nxt
            if merged in Vt:
                t = t[:i] + [merged] + t[i+2:]
                updated = True
                i = max(i - 1, 0)
            else:
                i += 1
    return t


def vocab_extend(Vi, Vt, docs, m, tokenizer_fn, max_size=250_000, freq_output=None):
    start = time.time()

    Vi0 = set(Vi)        # freeze original base vocab
    C = Vt - Vi0         # candidate tokens to consider
    F = Counter()

    total_docs = len(docs)
    for idx, d in enumerate(docs, 1):
        tV = tokenizer_fn(d)
        tV_merged = exhaustive_merge(Vt, tV)

        # count candidate token occurrences
        for tok in tV_merged:
            if tok in C:
                F[tok] += 1

        if idx % 1_000 == 0 or idx == total_docs:
            elapsed = time.time() - start
            print(f"Processed {idx}/{total_docs} documents "
                  f"({idx/total_docs:.1%}) in {elapsed/60:.2f} min")
            if F:
                print(f"Top tokens so far: {F.most_common(5)}")

    # save full frequency table if requested
    if freq_output:
        os.makedirs(os.path.dirname(freq_output), exist_ok=True)
        with open(freq_output, "w", encoding="utf-8") as f:
            for token, freq in F.most_common():
                f.write(f"{token}\t{freq}\n")

    # prune by min frequency
    Tpruned = [t for t, freq in F.items() if freq >= m]
    Tpruned.sort(key=lambda t: F[t], reverse=True)

    # cap vocab size
    num_to_add = max_size - len(Vi0)
    Tfinal = set(Tpruned[:max(0, num_to_add)])

    return Vi0 | Tfinal


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

    Vfinal = vocab_extend(
        Vi=Vi,
        Vt=Vt,
        docs=all_docs,
        m=args.min_freq,
        tokenizer_fn=base_tok.tokenize,
        freq_output=os.path.join(args.output_dir, "token_frequencies.tsv"),
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
