import argparse
import gzip
import json
import random
import os
import pandas as pd

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
    return t, T_new


def vocab_extend(Vi, Vt, docs, m, tokenizer_fn):
    V = set(Vi)
    T = set()
    F = Counter()

    total_docs = len(docs)
    for idx, d in enumerate(docs, 1):
        tV = tokenizer_fn(d)
        t_merged, T_new = exhaustive_merge(Vi, Vt, tV)
        V |= T_new
        T |= T_new
        for tok in T_new:
            F[tok] += t_merged.count(tok)
            
        if idx % 50 == 0 or idx == total_docs:     # show progress
            pct = 100 * idx / total_docs
            print(f"Processed {idx}/{total_docs} docs ({pct:.1f}%)")


    T_pruned = {t for t in T if F[t] > m}
    V_final = Vi | T_pruned
    return V_final, F


def read_jsonl_gz(path, text_key="text", limit=None, seed=42):
    lines = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if text_key in record:
                lines.append(record[text_key])
    if limit and len(lines) > limit:
        random.seed(seed)
        lines = random.sample(lines, limit)
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", required=True)
    parser.add_argument("--target_tokenizer", required=True)
    parser.add_argument("--data_dir", required=True, help="Directory of .jsonl.gz files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_freq", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit per file")
    parser.add_argument("--lang", required=True, help="Language code (e.g., hin_Deva)")
    args = parser.parse_args()

    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer)
    target_tok = AutoTokenizer.from_pretrained(args.target_tokenizer)

    Vi = set(base_tok.get_vocab().keys())
    Vt = set(target_tok.get_vocab().keys())
    
    lang_file = os.path.join(args.data_dir, f"{args.lang}.jsonl.gz")
    if not os.path.exists(lang_file):
        raise FileNotFoundError(f"Missing data file for {args.lang}: {lang_file}")

    lang_out_dir = os.path.join(args.output_dir, args.lang)
    os.makedirs(lang_out_dir, exist_ok=True)
    
    print(f"processing {args.lang}")
    docs = read_jsonl_gz(lang_file, text_key="text", limit=args.limit, seed=42)
    new_tokens, F = vocab_extend(
        Vi=Vi,
        Vt=Vt,
        docs=docs,
        m=args.min_freq,
        tokenizer_fn=base_tok.tokenize,
    )
    
    df = pd.DataFrame(
        [(tok, freq) for tok, freq in F.most_common() if tok in new_tokens],
        columns=["token", "frequency"]
    )
    df.to_csv(os.path.join(lang_out_dir, f"{args.lang}_tokens_freq.csv"), index=False, encoding="utf-8")
    print(f"Saved {len(new_tokens)} new tokens to {lang_out_dir} for language {args.lang}")