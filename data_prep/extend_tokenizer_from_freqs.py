import os
import argparse
import pandas as pd
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", required=True)
    parser.add_argument("--lang_dir", required=True, help="Path containing language subfolders")
    parser.add_argument("--alloc_csv", required=True, help="CSV file with token_alloc column")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_tokenizer)
    print(f"Loaded base tokenizer with {len(tok)} tokens")

    alloc_df = pd.read_csv(args.alloc_csv)
    alloc = dict(zip(alloc_df["language"], alloc_df["token_alloc"]))

    new_tokens = set()
    for lang in os.listdir(args.lang_dir):
        lang_path = os.path.join(args.lang_dir, lang, f"{lang}_tokens_freq.csv")
        if not os.path.exists(lang_path):
            continue
        if lang not in alloc:
            print(f"{lang} not found in allocation CSV — skipping")
            continue
        n = int(alloc[lang])
        df = pd.read_csv(lang_path)
        top_tokens = df["token"].head(n).tolist()
        new_tokens.update(top_tokens)
        print(f"{lang}: requested {n}, added {len(top_tokens)} (available {len(df)})")

    # Extend and save tokenizer
    added = tok.add_tokens(list(new_tokens))
    print(f"Added {added} new tokens. Total vocab size: {len(tok)}")
    os.makedirs(args.output_dir, exist_ok=True)
    tok.save_pretrained(args.output_dir)
    print(f"Extended tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
