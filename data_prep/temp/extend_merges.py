import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer", required=True)
    parser.add_argument("--target_tokenizer", required=True)
    parser.add_argument("--output_dir", default="extended_tokenizer")
    args = parser.parse_args()

    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    target_tok = AutoTokenizer.from_pretrained(args.target_tokenizer, use_fast=True, local_files_only=True)

    base_json_path = Path(base_tok.name_or_path) / "tokenizer.json"
    target_json_path = Path(target_tok.name_or_path) / "tokenizer.json"
    base_data = json.load(open(base_json_path, encoding="utf-8"))
    target_data = json.load(open(target_json_path, encoding="utf-8"))

    base_vocab = base_data["model"]["vocab"]
    base_merges = base_data["model"]["merges"]
    target_vocab = target_data["model"]["vocab"]
    target_merges = target_data["model"]["merges"]

    base_merges_set = {tuple(m) for m in base_merges}
    new_merges = [m for m in target_merges if tuple(m) not in base_merges_set]
    print("[DEBUG] new merges to add:", len(new_merges))
    base_merges.extend(new_merges)

    next_id = max(base_vocab.values()) + 1
    for token in target_vocab:
        if token not in base_vocab:
            base_vocab[token] = next_id
            next_id += 1

    base_data["model"]["vocab"] = base_vocab
    base_data["model"]["merges"] = base_merges

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(base_data, f, ensure_ascii=False, indent=2)

    base_dir = Path(args.base_tokenizer)
    for fname in ["tokenizer_config.json", "special_tokens_map.json"]:
        src = base_dir / fname
        if src.exists():
            dst = out_dir / fname
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(f" finished, saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
