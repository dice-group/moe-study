import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def tokenize_all_files(input_dir: Path, tokenizer_name: str, out_dir: Path, num_proc: int = 8):
    """
    Tokenize ALL .jsonl.gz files under input_dir and save ONE merged HF dataset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    buffer = []

    files = sorted(input_dir.rglob("*.jsonl.gz"))
    print(f"Found {len(files)} files")

    for file_path in files:
        print(f"Tokenizing {file_path}")
        ds = load_dataset("json", data_files=str(file_path), split="train", streaming=True)

        for ex in ds:
            text = ex.get("text", "")
            toks = tokenizer(text, add_special_tokens=False)
            buffer.append(toks)

    print(f"Total examples: {len(buffer)}")
    merged = Dataset.from_list(buffer)
    merged.save_to_disk(str(out_dir))

    print(f"Saved ONE merged dataset to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--num-proc", type=int, default=8)
    args = ap.parse_args()

    tokenize_all_files(args.input_dir, args.tokenizer, args.out_dir, args.num_proc)

if __name__ == "__main__":
    main()

"""
python tokenize_training_data.py \
  --input-dir /data/project_data/moe_study/fw_samples/preview_subset \
  --out-dir /data/project_data/moe_study/tokenized/llama_3.1-8b/preview_subset \
  --tokenizer meta-llama/Llama-3.1-8B
  scp -r joeldag@lola.cs.uni-paderborn.de:/data/project_data/moe_study/tokenized/llama_3.1-8b/preview_subset /data/project_data/moe_study/tokenized/llama_3.1-8b/
"""