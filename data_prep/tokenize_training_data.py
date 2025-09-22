import argparse, gzip, json

from pathlib import Path
from collections import Counter
from datasets import Dataset
from transformers import AutoTokenizer

def yield_examples(lang_dir):
    for f in lang_dir.glob("*.jsonl.gz"):
        with gzip.open(f, "rt", encoding="utf-8") as inp:
            for line in inp:
                try:
                    yield {"text": json.loads(line)["text"]}
                except Exception:
                    continue

def process_lang(lang_dir, out_dir, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    unk_id = tokenizer.unk_token_id
    counts = Counter()

    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=False, add_special_tokens=False)

        for ids in out["input_ids"]:
            counts["total"] += len(ids)
            counts["unk"] += sum(1 for i in ids if i == unk_id)
        return out

    ds = Dataset.from_generator(lambda: yield_examples(lang_dir))
    ds = ds.map(tokenize_fn, batched=True, num_proc=4)
    out_path = out_dir / lang_dir.name
    ds.save_to_disk(str(out_path))

    with open(out_dir / f"{lang_dir.name}.stats.json", "w") as out:
        json.dump({"language": lang_dir.name, **counts}, out)
    return lang_dir.name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lang_dirs = [d for d in args.parent_dir.iterdir() if d.is_dir()]
    for d in lang_dirs:
        print("Processing", d.name)
        process_lang(d, args.out_dir, args.tokenizer)

if __name__ == "__main__":
    main()
