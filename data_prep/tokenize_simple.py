import gzip
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

SRC_DIR = Path("/data/project_data/moe_study/fw_samples/preview_subset")
OUT_DIR = Path("/data/project_data/moe_study/tokenized/test_llama_3.2-1b")
MODEL_NAME = "meta-llama/Llama-3.2-1B"
MAX_LEN = 1024


def iter_jsonl_gz_files(src_dir):
    for path in sorted(src_dir.glob("*.jsonl.gz")):
        lang = path.name.replace(".jsonl.gz", "")
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text")
                    if isinstance(text, str) and text.strip():
                        yield {
                            "text": text,
                            "language": lang,
                        }
                except json.JSONDecodeError:
                    continue


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_generator(
        lambda: iter_jsonl_gz_files(SRC_DIR)
    )

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text", "language"],
        num_proc=8,
    )


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(OUT_DIR)


if __name__ == "__main__":
    main()
