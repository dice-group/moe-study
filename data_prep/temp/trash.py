from datasets import load_dataset
from transformers import AutoTokenizer

data_path = "/data/joel/MERLIN/AdaMoe/raw_dataset/fineweb_english.jsonl/*.jsonl.gz"
save_path = "/data/MERLIN/tokenized_test_data"
tokenizer = AutoTokenizer.from_pretrained("/data/MERLIN/multilingual_tokenizer/250K_vocab_size/trained_multilingual_tokenizer_llama-3.2-1B")

ds = load_dataset("json", data_files=data_path, split="train")
ds = ds.map(lambda e: tokenizer(e["text"], truncation=True), batched=True)
ds.save_to_disk(save_path)
