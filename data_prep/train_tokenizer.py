import argparse
import os
import datasets

from transformers import AutoTokenizer
from datasets import load_dataset, Features, Value
from tqdm import tqdm

def get_training_corpus(dataset):
    for example in tqdm(dataset, desc="Streaming corpus"):
        yield example["text"]

def main():
    parser = argparse.ArgumentParser(description="Train a new Llama 3.2 1b tokenizer on multilingual data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=250000, help="Tokenizer vocab size")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Base model")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Chunk size for streaming")
    args = parser.parse_args()

    # load data and tokenizer
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Data directory not found at {args.data_dir}")

    dataset = load_dataset(
        "json",
        data_files=f"{args.data_dir}/*/*.jsonl.gz",
        split="train",
        streaming=True,
    )
    dataset = dataset.remove_columns(["id", "metadata"])
    
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    training_corpus = get_training_corpus(dataset)

    # train and save new tokenizer
    new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, args.vocab_size)
    new_tokenizer.save_pretrained(args.output_dir)
    print(f"new tokenizer with vocab size {args.vocab_size} saved to {args.output_dir}")

    # optional test
    print("\n--- Testing the new tokenizer ---")
    reloaded_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    sample_text = "This is a test sentence in English. Y esta es una frase de prueba en español."
    tokens = reloaded_tokenizer.tokenize(sample_text)
    print(f"Sample text: {sample_text}")
    print(f"Tokens: {tokens}")
    print("---------------------------------")

if __name__ == "__main__":
    main()
