import argparse
import os
import datasets
import pandas as pd
import random

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

def sample_language_data(lang_dir, final_count, sample_size):
    ds = load_dataset(
        "json",
        data_files=f"{lang_dir}/*.jsonl.gz",
        split="train"
    )
    ds = ds.remove_columns([c for c in ["id", "metadata"] if c in ds.column_names])

    # if enough data available, randomly choose amount of docs from that data, oversample otherwise
    if final_count <= sample_size:
        ds = ds.shuffle(seed=42).select(range(final_count))
    else:
        multiplier = final_count // sample_size
        remainder = final_count % sample_size
        datasets_to_concat = [ds] * multiplier
        if remainder > 0:
            datasets_to_concat.append(ds.shuffle(seed=42).select(range(remainder)))
        ds = concatenate_datasets(datasets_to_concat)

    return ds

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
    parser.add_argument("--sample_sizes_tsv", type=str, required=True, help="TSV with sample_size and final_count")
    args = parser.parse_args()
    
    df = pd.read_csv(args.sample_sizes_tsv, sep="\t")
    datasets_list = []
    for _, row in df.iterrows():
        subset = row["subset"]
        sample_size = int(row["sample_size"])
        final_count = int(row["final_count"])
        lang_dir = os.path.join(args.data_dir, subset)

        if not os.path.exists(lang_dir):
            print(f"skipping {subset}, dir not found")
            continue

        print(f"Processing {subset}: sample_size={sample_size}, final_count={final_count}")
        lang_ds = sample_language_data(lang_dir, final_count, sample_size)
        datasets_list.append(lang_ds)

    full_dataset = concatenate_datasets(datasets_list)
    
    old_tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    training_corpus = get_training_corpus(full_dataset)

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
