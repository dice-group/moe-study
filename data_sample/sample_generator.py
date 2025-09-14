import os
import pandas as pd
from datasets import load_dataset

# Config
SAMPLE_PERCENTAGE = 1          # % of rows to sample (adjust as needed)
METADATA_PATH = "data_dir/fineweb2_metadata/llm_nested_samples/199_representative_mediods.csv"
OUTPUT_DIR = "."                # folder where *.tsv files will be written

# Helpers
def _calc_sample_size(percent: float, estimate: int) -> int:
    """Return the number of rows to keep given a % and an estimated total."""
    return int(percent / 100 * estimate)

def _save_samples(samples: list[dict], filename: str) -> None:
    """Write a list of dicts to a TSV file."""
    df = pd.DataFrame(samples)
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(samples)} samples to {out_path}")


# Core processing
def process_language_subset(subset: str, lang_code: str) -> None:
    """
    Stream a given language subset from the fineweb‑2 dataset and write a
    sample file named ``<lang_code>_sample.tsv``.
    """
    print(f"Processing language: {lang_code} ({subset})")
    try:
        fw = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name=subset,
            split="train",
            streaming=True,
        )
        # 10 000 is a rough size‑estimate for most non‑English subsets
        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, 10_000)

        samples = []
        for i, example in enumerate(fw):
            if i >= sample_size:
                break
            samples.append(example)

        _save_samples(samples, f"{lang_code}_sample.tsv")
    except Exception as e:
        print(f"Error processing {lang_code}: {e}")

def process_english() -> None:
    """Stream the default English split from the fineweb dataset."""
    print("Processing English data from fineweb...")
    try:
        fw = load_dataset(
            "HuggingFaceFW/fineweb",
            name="default",
            split="train",
            streaming=True,
        )
        # English corpus is much larger; we estimate 100 000 rows
        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, 100_000)

        samples = []
        for i, example in enumerate(fw):
            if i >= sample_size:
                break
            samples.append(example)

        _save_samples(samples, "english_sample.tsv")
    except Exception as e:
        print(f"Error processing English data: {e}")

def main() -> None:
    # Load metadata CSV once
    metadata_df = pd.read_csv(METADATA_PATH)

    # Process each language/subset row
    for _, row in metadata_df.iterrows():
        process_language_subset(row["subset"], row["code"])

    # Finally handle English (default) split
    process_english()


if __name__ == "__main__":
    main()