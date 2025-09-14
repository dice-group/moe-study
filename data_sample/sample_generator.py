import os
import pandas as pd
from datasets import load_dataset

# Config
SAMPLE_PERCENTAGE = 1          # % of rows to sample (adjust as needed)
METADATA_PATH = "data_dir/fineweb2_metadata/llm_nested_samples/199_representative_mediods.csv" # covering all selected languages
OUTPUT_DIR = "data_dir/fw_samples/"                # folder where *.tsv files will be written
MIN_DOC_COUNT = 10_000  # minimum threshold to apply sampling logic

# Helpers
def _calc_sample_size(percent: float, estimate: int) -> int:
    """Return the number of rows to keep given a % and an estimated total."""
    # If the estimate is less than MIN_DOC_COUNT, keep all rows.
    if estimate < MIN_DOC_COUNT:
        return estimate

    # If the estimate is above MIN_DOC_COUNT, keep a base of MIN_DOC_COUNT
    # plus the requested percentage of the remaining rows.
    extra = (estimate - MIN_DOC_COUNT) * (percent / 100)
    return int(MIN_DOC_COUNT + extra)

def _save_samples(samples: list[dict], filename: str) -> None:
    """Write a list of dicts to a TSV file."""
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame(samples)
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(samples)} samples to {out_path}")


# Core processing
def process_language_subset(subset: str, name: str, doc_count: int) -> None:
    """
    Stream a given language subset from the fineweb‑2 dataset and write a
    sample file named ``<lang_code>_sample.tsv``.
    """
    print(f"Processing language: {name} ({subset})")
    try:
        fw = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name=subset,
            split="train",
            streaming=True,
        )
        
        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, doc_count)

        samples = []
        # TODO: Review this decision
        # NOTE: We are sampling in series, while this might reduce diversity of sources, it maintains consistency and takes less time
        for i, example in enumerate(fw):
            if i >= sample_size:
                break
            samples.append(example)

        _save_samples(samples, f"{subset}_sample.tsv")
    except Exception as e:
        print(f"Error processing {name} ({subset}): {e}")

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
        # English corpus has 25.9 billion rows
        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, 25_900_000_000)

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
        process_language_subset(row["subset"], row["name"], int(row["documents"]))

    # Finally handle English (default) split
    process_english()


if __name__ == "__main__":
    main()