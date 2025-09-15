import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import csv
import tempfile

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

def _output_path(filename: str) -> str:
    """Return the full path for a generated TSV file."""
    return os.path.join(OUTPUT_DIR, filename)

# write rows line‑by‑line to a temp file
def _write_samples_to_temp(samples_iter, final_path: str) -> None:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    # Create a temp file in the same directory so rename works atomically
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=os.path.dirname(final_path))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as tmp_file:
            writer = None
            for row in samples_iter:
                if writer is None:
                    # Initialise writer with fieldnames from the first row
                    writer = csv.DictWriter(tmp_file, fieldnames=row.keys(), delimiter="\t")
                    writer.writeheader()
                writer.writerow(row)

        # replace the target file with the temp file
        os.replace(tmp_path, final_path)
    except Exception:
        # Clean up the temporary file on error
        try:
            os.remove(tmp_path)
        except OSError:
            pass  # ignore if the file was already removed or never created
        raise


def process_language_subset(subset: str, name: str, doc_count: int) -> None:
    out_file = f"{subset}_sample.tsv"
    out_path = _output_path(out_file)

    if os.path.isfile(out_path):
        print(f"Skipping {name} ({subset}) – {out_file} already exists.")
        return

    print(f"Processing language: {name} ({subset})")
    try:
        fw = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name=subset,
            split="train",
            streaming=True,
        )

        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, doc_count)

        # Build a generator that yields exactly ``sample_size`` rows
        def _sample_gen():
            for i, example in enumerate(fw):
                # TODO: Review this decision
                # NOTE: We are sampling in series, while this might reduce diversity of sources, it maintains consistency and takes less time
                if i >= sample_size:
                    break
                yield example

        _write_samples_to_temp(_sample_gen(), out_path)
        
        print(f"Saved {sample_size} samples to {out_path}")
    except Exception as e:
        print(f"Error processing {name} ({subset}): {e}")


def process_english() -> None:
    out_file = "english_sample.tsv"
    out_path = _output_path(out_file)

    if os.path.isfile(out_path):
        print(f"Skipping English – {out_file} already exists.")
        return

    print("Processing English data from fineweb...")
    try:
        fw = load_dataset(
            "HuggingFaceFW/fineweb",
            name="default",
            split="train",
            streaming=True,
        )
        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, 25_900_000_000)

        def _sample_gen():
            for i, example in enumerate(fw):
                if i >= sample_size:
                    break
                yield example

        _write_samples_to_temp(_sample_gen(), out_path)
    except Exception as e:
        print(f"Error processing English data: {e}")

def main() -> None:
    # Load metadata CSV once
    metadata_df = pd.read_csv(METADATA_PATH)
    
    rows_list = [row for _, row in metadata_df.iterrows()]

    # Process each language/subset row
    for row in tqdm(rows_list, desc='Processing FineWeb2 Languages'):
        process_language_subset(row["subset"], row["name"], int(row["documents"]))

    # Finally handle English (default) split
    process_english()


if __name__ == "__main__":
    main()