# Sample usage: bash data_sample/pylauncher.sh normal data_sample.sample_generator
import os
import pandas as pd
import csv
import tempfile
import subprocess
import argparse
import multiprocessing
from tqdm import tqdm
from datasets import load_dataset

# Config
SAMPLE_PERCENTAGE = 1          # % of rows to sample (adjust as needed)
METADATA_PATH = "data_dir/fineweb2_metadata/llm_nested_samples/199_representative_mediods.csv" # covering all selected languages
OUTPUT_DIR = "data_dir/fw_samples/"                # folder where *.tsv files will be written
MIN_DOC_COUNT = 10_000  # minimum threshold to apply sampling logic
FLUSH_EVERY=1_000_000

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
def _write_samples_to_temp(samples_iter, sample_size, final_path: str) -> None:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    # Create a temp file in the same directory so rename works atomically
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=os.path.dirname(final_path))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as tmp_file:
            writer = None
            for i, row in enumerate(tqdm(samples_iter, total=sample_size), start=1):
                if writer is None:
                    # Initialise writer with fieldnames from the first row
                    writer = csv.DictWriter(tmp_file, fieldnames=row.keys(), delimiter="\t")
                    writer.writeheader()
                writer.writerow(row)
                
                if i % FLUSH_EVERY == 0:
                    print(f'Processed {i} documents\n', flush=True)
                    tmp_file.flush()

        # replace the target file with the temp file
        os.replace(tmp_path, final_path)
    except Exception:
        # Clean up the temporary file on error
        try:
            os.remove(tmp_path)
        except OSError:
            pass  # ignore if the file was already removed or never created
        raise


def _process_subset(subset: str | None, name: str, doc_count: int, is_english: bool = False) -> None:
    """
    Load the appropriate FineWeb dataset, compute the sample size, stream,
    and write the first ``sample_size`` rows to a TSV file.
    """
    print(f'Total documents: {doc_count}\n', flush=True)
    
    out_file = f"{subset}_sample.tsv" if subset else "english_sample.tsv"
    out_path = _output_path(out_file)

    if os.path.isfile(out_path):
        print(f"Skipping {name} – {out_file} already exists.")
        return

    print(f"Processing {'English' if is_english else name} data...")
    try:
        dataset_kwargs = {
            "name": "default" if is_english else subset,
            "split": "train",
            "streaming": True,
        }
        # NOTE: May lead to "Too Many Requests" issue if using without a login / access token
        fw = load_dataset(
            "HuggingFaceFW/fineweb" if is_english else "HuggingFaceFW/fineweb-2",
            **dataset_kwargs,
        )
        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, doc_count)
        
        print(f'Sample size: {sample_size}\n', flush=True)

        def _sample_gen():
            # TODO: Review this decision
            # NOTE: We are sampling in series, while this might reduce diversity of sources, it maintains consistency and takes less time
            for i, example in enumerate(fw):
                if i >= sample_size:
                    break
                yield example

        _write_samples_to_temp(_sample_gen(), sample_size, out_path)
        print(f"Saved {sample_size} samples to {out_path}")
    except Exception as e:
        print(f"Error processing {name}: {e}")


def process_language_subset(subset: str, name: str, doc_count: int) -> None:
    _process_subset(subset, name, doc_count, is_english=False)

def process_english() -> None:
    _process_subset(
        subset=None,
        name="English",
        doc_count=25_900_000_000,   # total docs for the default English split
        is_english=True,
    )


def _submit_slurm_job(row: dict | None, is_english: bool = False) -> None:
    """
    Submit a Slurm job for either a language row or the English split.
    """
    job_name = "english" if is_english else f"{row['subset']}"
    log_dir = os.path.join(OUTPUT_DIR, "slurm_logs")
    os.makedirs(log_dir, exist_ok=True)

    base_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={os.path.join(log_dir, '%x__slurm-%j.out')}",
        "--cpus-per-task=4",
        "--mem=10G", # Each chunk (parquet) is around 4.85 GB (maximum)
        "--time=20:00:00",   # TODO: make this dynamic if needed
        "data_sample/pylauncher.sh",
        "nested",
        f"{__spec__.name}",
    ]

    if is_english:
        base_cmd.append("--english")
    else:
        base_cmd.extend([
            f"--subset={row['subset']}",
            f"--name=\"{row['name']}\"",
            f"--documents={row['documents']}",
        ])
    
    # NOTE: This will already pass on the loaded module and activated environment to the job
    subprocess.run(base_cmd, check=True)


def _submit_english_slurm_job():
    _submit_slurm_job(row=None, is_english=True)

def _process_row(row):
    """Helper for multiprocessing – calls process_language_subset for a row."""
    process_language_subset(row["subset"], row["name"], int(row["documents"]))

def main() -> None:
    parser = argparse.ArgumentParser(description="FineWeb2 sample generator")
    parser.add_argument("--subset", type=str, help="If set, process only this subset")
    parser.add_argument("--name", type=str, help="Name of the subset (required with --subset)")
    parser.add_argument("--documents", type=int, help="Document count (required with --subset)")
    parser.add_argument("--english", action="store_true", help="Process only the English split")
    args = parser.parse_args()

    # Load metadata CSV once
    metadata_df = pd.read_csv(METADATA_PATH)

    # Single‑subset mode
    if args.subset:
        process_language_subset(args.subset, args.name, args.documents)
        return

    # English‑only mode
    if args.english:
        process_english()
        return

    rows_list = [row for _, row in metadata_df.iterrows()]

    # Decide execution mode
    if os.getenv("SLURM_ACTIVE", "false").lower() == "true":
        # Submit a Slurm job for each language row
        for row in rows_list:
            _submit_slurm_job(row)

        # Also submit a job for the English split
        _submit_english_slurm_job()
    else:
        # Multiprocessing: process language subsets in parallel
        with multiprocessing.Pool() as pool:
            pool.map(_process_row, rows_list)

        # Finally handle English (default) split
        process_english()
    
if __name__ == "__main__":
    main()