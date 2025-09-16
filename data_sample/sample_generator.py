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
METADATA_PATH = "data_dir/fineweb2_metadata/llm_nested_samples/debug.csv" # covering all selected languages
OUTPUT_DIR = "data_dir/fw_samples/"                # folder where *.tsv files will be written
MIN_DOC_COUNT = 10_000  # minimum threshold to apply sampling logic
FLUSH_EVERY=10_000

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
            for i, row in enumerate(samples_iter, start=1):
                if writer is None:
                    # Initialise writer with fieldnames from the first row
                    writer = csv.DictWriter(tmp_file, fieldnames=row.keys(), delimiter="\t")
                    writer.writeheader()
                writer.writerow(row)
                
                if i % FLUSH_EVERY == 0:
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
        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, 25_900_000_000) # As per https://huggingface.co/datasets/HuggingFaceFW/fineweb, the train split has 25.9 billion documents

        def _sample_gen():
            for i, example in enumerate(fw):
                if i >= sample_size:
                    break
                yield example

        _write_samples_to_temp(_sample_gen(), out_path)
    except Exception as e:
        print(f"Error processing English data: {e}")
        
def _submit_slurm_job(row):
    """Create and submit a simple Slurm job that runs this script for one subset."""

    # Create a unique job name (e.g., based on the row name and a timestamp)
    job_name = f"{row['subset']}"

    # Directory where Slurm logs will be stored – adjust as needed
    log_dir = os.path.join(OUTPUT_DIR, "slurm_logs")
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Submit via sbatch with job‑name and output‑file options
    # NOTE: This will already pass on the loaded module and activated environment to the job
    subprocess.run([
        "sbatch",
        f"--job-name={job_name}",
        f"--output={os.path.join(log_dir, '%x__slurm-%j.out')}",
        f"--cpus-per-task=4",
        f"--mem=2G",
        f"--time=10:00:00", # TODO: Compute this based on number of documents
        f"data_sample/pylauncher.sh", # Command starts here
        f"nested",
        f"{__spec__.name}",
        f"--subset={row['subset']}",
        f"--name=\"{row['name']}\"",
        f"--documents={row['documents']}",
    ], check=True)


def _submit_english_slurm_job():
    """Submit a Slurm job that processes the English split."""
    log_dir = os.path.join(OUTPUT_DIR, "slurm_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # NOTE: This will already pass on the loaded module and activated environment to the job
    subprocess.run([
        "sbatch",
        f"--job-name=english",
        f"--output={os.path.join(log_dir, '%x__slurm-%j.out')}",
        f"--cpus-per-task=4",
        f"--mem=2G",
        f"--time=10:00:00",
        f"data_sample/pylauncher.sh",
        f"nested",
        f"{__spec__.name}",
        f"--english",
    ], check=True)

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