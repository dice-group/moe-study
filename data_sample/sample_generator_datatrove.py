#!/usr/bin/env python3
# Sample usage:
#   python data_sample/sample_generator_datatrove.py <sampling_plan.csv> <output_dir> --num-proc 12
# or legacy module mode:
#   bash data_sample/pylauncher.sh normal data_sample.sample_generator_datatrove
import argparse
import os
import pandas as pd
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor
from datatrove.executor import SlurmPipelineExecutor

# Adding this to avoid: "429 Client Error: Too Many Requests for url ... We had to rate limit you, you hit the quota of 5000 resolvers requests per 5 minutes period."
RANDOM_DELAY_LIMIT = 3600 # the maximum number of seconds to delay the start of each task.

DOCS_PER_SEC = 500  # Approx. docs processed per second by a single task

SAMPLE_PERCENTAGE = 10          # % of rows to sample (adjust as needed)
METADATA_PATH = "data_dir/fineweb2_metadata/misc/leftover_l2v.csv" # covering all selected languages
OUTPUT_DIR = "data_dir/fw_l2v_samples/" # folder where *.tsv files will be written
MIN_DOC_COUNT = 1_000_000  # minimum threshold to apply sampling logic
#MIN_DOC_COUNT = 0  # minimum threshold to apply sampling logic

TASKS_PER_SUBSET = 1
ENGLISH_DOC_COUNT = 25_900_000_000 # 25.9 billion documents in fineweb
ENGLISH_UNDERSAMPLED_COUNT = ENGLISH_DOC_COUNT // 50 # Undersampling English since it's highly overrepresented # bringing it to ~500 million

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

def _output_path(*parts: str) -> str:
    """Return the full path for a generated file, joining any number of
    sub‑paths with the output directory."""
    return os.path.join(OUTPUT_DIR, *parts)


def _get_subset_args(subset: str, sample_size: int):
    # English is sampled from a different dataset
    ds_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{subset}/train"
    if subset.lower() == 'english':
        ds_path = f"hf://datasets/HuggingFaceFW/fineweb/data/" # All of the data, for other subsets look into: https://huggingface.co/datasets/HuggingFaceFW/fineweb#using-%F0%9F%8F%AD-datatrove
    
    output_filepath = _output_path('samples', subset)
    
    jsonl_writer = JsonlWriter(output_filepath)
    parquet_reader = ParquetReader(ds_path, limit=sample_size, file_progress=True, doc_progress=True)
    
    return jsonl_writer, parquet_reader

# Executor factory functions
def get_lpe_instance(subset: str, sample_size: int, logging_dir_base: str | None = None):
    jsonl_writer, parquet_reader = _get_subset_args(subset, sample_size)
    if logging_dir_base:
        log_dir = os.path.join(logging_dir_base, subset)
    else:
        log_dir = _output_path('slurm_dir', subset)
    os.makedirs(log_dir, exist_ok=True)
    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            parquet_reader,
            jsonl_writer
        ],
        tasks=TASKS_PER_SUBSET,
        logging_dir=log_dir,
    )
    return pipeline_exec

def _seconds_to_hhmmss(total_seconds: int) -> str:
    """Convert seconds to a Slurm‑compatible HH:MM:SS string."""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_spe_instance(subset: str, sample_size: int, logging_dir_base: str | None = None):
    jsonl_writer, parquet_reader = _get_subset_args(subset, sample_size)

    if logging_dir_base:
        log_dir = os.path.join(logging_dir_base, subset)
    else:
        log_dir = _output_path('slurm_dir', subset) # this needs to be different for every subset
    os.makedirs(log_dir, exist_ok=True)
    
    estimated_seconds = max(sample_size // DOCS_PER_SEC, 3600) + RANDOM_DELAY_LIMIT
    estimated_time_str = _seconds_to_hhmmss(estimated_seconds)
    
    pipeline_exec = SlurmPipelineExecutor(
        pipeline=[
            parquet_reader,
            jsonl_writer
        ],
        tasks=TASKS_PER_SUBSET,
        job_name=f"{subset}__sampling",
        requeue=False, # If the job fails, shows the proper error message
        qos="", # leaving it blank because the default qos causes error
        logging_dir=log_dir, # Scripts and other aux files are also stored in this
        time=estimated_time_str, # estimated based on 500 docs per second
        partition="normal", # Cluster specific, remains same for PC²
        cpus_per_task=8,
        mem_per_cpu_gb=5,
        randomize_start_duration=RANDOM_DELAY_LIMIT,
    )
    return pipeline_exec

def _write_sample_sizes(sample_info):
    
    sample_sizes_tsv_filepath = _output_path("sample_sizes.tsv")
    
    with open(sample_sizes_tsv_filepath, "w", encoding="utf-8") as f:
        f.write("subset\tname\tresource_category\tsample_size\n")
        for subset, lang_name, res_cat, sample_size in sample_info:
            f.write(f"{subset}\t{lang_name}\t{res_cat}\t{sample_size}\n")

def _run_plan_mode(plan_csv: str, output_dir: str, num_proc: int, logging_dir: str | None = None) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(plan_csv)
    required = {"subset", "documents"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[ERROR] Missing columns in plan CSV {plan_csv}: {sorted(missing)}")

    exec_factory = get_spe_instance if os.getenv("SLURM_ACTIVE", "false").lower() == "true" else get_lpe_instance
    sample_sizes = []
    executor_list = []
    for _, row in df.iterrows():
        subset = str(row["subset"])
        sample_size = int(row["documents"])
        sample_sizes.append((subset, subset, "plan", sample_size))
        executor_list.append(exec_factory(subset, sample_size, logging_dir))

    _write_sample_sizes(sample_sizes)
    for exe in executor_list:
        exe.run()


def _run_legacy_mode(logging_dir: str | None = None) -> None:
    # Load metadata CSV once
    metadata_df = pd.read_csv(METADATA_PATH)
    rows_list = [row for _, row in metadata_df.iterrows()]
    
    executor_list = []
    exec_factory = None
    # Decide execution mode
    if os.getenv("SLURM_ACTIVE", "false").lower() == "true":
        exec_factory = get_spe_instance
    else:
        exec_factory = get_lpe_instance
    
    # Compute the sample size beforehand and write it to a tsv file before starting executor run
    sample_sizes = []
    # For fineweb2
    for row in rows_list:
        doc_count = int(row["documents"])
        sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, doc_count)
        subset = row["subset"]
        lang_name = row["name"]
        res_cat = row["resource_category"]
        sample_sizes.append((subset, lang_name, res_cat, sample_size))
        # Get language executor
        executor_list.append(exec_factory(subset, sample_size, logging_dir))
    
    # For English (fineweb)
    eng_subset = 'english'
    eng_lang_name = 'English'
    eng_res_cat = 'high'
    eng_sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, ENGLISH_UNDERSAMPLED_COUNT)
    sample_sizes.append((eng_subset, eng_lang_name, eng_res_cat, eng_sample_size))
    # Get English executor
    executor_list.append(exec_factory(eng_subset, eng_sample_size, logging_dir))
    # Write the TSV before launching any executors
    _write_sample_sizes(sample_sizes)
    # Run all executors
    for exec in executor_list:
        exec.run()

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("plan_csv", nargs="?")
    parser.add_argument("output_dir", nargs="?")
    parser.add_argument("--num-proc", type=int, default=12)
    parser.add_argument("--logging-dir", type=str, default="")
    args, _ = parser.parse_known_args()

    if args.plan_csv and args.output_dir:
        _run_plan_mode(args.plan_csv, args.output_dir, args.num_proc, args.logging_dir or None)
        return

    _run_legacy_mode(args.logging_dir or None)

if __name__ == "__main__":
    main()
