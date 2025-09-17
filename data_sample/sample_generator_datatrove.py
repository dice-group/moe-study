# Sample usage: bash data_sample/pylauncher.sh normal data_sample.sample_generator_datatrove
import os
import pandas as pd
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor
from datatrove.executor import SlurmPipelineExecutor


SAMPLE_PERCENTAGE = 1          # % of rows to sample (adjust as needed)
METADATA_PATH = "data_dir/fineweb2_metadata/llm_nested_samples/199_representative_mediods.csv" # covering all selected languages
OUTPUT_DIR = "data_dir/fw_samples/"                # folder where *.tsv files will be written
MIN_DOC_COUNT = 10_000  # minimum threshold to apply sampling logic

TASKS_PER_SUBSET = 1
ENGLISH_DOC_COUNT = 25_900_000_000 # 25.9 billion documents in fineweb

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


def _get_subset_args(subset: str, doc_count: int):
    # English is sampled from a different dataset
    ds_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{subset}/train"
    if subset.lower() == 'english':
        ds_path = f"hf://datasets/HuggingFaceFW/fineweb/data/" # All of the data, for other subsets look into: https://huggingface.co/datasets/HuggingFaceFW/fineweb#using-%F0%9F%8F%AD-datatrove
    
    sample_size = _calc_sample_size(SAMPLE_PERCENTAGE, doc_count)
    
    output_filepath = _output_path('samples', subset)
    
    jsonl_writer = JsonlWriter(output_filepath)
    parquet_reader = ParquetReader(ds_path, limit=sample_size)
    
    return jsonl_writer, parquet_reader

# Executor factory functions
def get_lpe_instance(subset: str, name: str, doc_count: int):
    jsonl_writer, parquet_reader = _get_subset_args(subset, doc_count)
    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            parquet_reader,
            jsonl_writer
        ],
        tasks=TASKS_PER_SUBSET
    )
    return pipeline_exec

def get_spe_instance(subset: str, name: str, doc_count: int):
    jsonl_writer, parquet_reader = _get_subset_args(subset, doc_count)
    
    log_dir = _output_path('slurm_dir', subset) # this needs to be different for every subset
    os.makedirs(log_dir, exist_ok=True)
    
    pipeline_exec = SlurmPipelineExecutor(
        pipeline=[
            parquet_reader,
            jsonl_writer
        ],
        tasks=TASKS_PER_SUBSET,
        job_name=f"sampling__{subset}",
        requeue=False, # If the job fails, shows the proper error message
        qos="cont", # PC² specific QoS
        logging_dir=log_dir, # Scripts and other aux files are also stored in this
        time="20:00:00",
        partition="normal",
        cpus_per_task=4,
        mem_per_cpu_gb=5
    )
    return pipeline_exec

def main() -> None:
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
    
    # For fineweb2
    for row in rows_list:
        executor_list.append(exec_factory(row["subset"], row["name"], int(row["documents"])))
    # For English (fineweb)
    executor_list.append(exec_factory('english', 'English', ENGLISH_DOC_COUNT))    
    for exec in executor_list:
        exec.run()

if __name__ == "__main__":
    main()
