# Define % of data to sample
# Read data_dir/fineweb2_metadata/llm_nested_samples/199_representative_mediods.csv
# For each unique language/row (subset), stream data (subset, split: train) from fineweb2 dataset on huggingface: HuggingFaceFW/fineweb-2
# Stream upto % of data sample and write it to the <lang_code>_sample file

# Read HuggingFaceFW/fineweb for English (subset default) and stream % of data sample
# Output to the english_sample file