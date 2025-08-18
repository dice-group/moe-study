#!/bin/bash
set -eu
CUR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # This gets overriden on the next 'source' call

source $CUR_SCRIPT_DIR/setup/env.sh

OUTPUT_DIR=$PROJECT_ROOT/data_dir/fineweb2_metadata/
mkdir -p $OUTPUT_DIR

download_if_missing() {
  local url=$1
  local dest=$2

  # Skip download if file already exists and is non‑empty
  if [[ -s "$dest" ]]; then
    echo "$dest already present – skipping download."
    return 0
  fi

  echo "Downloading $url to $dest"
  wget "$url" -O "$dest"
}

# Download files
download_if_missing \
  "https://raw.githubusercontent.com/huggingface/fineweb-2/refs/heads/main/fineweb2-language-distribution.csv" \
  "$OUTPUT_DIR/fineweb2-language-distribution.csv"

download_if_missing \
  "https://microsoft.github.io/linguisticdiversity/assets/lang2tax.txt" \
  "$OUTPUT_DIR/lang2tax.csv"

download_if_missing \
    "https://files.dice-research.org/projects/MERLIN/resources/lowres_ds/lang_resource_dataset.tsv" \
  "$OUTPUT_DIR/lang_resource_dataset.tsv"