#!/bin/bash
set -eu
CUR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # This gets overriden on the next 'source' call

source $CUR_SCRIPT_DIR/setup/env.sh

OUTPUT_DIR=$PROJECT_ROOT/data_dir/fineweb2_metadata/
mkdir -p $OUTPUT_DIR
# Download language distribution csv
wget https://raw.githubusercontent.com/huggingface/fineweb-2/refs/heads/main/fineweb2-language-distribution.csv -O $OUTPUT_DIR/fineweb2-language-distribution.csv
