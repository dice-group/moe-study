#!/bin/bash
set -eu
CUR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # This likely gets overriden on the next 'source' call
# Module loader specific to noctua2 computing cluster
source $CUR_SCRIPT_DIR/noctua2/module_loader.sh