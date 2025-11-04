#!/bin/bash
set -euo pipefail

# Config
WANDB_PROJECT="multilingual_tokenizer_perplexity_eval"

# Base model job
base_job=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=loss-base
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=base-%j.out
set -euo pipefail
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="loss-base-\$SLURM_JOB_ID"
llamafactory train --config "./llamafactory_config_llama3.2-1B_base.yaml"
EOF
)
base_id=$(echo "$base_job" | awk '{print $4}')
echo "Submitted base training job: ${base_id}"

# Extended model job
ext_job=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=loss-extended
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=extended-%j.out
set -euo pipefail
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_NAME="loss-extended-\$SLURM_JOB_ID"
llamafactory train --config "./llamafactory_config_llama3.2-1B_extended_multilingual.yaml"
EOF
)
ext_id=$(echo "$ext_job" | awk '{print $4}')
echo "Submitted extended training job: ${ext_id}"
