#!/usr/bin/env bash
# Single config file. Pick profile with PROFILE=...

PROFILE="${PROFILE:-llama32_1b}"

WANDB_PROJECT="${WANDB_PROJECT:-moe-study-comparison_labeled}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-matrix-${PROFILE}-$(date +%Y%m%d_%H%M%S)}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT:-$(pwd)}/LLaMA-Factory/data}"
TOKENIZED_BASE_DIR="${TOKENIZED_BASE_DIR:-/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/fw_l2v_samples_moe_study_comparison_labeled}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/hpc-prf-merlin/project_data/moe_study/moe_study_comparison_outputs_labeled}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT:-$(pwd)}/scripts/slurm/comparison/logs/matrix_labeled}"

RUN_MODE="${RUN_MODE:-slurm}"
RUN_APPROACHES="${RUN_APPROACHES:-cola hydralora adamole lora}"
FLASH_ATTN="${FLASH_ATTN:-fa2}"
AUTO_FIND_BATCH_SIZE="${AUTO_FIND_BATCH_SIZE:-false}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-32}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-100000}"
COMMON_GRAD_ACC="${COMMON_GRAD_ACC:-2}"

# shared LoRA setup for fair runs across approaches
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}"

DEFAULT_PARTITION="${DEFAULT_PARTITION:-gpu}"
DEFAULT_GPU_TYPE="${DEFAULT_GPU_TYPE:-h100}"
DEFAULT_GPU_COUNT="${DEFAULT_GPU_COUNT:-1}"
DEFAULT_CPUS_PER_TASK="${DEFAULT_CPUS_PER_TASK:-32}"
DEFAULT_MEM="${DEFAULT_MEM:-128G}"
DEFAULT_WALLTIME="${DEFAULT_WALLTIME:-27:00:00}"

# tier format:
# <tier_id>|<dataset_name>|<tokenizer_name>|<model_name_or_path>|<optional_tokenized_path_override>

case "${PROFILE}" in
  llama32_1b)
    LANGUAGE_TIERS=(
      "tier10|10_langs|llama-3.2-1B_tokenizer|meta-llama/Llama-3.2-1B|"
      # "tier96|96_langs|llama-3.2-1B_tokenizer|meta-llama/Llama-3.2-1B|"
      # "tier200|200_langs|llama-3.2-1B_tokenizer|meta-llama/Llama-3.2-1B|"
    )

    # from comparison_config_llama3_2_1b.json
    COLA_VARIANTS=(
      "paper-best|21|${COMMON_GRAD_ACC}|7300|7300|COLA_NUM_A=1 COLA_NUM_B=3 COLA_STRATEGY=fully"
    )
    HYDRALORA_VARIANTS=(
      "paper-best|17|${COMMON_GRAD_ACC}|6700|6700|USE_HYDRALORA_EXPERTS=True HYDRALORA_NUM_EXPERTS=3 HYDRALORA_TOP_K=1 HYDRALORA_NUM_B=3"
    )
    ADAMOLE_VARIANTS=(
      "paper-best|15|${COMMON_GRAD_ACC}|5100|5100|ADAMOLE_NUM_EXPERTS=4 ADAMOLE_AUX_LOSS_COEF=0.001 ADAMOLE_MAX_THRESHOLD=0.25"
    )
    LORA_VARIANTS=(
      "paper-best|21|${COMMON_GRAD_ACC}|7300|7300|"
    )
    ;;

  llama32_3b)
    LANGUAGE_TIERS=(
      # "tier10|10_langs|llama-3.2-3B_tokenizer|meta-llama/Llama-3.2-3B|"
      # "tier96|96_langs|llama-3.2-3B_tokenizer|meta-llama/Llama-3.2-3B|"
      # "tier200|200_langs|llama-3.2-3B_tokenizer|meta-llama/Llama-3.2-3B|"
    )

    # from comparison_config_llama3_2_3b.json
    COLA_VARIANTS=(
      "paper-best|11|${COMMON_GRAD_ACC}|3000|3000|COLA_NUM_A=1 COLA_NUM_B=3 COLA_STRATEGY=fully"
    )
    HYDRALORA_VARIANTS=(
      "paper-best|8|${COMMON_GRAD_ACC}|3000|3000|USE_HYDRALORA_EXPERTS=True HYDRALORA_NUM_EXPERTS=3 HYDRALORA_TOP_K=1 HYDRALORA_NUM_B=3"
    )
    ADAMOLE_VARIANTS=(
      "paper-best|6|${COMMON_GRAD_ACC}|3000|3000|ADAMOLE_NUM_EXPERTS=4 ADAMOLE_AUX_LOSS_COEF=0.001 ADAMOLE_MAX_THRESHOLD=0.25"
    )
    # no lora baseline in 3b json yet
    LORA_VARIANTS=()
    ;;

  llama31_8b)
    LANGUAGE_TIERS=(
      # "tier10|10_langs|llama-3.1-8B_tokenizer|meta-llama/Llama-3.1-8B|"
      # "tier96|96_langs|llama-3.1-8B_tokenizer|meta-llama/Llama-3.1-8B|"
      # "tier200|200_langs|llama-3.1-8B_tokenizer|meta-llama/Llama-3.1-8B|"
    )

    # from comparison_config_llama3_1_8b.json
    COLA_VARIANTS=(
      "paper-best|5|${COMMON_GRAD_ACC}|3000|3000|COLA_NUM_A=1 COLA_NUM_B=3 COLA_STRATEGY=fully"
    )
    HYDRALORA_VARIANTS=(
      "paper-best|3|${COMMON_GRAD_ACC}|3000|3000|USE_HYDRALORA_EXPERTS=True HYDRALORA_NUM_EXPERTS=3 HYDRALORA_TOP_K=1 HYDRALORA_NUM_B=3"
    )
    ADAMOLE_VARIANTS=(
      "paper-best|3|${COMMON_GRAD_ACC}|3000|3000|ADAMOLE_NUM_EXPERTS=4 ADAMOLE_AUX_LOSS_COEF=0.001 ADAMOLE_MAX_THRESHOLD=0.25"
    )
    # no lora baseline in 8b json yet
    LORA_VARIANTS=()
    ;;

  *)
    echo "[ERROR] Unknown PROFILE=${PROFILE}. Use: llama32_1b | llama32_3b | llama31_8b" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# optional explicit walltime overrides; if empty, runner derives from tier gpu-hours / gpu-count
# declare -A TIER_WALLTIME_MAP=(["tier10"]="27:00:00")
declare -A TIER_WALLTIME_MAP=()

declare -A TIER_GPU_COUNT_MAP=(
  ["tier10"]="${DEFAULT_GPU_COUNT}"
  ["tier96"]="${DEFAULT_GPU_COUNT}"
  ["tier200"]="${DEFAULT_GPU_COUNT}"
)

declare -A TIER_GPU_TYPE_MAP=(
  ["tier10"]="${DEFAULT_GPU_TYPE}"
  ["tier96"]="${DEFAULT_GPU_TYPE}"
  ["tier200"]="${DEFAULT_GPU_TYPE}"
)

declare -A TIER_PARTITION_MAP=(
  ["tier10"]="${DEFAULT_PARTITION}"
  ["tier96"]="${DEFAULT_PARTITION}"
  ["tier200"]="${DEFAULT_PARTITION}"
)

declare -A TIER_CPUS_PER_TASK_MAP=(
  ["tier10"]="${DEFAULT_CPUS_PER_TASK}"
  ["tier96"]="${DEFAULT_CPUS_PER_TASK}"
  ["tier200"]="${DEFAULT_CPUS_PER_TASK}"
)

declare -A TIER_MEM_MAP=(
  ["tier10"]="${DEFAULT_MEM}"
  ["tier96"]="${DEFAULT_MEM}"
  ["tier200"]="${DEFAULT_MEM}"
)
