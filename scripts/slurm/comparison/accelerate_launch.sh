#!/usr/bin/env bash
set -euo pipefail

build_accelerate_launch() {
  local train_script=$1
  local accelerate_config=$2
  local repo_root=$3

  LAUNCH_PREFIX=()
  LAUNCH_CMD=()
  ENTRYPOINT=()

  local nnodes="${SLURM_NNODES:-1}"
  local node_id="${SLURM_NODEID:-0}"
  local master_addr_source="fallback"
  local master_host=""
  local master_addr=""

  if [[ -n "${MASTER_ADDR_OVERRIDE:-}" ]]; then
    master_addr="${MASTER_ADDR_OVERRIDE}"
    master_host="${MASTER_ADDR_OVERRIDE}"
    master_addr_source="override"
  elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    master_host=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_addr=$(getent ahosts "${master_host}" | awk '$1 ~ /^[0-9]+(\\.[0-9]+){3}$/ {print $1; exit}')
    if [[ -z "${master_addr}" ]]; then
      master_addr="${master_host}"
      master_addr_source="hostname"
    else
      master_addr_source="getent_ahosts"
    fi
  else
    master_addr="127.0.0.1"
    master_host="${master_addr}"
    master_addr_source="localhost"
  fi

  local master_port="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000) ))}"
  export MASTER_ADDR="${master_addr}"
  export MASTER_PORT="${master_port}"

  if [[ -z "${GPUS_PER_NODE:-}" || "${GPUS_PER_NODE}" -le 0 ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
      GPUS_PER_NODE=$(echo "${SLURM_GPUS_ON_NODE}" | awk -F: '{print $NF}' | grep -oE '[0-9]+' | head -n 1)
    elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
      IFS=',' read -ra _gpu_list <<< "${SLURM_JOB_GPUS}"
      GPUS_PER_NODE=${#_gpu_list[@]}
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -ra _gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
      GPUS_PER_NODE=${#_gpu_list[@]}
    else
      GPUS_PER_NODE=1
    fi
  fi

  local total_procs=$((nnodes * GPUS_PER_NODE))
  export TOTAL_PROCESSES="${total_procs}"

  if [[ -n "${accelerate_config}" ]]; then
    if [[ ! -f "${accelerate_config}" ]]; then
      echo "[ERROR] Accelerate config ${accelerate_config} not found." >&2
      exit 1
    fi
    export ACCELERATE_USE_FSDP="${ACCELERATE_USE_FSDP:-1}"

    if [[ "${nnodes}" -gt 1 ]]; then
      local wait_script="${WAIT_FOR_MASTER_SCRIPT:-${repo_root}/scripts/slurm/comparison/wait_for_master.sh}"
      local wait_body=":"
      if [[ -x "${wait_script}" ]]; then
        wait_body="bash ${wait_script} ${MASTER_ADDR} ${MASTER_PORT} ${MASTER_CONNECT_TIMEOUT:-90}"
      else
        echo "[WARN] wait_for_master script not found at ${wait_script}" >&2
      fi
      LAUNCH_PREFIX=(srun --ntasks="${nnodes}" --ntasks-per-node=1 --export=ALL)
      LAUNCH_CMD=(
        bash -lc
        "if [[ ${nnodes} -gt 1 && \$SLURM_NODEID -ne 0 ]]; then ${wait_body}; fi; \
accelerate launch \
  --config_file ${accelerate_config} \
  --num_machines ${nnodes} \
  --num_processes ${total_procs} \
  --machine_rank \$SLURM_NODEID \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  ${train_script} \"\$@\""
      )
      ENTRYPOINT=("_")
    else
      LAUNCH_CMD=(
        accelerate launch
        --config_file "${accelerate_config}"
        --num_machines 1
        --machine_rank 0
        --num_processes "${total_procs}"
        --main_process_ip "${MASTER_ADDR}"
        --main_process_port "${MASTER_PORT}"
        "${train_script}"
      )
    fi
  else
    if [[ "${GPUS_PER_NODE}" -ne 1 ]]; then
      echo "[ERROR] ACCELERATE_CONFIG not set for ${GPUS_PER_NODE} GPU(s)." >&2
      exit 1
    fi
    LAUNCH_CMD=(accelerate launch --num_processes 1 "${train_script}")
  fi

  echo "[INFO] dist config: nnodes=${nnodes} node_id=${node_id} gpus_per_node=${GPUS_PER_NODE} total_procs=${TOTAL_PROCESSES} master=${MASTER_ADDR}:${MASTER_PORT} source=${master_addr_source}"
}
