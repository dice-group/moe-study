#!/bin/bash
# Source after activating Merlin, before any Python/GPU preflight. The archive
# contains unchanged startup dependencies, not adapters, data, or credentials.
if [[ "${1:-}" == --prepare ]]; then
    set -euo pipefail
    root=$2 bundle=$3 digest=$4
    case "$root" in /dev/shm/moe-study-cache-*) ;; *) exit 90;; esac
    umask 077
    exec 9>"$root.lock"
    flock -w 600 9
    [[ ! -e "$root" || -O "$root" ]] || exit 90
    if [[ -f "$root/.complete" ]]; then exit 0; fi
    [[ ! -e "$root" ]] || exit 90
    temporary="$root.tmp.$$"
    mkdir "$temporary"
    trap 'rm -rf "$temporary"' EXIT
    printf '%s  %s\n' "$digest" "$bundle" | sha256sum --check -
    tar -xzf "$bundle" -C "$temporary"
    touch "$temporary/.complete"
    mv "$temporary" "$root"
    exit 0
fi

_cache_repo=${PROOF_PIPELINE_REPO_ROOT:?}
_cache_campaign=$(jq -er .campaign_id "$_cache_repo/configs/cross_arch_study/96_languages.json") || return 90
_cache_manifest="$_cache_repo/outputs/pipeline_runs/$_cache_campaign/runtime/node_cache_ready.json"
_cache_digest=$(jq -er .sha256 "$_cache_manifest") || return 90
_cache_archive=$(jq -er .archive "$_cache_manifest") || return 90
_cache_libraries=$(jq -er .libraries_directory "$_cache_manifest") || return 90
_cache_python=$(jq -er .python_directory "$_cache_manifest") || return 90
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "$_cache_digest" =~ ^[0-9a-f]{64}$ ]] || return 90
export CROSS_ARCH_CACHE_ROOT="/dev/shm/moe-study-cache-${SLURM_JOB_ID}-${_cache_digest:0:12}"
cleanup_cross_arch_cache() {
    case "${CROSS_ARCH_CACHE_ROOT:-}" in
        /dev/shm/moe-study-cache-*) rm -rf "$CROSS_ARCH_CACHE_ROOT" "$CROSS_ARCH_CACHE_ROOT.lock";;
    esac
    case "${CROSS_ARCH_MODEL_CACHE_ROOT:-}" in
        /dev/shm/moe-study-model-*) rm -rf "$CROSS_ARCH_MODEL_CACHE_ROOT";;
    esac
}
trap cleanup_cross_arch_cache EXIT
timeout --signal=TERM --kill-after=30 630 bash "${BASH_SOURCE[0]}" --prepare \
    "$CROSS_ARCH_CACHE_ROOT" "$_cache_repo/$_cache_archive" "$_cache_digest" || return $?
export MOE_STUDY_LOCAL_TORCH_LIBS="$CROSS_ARCH_CACHE_ROOT/$_cache_libraries"
export MOE_STUDY_LOCAL_PYTHON="$CROSS_ARCH_CACHE_ROOT/$_cache_python"
_cache_interpreter=$(jq -er .python "$_cache_repo/configs/cross_arch_study/noctua2_runtime.json") || return 90
export LD_LIBRARY_PATH="$MOE_STUDY_LOCAL_TORCH_LIBS:$(dirname "$(dirname "$_cache_interpreter")")/lib:${LD_LIBRARY_PATH:-}"
printf '[NODE_CACHE] verified archive=%s root=%s\n' "$_cache_digest" "$CROSS_ARCH_CACHE_ROOT"
if [[ -n "${PROOF_PIPELINE_CROSS_ARCH_MODEL:-}" ]]; then
    case "$PROOF_PIPELINE_CROSS_ARCH_MODEL" in gemma2_9b|falcon_7b) ;; *) return 90;; esac
    export CROSS_ARCH_MODEL_CACHE_ROOT="/dev/shm/moe-study-model-${SLURM_JOB_ID}-${PROOF_PIPELINE_CROSS_ARCH_MODEL}"
    timeout --signal=TERM --kill-after=30 900 "$PROOF_PIPELINE_PYTHON" \
        -m pipeline_tools.cross_arch_model_cache --model "$PROOF_PIPELINE_CROSS_ARCH_MODEL" || return $?
fi
