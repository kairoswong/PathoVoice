#!/bin/bash
set -euo pipefail

declare -a JOB_PIDS=()

cleanup() {
    echo "[INFO] Stopping training jobs launched by this script..."
    for pid in "${JOB_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${JOB_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 130
}
trap cleanup INT TERM

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Environment
CONDA_ENV="${CONDA_ENV:-pathovoice}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    echo "[INFO] Activating conda environment '$CONDA_ENV' (current: '${CONDA_DEFAULT_ENV:-none}')"
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    if [[ -z "$CONDA_BASE" ]]; then
        echo "[ERROR] conda not found. Activate manually with: conda activate $CONDA_ENV"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "[INFO] Activated conda environment: $CONDA_DEFAULT_ENV"
fi

# Config
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("${MODELS:-tiny base small medium}")
    read -r -a MODELS <<< "${MODELS}"
fi

TRAINING_MODULE="asr.main.trainer"
MULTITASK="${MULTITASK:-false}"
CONFIG_DIR="${CONFIG_DIR:-./models/asr/configs}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-2000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-10}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
LOAD_STABILIZE="${LOAD_STABILIZE:-20}"
LOG_DIR="${LOG_DIR:-./logs/train}"
mkdir -p "$LOG_DIR"

# Functions
check_nvidia() {
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[ERROR] nvidia-smi not found. Install NVIDIA drivers first."
        exit 1
    fi
}

gpu_count() {
    nvidia-smi -L | wc -l
}

gpu_task_count() {
    local gpu="$1"
    nvidia-smi --query-compute-apps=gpu --format=csv,noheader,nounits | \
        grep -w "$gpu" | wc -l
}

find_free_gpu() {
    local total_gpus best_gpu=-1 best_mem=-1
    local free_mem task_count
    total_gpus=$(gpu_count)

    for ((gpu=0; gpu<total_gpus; gpu++)); do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
        task_count=$(gpu_task_count "$gpu")

        if [[ $free_mem -ge $MEMORY_THRESHOLD && $task_count -lt $JOBS_PER_GPU ]]; then
            if [[ $free_mem -gt $best_mem ]]; then
                best_gpu=$gpu
                best_mem=$free_mem
            elif [[ $free_mem -eq $best_mem ]] && (( RANDOM % 2 )); then
                best_gpu=$gpu
            fi
        fi
    done

    echo "$best_gpu"
}

launch_job() {
    local gpu="$1"
    local model="$2"
    local config_path="${CONFIG_DIR}/${model}.yaml"
    local log_file="${LOG_DIR}/${model}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
    local flags=()

    if [[ ! -f "$config_path" ]]; then
        echo "[ERROR] Config file not found: $config_path"
        exit 1
    fi
    if [[ "$MULTITASK" == "true" ]]; then
        flags+=(--multitask)
    fi

    echo "[INFO] Starting model: $model on GPU:$gpu (log: $log_file)"
    CUDA_VISIBLE_DEVICES="$gpu" \
        python3 -u -m "$TRAINING_MODULE" \
        --config-path "$config_path" \
        "${flags[@]}" \
        > "$log_file" 2>&1 &
    JOB_PIDS+=("$!")
}

# Main
check_nvidia
if [[ $(gpu_count) -eq 0 ]]; then
    echo "[ERROR] No GPUs detected."
    exit 1
fi

for model in "${MODELS[@]}"; do
    while true; do
        gpu=$(find_free_gpu)
        if [[ "$gpu" -ge 0 ]]; then
            launch_job "$gpu" "$model"
            echo "[INFO] Waiting ${LOAD_STABILIZE}s for GPU memory to stabilize..."
            sleep "$LOAD_STABILIZE"
            break
        fi
        echo "[WAIT] No free GPU available, retrying in ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    done
done

failed=0
for pid in "${JOB_PIDS[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [[ "$failed" -eq 0 ]]; then
    echo "[DONE] All training jobs finished successfully."
else
    echo "[ERROR] One or more training jobs failed. Check logs in $LOG_DIR."
    exit 1
fi
