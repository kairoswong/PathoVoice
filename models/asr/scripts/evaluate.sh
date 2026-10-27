#!/bin/bash
set -euo pipefail

# Environment
CONDA_ENV="${CONDA_ENV:-pathovoice}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    echo "[INFO] Activating conda environment '$CONDA_ENV' (current: '${CONDA_DEFAULT_ENV:-none}')"
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
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

TASK="${TASK:-transcribe}"
SEED="${SEED:-3407}"
AUGMENT="${AUGMENT:-true}"
MULTITASK="${MULTITASK:-false}"
TASK_TAG="transcribe"
if [[ "$MULTITASK" == "true" ]]; then
    TASK_TAG="multitask"
fi
AUGMENT_TAG=""
if [[ "$AUGMENT" == "true" ]]; then
    AUGMENT_TAG="_aug"
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./models/asr/checkpoints/${TASK_TAG}${AUGMENT_TAG}/seed${SEED}}"
CONFIG_DIR="${CONFIG_DIR:-./models/asr/configs}"
TEST_AUDIO_DIR="${TEST_AUDIO_DIR:-./models/asr/datasets/test_set/wav}"
TEST_LABEL_FILE="${TEST_LABEL_FILE:-./models/asr/datasets/test_set/transcript}"
TEST_TRANSLATE_FILE="${TEST_TRANSLATE_FILE:-./models/asr/datasets/test_set/translate}"
N_MELS="${N_MELS:-80}"
TESTING_BATCH_SIZE="${TESTING_BATCH_SIZE:-10}"
LOG_DIR="${LOG_DIR:-./logs/evaluate}"
mkdir -p "$LOG_DIR"

# Functions
evaluate_model() {
    local model="$1"
    local config_path="${CONFIG_DIR}/${model}.yaml"
    local model_path="${CHECKPOINT_DIR}/${model}"
    local log_file="${LOG_DIR}/eval_${model}_${TASK}.log"
    local flags=(
        --config-path "$config_path"
        --model-path "$model_path"
        --test-audio-dir "$TEST_AUDIO_DIR"
        --test-label-file "$TEST_LABEL_FILE"
        --task "$TASK"
        --n-mels "$N_MELS"
        --testing-batch-size "$TESTING_BATCH_SIZE"
    )

    if [[ ! -f "$config_path" ]]; then
        echo "[ERROR] Config file not found: $config_path"
        return 1
    fi
    if [[ "$TASK" == "translate" ]]; then
        flags+=(--test-translate-file "$TEST_TRANSLATE_FILE")
    fi
    if [[ -n "${METRICS:-}" ]]; then
        flags+=(--metrics "$METRICS")
    fi
    if [[ -n "${OUTPUT_CSV_DIR:-}" ]]; then
        mkdir -p "$OUTPUT_CSV_DIR"
        flags+=(--output-csv "${OUTPUT_CSV_DIR}/${model}_${TASK}.csv")
    fi

    echo "[INFO] Evaluating $model ($TASK) -> $log_file"
    if ! python3 -u -m asr.main.evaluate "${flags[@]}" > "$log_file" 2>&1; then
        echo "[ERROR] Evaluation failed for $model, see $log_file"
        return 1
    fi
}

# Main
for model in "${MODELS[@]}"; do
    evaluate_model "$model"
done

echo "[DONE] All evaluations completed."
