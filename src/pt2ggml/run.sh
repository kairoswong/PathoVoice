#!/usr/bin/env bash
# pt2ggml: .pt -> ggml .bin -> quantize -> verify
#
# Usage:
#   ./src/pt2ggml/run.sh models/asr/checkpoints/transcribe_aug/seed3407/tiny/main.pt
#   ./src/pt2ggml/run.sh tiny.pt -q q5_1,q8_0
#
set -euo pipefail

usage() {
    sed -n '2,9p' "$0"
    exit 1
}

# -----------------------------------
# Args

declare -a QUANT_TYPES=()
CHECK_ONLY=0
FORCE=0
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--out-dir)    OUT_DIR="$2"; shift 2 ;;
        -n|--name)       NAME="$2"; shift 2 ;;
        -q|--quantize)   IFS=, read -r -a QUANT_TYPES <<< "$2"; shift 2 ;;
        --check-only)    CHECK_ONLY=1; shift ;;
        -f|--force)      FORCE=1; shift ;;
        --skip-build)    SKIP_BUILD=1; shift ;;
        -h|--help)       usage ;;
        -*)              echo "[error] unknown option: $1" >&2; usage ;;
        *)               PT="$1"; shift ;;
    esac
done

[[ -n "${PT:-}" ]] || usage

# -----------------------------------
# Paths

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHISPER_SRC="$REPO_ROOT/whisper.cpp"
BUILD_DIR="$WHISPER_SRC/build"

PYTHON="${PYTHON:-python}"
CONDA_ENV="${CONDA_ENV:-pathovoice}"
if [[ "$CHECK_ONLY" == 0 && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$CONDA_BASE" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
fi

[[ -f "$PT" ]] || { echo "[error] not found: $PT" >&2; exit 1; }
PT="$(realpath "$PT")"
OUT_DIR="${OUT_DIR:-$(dirname "$PT")}"
NAME="${NAME:-ggml-$(basename "${PT%.*}")}"
BIN_PATH="$OUT_DIR/$NAME.bin"

find_tool() {
    local candidate
    for candidate in "$BUILD_DIR/bin/$1" "$BUILD_DIR/bin/Release/$1" "$BUILD_DIR/$1"; do
        [[ -x "$candidate" ]] && { echo "$candidate"; return 0; }
    done
    find "$BUILD_DIR" -type f -name "$1" -perm -u+x 2>/dev/null | head -1
}

# -----------------------------------
# Build whisper.cpp when needed

build_whisper() {
    [[ "$SKIP_BUILD" == 1 ]] && { echo "[error] binaries missing and --skip-build given" >&2; exit 1; }
    echo "[build] compiling whisper.cpp (first run only)..."
    cmake -S "$WHISPER_SRC" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
}

NEED_QUANTIZE=0
[[ ${#QUANT_TYPES[@]} -gt 0 ]] && NEED_QUANTIZE=1
[[ -z "$(find_tool whisper-bench)" || ( "$NEED_QUANTIZE" == 1 && -z "$(find_tool whisper-quantize)" ) ]] \
    && build_whisper

BENCH_EXE="$(find_tool whisper-bench)"
QUANTIZE_EXE="$(find_tool whisper-quantize)"
[[ -n "$BENCH_EXE" ]] || { echo "[error] whisper-bench not found after build" >&2; exit 1; }

# -----------------------------------
# Convert

echo
echo "=== pt2ggml ==="
cd "$REPO_ROOT"
CONVERT_ARGS=("$PT" -o "$BIN_PATH")
[[ "$CHECK_ONLY" == 1 ]] && CONVERT_ARGS+=(--check)
[[ "$FORCE" == 1 ]] && CONVERT_ARGS+=(--force)
"$PYTHON" -m src.pt2ggml.convert "${CONVERT_ARGS[@]}"
[[ "$CHECK_ONLY" == 1 ]] && exit 0

# -----------------------------------
# Verify by loading the model

verify_model() {
    local path="$1" output
    output="$("$BENCH_EXE" -m "$path" 2>&1 || true)"
    grep -qE 'wrong size in model file|wrong shape in model file|failed to load model|failed to initialize|unknown tensor' <<<"$output" \
        && return 1
    grep -q 'model size' <<<"$output" && grep -q 'encode time' <<<"$output"
}

verify_model "$BIN_PATH" || { echo "[error] model failed to load" >&2; exit 1; }
echo "[verify] load OK"

# -----------------------------------
# Quantize

for qtype in "${QUANT_TYPES[@]}"; do
    qtype="$(tr '[:upper:]' '[:lower:]' <<<"${qtype// /}")"
    [[ -n "$qtype" ]] || continue

    out="$OUT_DIR/$NAME-$qtype.bin"
    echo
    echo "[quantize] $qtype"
    "$QUANTIZE_EXE" "$BIN_PATH" "$out" "$qtype" >/dev/null

    "$PYTHON" -m src.pt2ggml.convert "$out" --check >/dev/null && verify_model "$out" \
        || { rm -f "$out"; echo "[error] quantize $qtype produced a corrupt file (removed)" >&2; exit 1; }

    printf '  OK (%s MiB)\n' "$(du -m "$out" | cut -f1)"
done

# -----------------------------------
# Summary

echo
echo "=== files ==="
find "$OUT_DIR" -maxdepth 1 -name "$NAME*.bin" -printf '%s\t%f\n' |
    sort -rn |
    while IFS=$'\t' read -r size file; do
        printf '  %-40s %7d MiB\n' "$file" "$(( (size + 524288) / 1048576 ))"
    done
