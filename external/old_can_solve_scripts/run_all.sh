#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JULIA_BIN="${ROOT_DIR}/.julia-bin/julia-1.10.4/bin/julia"
INPUT_ROOT="${HOME}/PDCS_CBLIB"
OLD_SOURCE_DIR="${ROOT_DIR}/external/PDCS-main"
GPU_SPEC="auto"
TIME_LIMIT="3600"
TOLERANCE="1e-6"
PRINT_FREQUENCY="20000"
JULIA_THREADS="14"
USE_AGGRESSIVE="true"
MODEL_LOADER="moi"
RUN_ID="old_pdcs_exp_$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-root) INPUT_ROOT="$2"; shift 2 ;;
        --old-source-dir) OLD_SOURCE_DIR="$2"; shift 2 ;;
        --gpus) GPU_SPEC="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --tolerance) TOLERANCE="$2"; shift 2 ;;
        --print-frequency) PRINT_FREQUENCY="$2"; shift 2 ;;
        --julia-threads) JULIA_THREADS="$2"; shift 2 ;;
        --use-aggressive) USE_AGGRESSIVE="$2"; shift 2 ;;
        --model-loader) MODEL_LOADER="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --julia) JULIA_BIN="$2"; shift 2 ;;
        --help|-h)
            printf '%s\n' \
                "Usage: $0 [options]" \
                "  --input-root PATH" \
                "  --old-source-dir PATH" \
                "  --gpus auto|INDEX[,INDEX...]" \
                "  --time-limit SECONDS (default: 3600)" \
                "  --tolerance VALUE (default: 1e-6)" \
                "  --print-frequency ITERATIONS (default: 20000)" \
                "  --julia-threads COUNT (default: 14)" \
                "  --use-aggressive true|false (default: true)" \
                "  --model-loader moi|cbf (default: moi; historical scripts used moi)" \
                "  --output-dir PATH"
            exit 0
            ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

INPUT_ROOT="$(readlink -f "$INPUT_ROOT")"
OLD_SOURCE_DIR="$(readlink -f "$OLD_SOURCE_DIR")"
[[ -d "$OLD_SOURCE_DIR" ]] || { printf 'Missing old source: %s\n' "$OLD_SOURCE_DIR" >&2; exit 2; }
[[ "$USE_AGGRESSIVE" == "true" || "$USE_AGGRESSIVE" == "false" ]] || {
    printf '%s\n' '--use-aggressive must be true or false' >&2
    exit 2
}
[[ "$MODEL_LOADER" == "moi" || "$MODEL_LOADER" == "cbf" ]] || {
    printf '%s\n' '--model-loader must be moi or cbf' >&2
    exit 2
}
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/results/${RUN_ID}"
fi
mkdir -p "$OUTPUT_DIR/cases"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"

if [[ "$GPU_SPEC" == "auto" ]]; then
    GPU_SPEC="$({
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits |
            awk -F, '{gsub(/ /,""); if ($2 < 100 && $3 < 10) print $1}' |
            paste -sd, -
    })"
fi
[[ -n "$GPU_SPEC" ]] || { printf 'No idle GPU found.\n' >&2; exit 2; }
IFS=',' read -r -a GPUS <<< "$GPU_SPEC"

CASES=(batch batchs101006m batchs121208m batchs151208m batchs201210m enpro56)
INPUT_PREFIX="cblib_misocp2socp_presolved/exp_cone"
for case_name in "${CASES[@]}"; do
    input="${INPUT_ROOT}/${INPUT_PREFIX}/${case_name}.cbf.gz"
    [[ -f "$input" ]] || { printf 'Missing input: %s\n' "$input" >&2; exit 2; }
done

started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'old_source=%s\n' "$OLD_SOURCE_DIR"
    printf 'input_root=%s\n' "$INPUT_ROOT"
    printf 'gpus=%s\n' "$GPU_SPEC"
    printf 'time_limit_seconds=%s\n' "$TIME_LIMIT"
    printf 'tolerance=%s\n' "$TOLERANCE"
    printf 'print_frequency=%s\n' "$PRINT_FREQUENCY"
    printf 'julia_threads=%s\n' "$JULIA_THREADS"
    printf 'use_aggressive=%s\n' "$USE_AGGRESSIVE"
    printf 'model_loader=%s\n' "$MODEL_LOADER"
    printf 'julia=%s\n' "$JULIA_BIN"
    printf 'started_utc=%s\n' "$started_utc"
} > "$OUTPUT_DIR/environment.txt"
if git -C "$OLD_SOURCE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$OLD_SOURCE_DIR" rev-parse HEAD > "$OUTPUT_DIR/old_source_commit.txt"
    git -C "$OLD_SOURCE_DIR" status --short > "$OUTPUT_DIR/old_source_status.txt"
fi
find "$OLD_SOURCE_DIR" -type f \
    \( -name 'rpdhg_alg_gpu*.jl' -o -name 'exp_proj*.jl' \
       -o -name '*block_proj*.cu' -o -name '*block_proj*.ptx' \
       -o -name 'libfew_block_proj.so' \) \
    -print0 | sort -z | xargs -0 -r sha256sum > "$OUTPUT_DIR/old_source_sha256.txt"

export JULIA_DEPOT_PATH="${ROOT_DIR}/.julia-depot:${JULIA_DEPOT_PATH:-}"
status=0
outer_timeout=$(( ${TIME_LIMIT%.*} + 300 ))
next_case=0
while (( next_case < ${#CASES[@]} )); do
    pids=()
    for gpu in "${GPUS[@]}"; do
        (( next_case < ${#CASES[@]} )) || break
        case_name="${CASES[$next_case]}"
        case_dir="${OUTPUT_DIR}/cases/${case_name}"
        mkdir -p "$case_dir"
        printf 'LAUNCH case=%s gpu=%s output=%s\n' "$case_name" "$gpu" "$case_dir"
        JULIA_NUM_THREADS="$JULIA_THREADS" CUDA_VISIBLE_DEVICES="$gpu" \
            timeout --signal=TERM --kill-after=60 "$outer_timeout" \
            "$JULIA_BIN" --startup-file=no -O1 --project="$SCRIPT_DIR" \
            "$SCRIPT_DIR/run_old_case.jl" \
            --input "${INPUT_ROOT}/${INPUT_PREFIX}/${case_name}.cbf.gz" \
            --output-dir "$case_dir" \
            --time-limit "$TIME_LIMIT" \
            --tolerance "$TOLERANCE" \
            --print-frequency "$PRINT_FREQUENCY" \
            --device gpu \
            --use-aggressive "$USE_AGGRESSIVE" \
            --model-loader "$MODEL_LOADER" \
            --old-source-dir "$OLD_SOURCE_DIR" \
            > "$case_dir/launcher.log" 2>&1 &
        pids+=("$!")
        next_case=$((next_case + 1))
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
done

PYTHONDONTWRITEBYTECODE=1 python3 "$SCRIPT_DIR/summarize.py" "$OUTPUT_DIR" |
    tee "$OUTPUT_DIR/summary.log"
printf 'OLD_PDCS_RUN_COMPLETE output=%s status=%s\n' "$OUTPUT_DIR" "$status"
exit "$status"
