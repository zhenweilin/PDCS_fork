#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JULIA="${JULIA:-$HOME/.juliaup/bin/julia}"
GPU="${GPU:-3}"
TOL="${TOL:-1e-6}"
TIME_LIMIT="${TIME_LIMIT:-18000}"
VERBOSE="${VERBOSE:-2}"
INPUT_FOLDER="${INPUT_FOLDER:-/data/operationgpt/zhenwei/pdcs-paper-scales-n100/cbf}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-$ROOT/benchmark/results/multi_period_port/cupdcs_parallel}"
WORKERS="${WORKERS:-}"

usage() {
    cat <<'EOF'
Usage: run_multi_period_port_parallel.sh [options]

  --input-folder PATH    Folder containing .cbf.gz files
                         (default: /data/operationgpt/zhenwei/pdcs-paper-scales-n100/cbf)
  --output-folder PATH   Raw-log output directory
                         (default: benchmark/results/multi_period_port/cupdcs_parallel)
  --gpu INDEX            CUDA device index (default: 3)
  --tol VALUE            Absolute and relative tolerance (default: 1e-6)
  --time-limit SECONDS   Per-instance solver time limit (default: 18000)
  --verbose LEVEL        Solver verbosity: 0, 1, or 2 (default: 2)
  --workers N            JumpRW parser workers (default: Julia threads)
  -h, --help             Show this help
EOF
}

while (($#)); do
    case "$1" in
        --input-folder)      INPUT_FOLDER="$2"; shift 2 ;;
        --input-folder=*)    INPUT_FOLDER="${1#*=}"; shift ;;
        --output-folder)     OUTPUT_FOLDER="$2"; shift 2 ;;
        --output-folder=*)   OUTPUT_FOLDER="${1#*=}"; shift ;;
        --gpu)               GPU="$2"; shift 2 ;;
        --gpu=*)             GPU="${1#*=}"; shift ;;
        --tol)               TOL="$2"; shift 2 ;;
        --tol=*)             TOL="${1#*=}"; shift ;;
        --time-limit)        TIME_LIMIT="$2"; shift 2 ;;
        --time-limit=*)      TIME_LIMIT="${1#*=}"; shift ;;
        --verbose)           VERBOSE="$2"; shift 2 ;;
        --verbose=*)         VERBOSE="${1#*=}"; shift ;;
        --workers)           WORKERS="$2"; shift 2 ;;
        --workers=*)         WORKERS="${1#*=}"; shift ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -x "$JULIA" ]] || { echo "Julia is not executable: $JULIA" >&2; exit 2; }
[[ -d "$INPUT_FOLDER" ]] || { echo "Input folder does not exist: $INPUT_FOLDER" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be an integer index" >&2; exit 2; }
[[ "$VERBOSE" =~ ^[0-2]$ ]] || { echo "--verbose must be 0, 1, or 2" >&2; exit 2; }

mkdir -p "$OUTPUT_FOLDER"

julia_args=(
    --input_folder "$INPUT_FOLDER"
    --output_folder "$OUTPUT_FOLDER"
    --tolerance "$TOL"
    --verbose "$VERBOSE"
)
if [[ -n "$TIME_LIMIT" ]]; then
    julia_args+=(--time_limit "$TIME_LIMIT")
fi
if [[ -n "$WORKERS" ]]; then
    julia_args+=(--workers "$WORKERS")
fi

command=(
    env
    -u LD_LIBRARY_PATH
    "PATH=/usr/local/cuda-12.5/bin:$PATH"
    "CUDA_HOME=/usr/local/cuda-12.5"
    "JULIA_DEPOT_PATH=$ROOT/.julia-depot"
    "PDCS_SKIP_GPU_PRECOMPILE=1"
    "CUDA_VISIBLE_DEVICES=$GPU"
    "$JULIA"
    "--threads=auto"
    "--project=$ROOT"
    "$ROOT/benchmark/multi_period_port_cupdcs_parallel.jl"
    "${julia_args[@]}"
)

echo "RUNNER solver=cuPDCS_parallel gpu=$GPU tolerance=$TOL input=$INPUT_FOLDER output=$OUTPUT_FOLDER"
"${command[@]}" 2>&1 | tee "$OUTPUT_FOLDER/batch.raw.log"
exit "${PIPESTATUS[0]}"

