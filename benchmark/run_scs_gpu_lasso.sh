#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JULIA="${JULIA:-$ROOT/.julia-bin/julia}"
SCS_GPU_DEPOT="${SCS_GPU_DEPOT:-/tmp/pdcs_scs_gpu_depot}"
GPU="${GPU:-0}"
TOL="${TOL:-1e-6}"
TIME_LIMIT="${TIME_LIMIT:-600}"
VERBOSE_LEVEL="${VERBOSE_LEVEL:-2}"
CACHE=""
WARMUP_CACHE=""
OUTPUT_DIR=""

usage() {
    cat <<'EOF'
Usage: run_scs_gpu_lasso.sh --cache FILE --output-dir DIR [options]
  --warmup-cache FILE
  --gpu INDEX
  --tol VALUE
  --time-limit SECONDS
  --verbose-level 0|1|2

The isolated project pins CUDA_Runtime_jll 11.8 because the official
SCS_GPU_jll artifact currently provides CUDA 11.x builds only.
EOF
}

while (($#)); do
    case "$1" in
        --cache) CACHE="$2"; shift 2 ;;
        --cache=*) CACHE="${1#*=}"; shift ;;
        --warmup-cache) WARMUP_CACHE="$2"; shift 2 ;;
        --warmup-cache=*) WARMUP_CACHE="${1#*=}"; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --gpu) GPU="$2"; shift 2 ;;
        --gpu=*) GPU="${1#*=}"; shift ;;
        --tol) TOL="$2"; shift 2 ;;
        --tol=*) TOL="${1#*=}"; shift ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --time-limit=*) TIME_LIMIT="${1#*=}"; shift ;;
        --verbose-level) VERBOSE_LEVEL="$2"; shift 2 ;;
        --verbose-level=*) VERBOSE_LEVEL="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -x "$JULIA" ]] || { echo "Julia is not executable: $JULIA" >&2; exit 2; }
[[ -f "$CACHE" ]] || { echo "Cache is missing: $CACHE" >&2; exit 2; }
[[ -n "$OUTPUT_DIR" ]] ||
    { echo "--output-dir is required" >&2; exit 2; }
[[ -z "$WARMUP_CACHE" || -f "$WARMUP_CACHE" ]] ||
    { echo "Warm-up cache is missing: $WARMUP_CACHE" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be an index" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPU"
    "JULIA_DEPOT_PATH=$SCS_GPU_DEPOT"
    "$JULIA"
    "--project=$ROOT/benchmark/scs_gpu_env"
    "$ROOT/benchmark/ill_conditioned_lasso_scs_gpu.jl"
    --cache "$CACHE"
    --output-dir "$OUTPUT_DIR"
    --tol "$TOL"
    --time-limit "$TIME_LIMIT"
    --verbose-level "$VERBOSE_LEVEL"
)
if [[ -n "$WARMUP_CACHE" ]]; then
    command+=(--warmup-cache "$WARMUP_CACHE")
fi

echo "RUNNER solver=scs_gpu gpu=$GPU tolerance=$TOL cache=$CACHE"
"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/solver.raw.log"
exit "${PIPESTATUS[0]}"
