#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JULIA="${JULIA:-$ROOT/.julia-bin/julia}"
GPU="${GPU:-7}"
TOL="${TOL:-1e-6}"
TIME_LIMIT="${TIME_LIMIT:-600}"
VERBOSE_LEVEL="${VERBOSE_LEVEL:-2}"
DIRECT_SOLVE_METHOD="${DIRECT_SOLVE_METHOD:-cudss}"
CUCLARABEL_DEPOT="${CUCLARABEL_DEPOT:-$ROOT/.julia-depot-cuclarabel}"
CACHE=""
WARMUP_CACHE=""
OUTPUT=""

usage() {
    echo "Usage: $0 --cache FILE --output-dir DIR [--warmup-cache FILE] [--gpu INDEX]"
    echo "          [--tol VALUE] [--time-limit SECONDS]"
    echo "          [--direct-solve-method cudss|cudssmixed]"
}

while (($#)); do
    case "$1" in
        --cache) CACHE="$2"; shift 2 ;;
        --cache=*) CACHE="${1#*=}"; shift ;;
        --warmup-cache) WARMUP_CACHE="$2"; shift 2 ;;
        --warmup-cache=*) WARMUP_CACHE="${1#*=}"; shift ;;
        --output-dir) OUTPUT="$2"; shift 2 ;;
        --output-dir=*) OUTPUT="${1#*=}"; shift ;;
        --gpu) GPU="$2"; shift 2 ;;
        --gpu=*) GPU="${1#*=}"; shift ;;
        --tol) TOL="$2"; shift 2 ;;
        --tol=*) TOL="${1#*=}"; shift ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --time-limit=*) TIME_LIMIT="${1#*=}"; shift ;;
        --verbose-level) VERBOSE_LEVEL="$2"; shift 2 ;;
        --verbose-level=*) VERBOSE_LEVEL="${1#*=}"; shift ;;
        --direct-solve-method) DIRECT_SOLVE_METHOD="$2"; shift 2 ;;
        --direct-solve-method=*) DIRECT_SOLVE_METHOD="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$CACHE" && -f "$CACHE" ]] ||
    { echo "--cache must name an existing file" >&2; exit 2; }
[[ -z "$WARMUP_CACHE" || -f "$WARMUP_CACHE" ]] ||
    { echo "--warmup-cache must name an existing file" >&2; exit 2; }
[[ -n "$OUTPUT" ]] || { echo "--output-dir is required" >&2; exit 2; }
[[ "$DIRECT_SOLVE_METHOD" == "cudss" ||
   "$DIRECT_SOLVE_METHOD" == "cudssmixed" ]] ||
    { echo "invalid --direct-solve-method" >&2; exit 2; }

mkdir -p "$OUTPUT" "$CUCLARABEL_DEPOT"
RAW_LOG="$OUTPUT/cuclarabel.raw.log"
RESULT="$OUTPUT/cuclarabel_results.csv"
[[ ! -e "$RAW_LOG" && ! -e "$RESULT" ]] || {
    echo "Refusing to overwrite an existing CuClarabel result in $OUTPUT" >&2
    exit 2
}

export CUDA_VISIBLE_DEVICES="$GPU"
export JULIA_DEPOT_PATH="$CUCLARABEL_DEPOT:$ROOT/.julia-depot"

{
    echo "RUNNER solver=cuclarabel gpu=$GPU tolerance=$TOL cache=$CACHE"
    command=(
      "$JULIA" --project="$ROOT/benchmark/cuclarabel_env"
        "$ROOT/benchmark/ill_conditioned_lasso_cuclarabel.jl" \
        --cache "$CACHE" \
        --output-dir "$OUTPUT" \
        --tol "$TOL" \
        --time-limit "$TIME_LIMIT" \
        --verbose-level "$VERBOSE_LEVEL" \
        --direct-solve-method "$DIRECT_SOLVE_METHOD"
    )
    [[ -z "$WARMUP_CACHE" ]] || command+=(--warmup-cache "$WARMUP_CACHE")
    "${command[@]}"
} 2>&1 | tee "$RAW_LOG"
