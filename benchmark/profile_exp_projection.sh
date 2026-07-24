#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"
CONE_COUNT=1048576
CASES="similar,heterogeneous,mixed_grouped,mixed_interleaved"
STRATEGIES="threadWise,warpWise"
OUTPUT_DIR="$REPO_ROOT/benchmark/results/exp_projection/profile"
TRIALS=20

while (($#)); do
  case "$1" in
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --gpu) GPU_INDEX="$2"; shift 2 ;;
    --cone-count) CONE_COUNT="$2"; shift 2 ;;
    --cases) CASES="$2"; shift 2 ;;
    --strategies) STRATEGIES="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --help|-h)
      printf '%s\n' 'Usage: profile_exp_projection.sh [--gpu N] [--cuda-home PATH] [--cone-count N] [--output-dir PATH]'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

NCU="$CUDA_ROOT/bin/ncu"
[[ -x "$NCU" ]] || NCU="$(command -v ncu || true)"
[[ -x "$NCU" ]] || { printf 'Nsight Compute (ncu) was not found.\n' >&2; exit 1; }
[[ -x "$JULIA_BIN" ]] || { printf 'Julia is not executable: %s\n' "$JULIA_BIN" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR/timing" "$OUTPUT_DIR/ncu"

IFS=',' read -r -a CASE_ARRAY <<< "$CASES"
IFS=',' read -r -a STRATEGY_ARRAY <<< "$STRATEGIES"

for case_name in "${CASE_ARRAY[@]}"; do
  for strategy in "${STRATEGY_ARRAY[@]}"; do
    case "$strategy" in
      threadWise) native_kernel=massive_block_proj ;;
      warpWise) native_kernel=sufficient_block_proj ;;
      blockWise) native_kernel=moderate_block_proj ;;
      *) printf 'Unknown strategy: %s\n' "$strategy" >&2; exit 2 ;;
    esac
    stem="${case_name}_${strategy}"

    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PATH="$CUDA_ROOT/bin:$PATH" \
      CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
      JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1 \
      "$JULIA_BIN" --project="$REPO_ROOT" "$REPO_ROOT/benchmark/exp_warp_profile.jl" \
        --case "$case_name" --strategy "$strategy" --cone-count "$CONE_COUNT" \
        --trials "$TRIALS" --output "$OUTPUT_DIR/timing/${stem}.csv"

    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PATH="$CUDA_ROOT/bin:$PATH" \
      CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
      JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1 \
      "$NCU" --target-processes all \
        --kernel-name "regex:${native_kernel}" --launch-skip 1 --launch-count 1 \
        --section SpeedOfLight --section Occupancy --section SchedulerStats \
        --section WarpStateStats --section SourceCounters --force-overwrite \
        --export "$OUTPUT_DIR/ncu/$stem" \
        "$JULIA_BIN" --project="$REPO_ROOT" "$REPO_ROOT/benchmark/exp_warp_profile.jl" \
          --profile-one --case "$case_name" --strategy "$strategy" \
          --cone-count "$CONE_COUNT"

    "$NCU" --import "$OUTPUT_DIR/ncu/${stem}.ncu-rep" --page raw --csv \
      > "$OUTPUT_DIR/ncu/${stem}.csv"
  done
done

printf 'Exponential-cone timing and Nsight reports: %s\n' "$OUTPUT_DIR"
