#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_DIR=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
COUNT=1048576
DIMENSION=10
SEEDS="2026,2027,2028"
DELTAS="0,0.0001,0.001,0.01"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --cone-count) COUNT="$2"; shift 2 ;;
    --cone-dimension) DIMENSION="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --deltas) DELTAS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      printf '%s\n' 'Usage: profile_soc_parametric_ncu.sh --run-dir PATH [--gpu N --seeds LIST --deltas LIST --dry-run]'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" ]] || { printf '%s\n' '--run-dir is required' >&2; exit 2; }
NCU="$CUDA_ROOT/bin/ncu"
[[ -x "$NCU" ]] || { printf 'ncu not found: %s\n' "$NCU" >&2; exit 1; }
IFS=, read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=, read -r -a DELTA_ARRAY <<< "$DELTAS"
((DRY_RUN)) || { mkdir -p "$RUN_DIR/ncu/parametric"; "$NCU" --query-metrics > "$RUN_DIR/ncu/parametric/available_metrics.txt"; }

for delta in "${DELTA_ARRAY[@]}"; do
  label="${delta//./p}"
  for seed in "${SEED_ARRAY[@]}"; do
    cache="$RUN_DIR/parametric/delta_${label}/seed${seed}/case_cache.jls"
    [[ -f "$cache" || $DRY_RUN -eq 1 ]] || { printf 'Missing %s\n' "$cache" >&2; exit 1; }
    for strategy in threadWise warpWise; do
      kernel=massive_block_proj; [[ "$strategy" == warpWise ]] && kernel=sufficient_block_proj
      stem="$RUN_DIR/ncu/parametric/delta_${label}_${strategy}_seed${seed}"
      cmd=(env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH"
        CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
        JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1
        "$NCU" --target-processes all --kernel-name "regex:^${kernel}$"
        --nvtx --nvtx-include "PDCS_PROJECTION/" --launch-count 1
        --section SpeedOfLight --section Occupancy --section SchedulerStats
        --section WarpStateStats --section SourceCounters
        --force-overwrite=false --export "$stem"
        "$JULIA_BIN" "--project=$REPO_ROOT"
        "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
        --experiment parametric --mode profile-one --delta "$delta"
        --layout grouped --strategy "$strategy" --seed "$seed"
        --cone-count "$COUNT" --cone-dimension "$DIMENSION"
        --case-cache "$cache" --output-dir "$(dirname "$cache")")
      printf 'COMMAND:'; printf ' %q' "${cmd[@]}"; printf '\n'
      ((DRY_RUN)) && continue
      [[ ! -e "$stem.ncu-rep" ]] || { printf 'Refusing to overwrite %s\n' "$stem.ncu-rep" >&2; exit 1; }
      set +e; "${cmd[@]}" > "$stem.application.log" 2> "$stem.ncu.log"; status=$?; set -e
      printf '%s\n' "$status" > "$stem.exit_status.txt"
      grep -Eqi 'ERR_NVGPUCTRPERM|permission' "$stem.ncu.log" &&
        cp "$stem.ncu.log" "$stem.PROFILE_INCOMPLETE.txt"
      ((status==0)) || continue
      "$NCU" --import "$stem.ncu-rep" --page raw --csv > "$stem.csv" 2> "$stem.import.log"
    done
  done
done
