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
SEED=2026
DURATION=30
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
    --seed) SEED="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --deltas) DELTAS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      printf '%s\n' 'Usage: run_soc_parametric_utilization.sh --run-dir PATH [--gpu N --seed N --duration 30 --deltas LIST --dry-run]'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" ]] || { printf '%s\n' '--run-dir is required' >&2; exit 2; }
IFS=, read -r -a DELTA_ARRAY <<< "$DELTAS"
DMON_PID=
cleanup() { [[ -z "${DMON_PID:-}" ]] || { kill "$DMON_PID" 2>/dev/null || true; wait "$DMON_PID" 2>/dev/null || true; DMON_PID=; }; }
trap cleanup EXIT INT TERM

for delta in "${DELTA_ARRAY[@]}"; do
  label="${delta//./p}"
  cache="$RUN_DIR/parametric/delta_${label}/seed${SEED}/case_cache.jls"
  [[ -f "$cache" || $DRY_RUN -eq 1 ]] || { printf 'Missing %s\n' "$cache" >&2; exit 1; }
  for strategy in threadWise warpWise; do
    stem="$RUN_DIR/nvidia_smi/parametric_delta_${label}_${strategy}_seed${SEED}"
    cmd=(env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH"
      CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
      JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1
      "$JULIA_BIN" "--project=$REPO_ROOT"
      "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
      --experiment parametric --mode duration --duration "$DURATION"
      --delta "$delta" --layout grouped --strategy "$strategy" --seed "$SEED"
      --cone-count "$COUNT" --cone-dimension "$DIMENSION"
      --case-cache "$cache" --output-dir "$(dirname "$cache")")
    printf 'COMMAND:'; printf ' %q' "${cmd[@]}"; printf '\n'
    ((DRY_RUN)) && continue
    mkdir -p "$RUN_DIR/nvidia_smi"
    [[ ! -e "$stem.dmon.txt" ]] || { printf 'Refusing to overwrite %s\n' "$stem.dmon.txt" >&2; exit 1; }
    nvidia-smi dmon -i "$GPU" -s pucvmet -d 1 > "$stem.dmon.txt" 2>&1 &
    DMON_PID=$!
    set +e; "${cmd[@]}" > "$stem.application.log" 2> "$stem.application.err"; status=$?; set -e
    cleanup
    printf '%s\n' "$status" > "$stem.exit_status.txt"
    ((status==0)) || exit "$status"
  done
done
trap - EXIT INT TERM
