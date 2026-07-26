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
EXPERIMENTS="iteration,branch"
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
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      printf '%s\n' 'Usage: run_soc_divergence_2x2_utilization.sh --run-dir PATH [--gpu N --seed N --duration 30 --dry-run]'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" ]] || { printf '%s\n' '--run-dir is required' >&2; exit 2; }
IFS=, read -r -a EXP_ARRAY <<< "$EXPERIMENTS"

DMON_PID=
cleanup() {
  if [[ -n "${DMON_PID:-}" ]]; then
    kill "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
    DMON_PID=
  fi
}
trap cleanup EXIT INT TERM

for experiment in "${EXP_ARRAY[@]}"; do
  cache="$RUN_DIR/$experiment/seed$SEED/case_cache.jls"
  [[ -f "$cache" || $DRY_RUN -eq 1 ]] ||
    { printf 'Missing case cache: %s\n' "$cache" >&2; exit 1; }
  for layout in grouped interleaved; do
    for strategy in threadWise warpWise; do
      stem="${experiment}_${layout}_${strategy}_seed${SEED}"
      output="$RUN_DIR/nvidia_smi/$stem"
      cmd=(env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH"
        CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
        JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1
        "$JULIA_BIN" "--project=$REPO_ROOT"
        "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
        --experiment "$experiment" --mode duration --duration "$DURATION"
        --layout "$layout" --strategy "$strategy" --seed "$SEED"
        --cone-count "$COUNT" --cone-dimension "$DIMENSION"
        --case-cache "$cache" --output-dir "$RUN_DIR/$experiment/seed$SEED")
      printf 'COMMAND:'; printf ' %q' "${cmd[@]}"; printf '\n'
      ((DRY_RUN)) && continue
      mkdir -p "$RUN_DIR/nvidia_smi"
      [[ ! -e "$output.dmon.txt" ]] ||
        { printf 'Refusing to overwrite %s.dmon.txt\n' "$output" >&2; exit 1; }
      nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
        --format=csv > "$output.processes_before.txt" 2>&1 || true
      nvidia-smi dmon -i "$GPU" -s pucvmet -d 1 > "$output.dmon.txt" 2>&1 &
      DMON_PID=$!
      set +e
      "${cmd[@]}" > "$output.application.log" 2> "$output.application.err"
      status=$?
      set -e
      cleanup
      printf '%s\n' "$status" > "$output.exit_status.txt"
      nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
        --format=csv > "$output.processes_after.txt" 2>&1 || true
      ((status==0)) || exit "$status"
    done
  done
done
trap - EXIT INT TERM
