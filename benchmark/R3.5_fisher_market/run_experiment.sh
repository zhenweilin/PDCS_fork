#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIBLING_ROOT="$(cd "$REPO_ROOT/../PDCS_fork" && pwd)"
CONFIG="$SCRIPT_DIR/experiment.toml"
JULIA="$SIBLING_ROOT/.julia-bin/julia"
DEPOT="$SIBLING_ROOT/.julia-depot"

GPUS="0"
OUTPUT="$SCRIPT_DIR/results/formal_diagonal_scalar"
TIME_LIMIT="600"
FORCE="false"

usage() {
  printf '%s\n' \
    "Usage: run_experiment.sh [options]" \
    "  --gpus comma,separated,ids  (default: 0)" \
    "  --output DIR" \
    "  --time-limit SECONDS       (default: 600)" \
    "  --force true|false"
}

while (($#)); do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --time-limit) TIME_LIMIT="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

OUTPUT="$(realpath -m "$OUTPUT")"
mkdir -p "$OUTPUT/results" "$OUTPUT/logs" "$OUTPUT/environment"
export JULIA_DEPOT_PATH="$DEPOT"
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3
export PDCS_SKIP_GPU_PRECOMPILE=1
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$REPO_ROOT/benchmark/AE-11/results/robustness_alpha1p5_1e-6/environment/production_sm90"

IFS=',' read -r -a gpu_array <<<"$GPUS"
((${#gpu_array[@]} > 0)) || { printf 'No GPU selected\n' >&2; exit 2; }
seeds=(2026 2027 2028 2029 2030)

run_one() {
  local gpu="$1" seed="$2" mode="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$JULIA" --startup-file=no \
    --project="$REPO_ROOT" "$SCRIPT_DIR/run_case.jl" \
    --config "$CONFIG" --m 100 --n 5000 --density 1.0 --seed "$seed" \
    --formulation direct --rescaling "$mode" --tolerance 1e-6 \
    --time-limit "$TIME_LIMIT" --print-frequency 1000 --verbose 0 \
    --force "$FORCE" --result "$OUTPUT/results/seed${seed}_${mode}.toml" \
    >"$OUTPUT/logs/seed${seed}_${mode}.log" 2>&1
  local status=$?
  [[ "$status" == 0 || "$status" == 3 ]]
}

run_shard() {
  local slot="$1" gpu="${gpu_array[$1]}" index seed first second
  for index in "${!seeds[@]}"; do
    ((index % ${#gpu_array[@]} == slot)) || continue
    seed="${seeds[$index]}"
    if ((index % 2 == 0)); then
      first=diagonal; second=scalar_cone
    else
      first=scalar_cone; second=diagonal
    fi
    run_one "$gpu" "$seed" "$first" || return 1
    run_one "$gpu" "$seed" "$second" || return 1
  done
}

{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  "$JULIA" --version
  nvidia-smi
  git -C "$REPO_ROOT" rev-parse HEAD
  git -C "$SIBLING_ROOT" rev-parse HEAD
  sha256sum "$CONFIG" "$SCRIPT_DIR/FisherDirectFormulation.jl" \
    "$SCRIPT_DIR/run_case.jl" "$SCRIPT_DIR/analyze_results.jl" "$0"
} >"$OUTPUT/environment/environment.txt" 2>&1

pids=()
for slot in "${!gpu_array[@]}"; do
  run_shard "$slot" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
((failed == 0)) || exit 1

"$JULIA" --startup-file=no --project="$REPO_ROOT" \
  "$SCRIPT_DIR/analyze_results.jl" "$CONFIG" "$OUTPUT/results" \
  "$OUTPUT/analysis"
