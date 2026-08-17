#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIBLING_ROOT="$(cd "$REPO_ROOT/../PDCS_fork" && pwd)"
CONFIG="$SCRIPT_DIR/robustness_experiment.toml"
JULIA="$SIBLING_ROOT/.julia-bin/julia-1.12.6/bin/julia"
DEPOT="$SIBLING_ROOT/.julia-depot"

GPUS="0"
OUTPUT="$SCRIPT_DIR/results/robustness_alpha1p5_1e-6"
TIME_LIMIT="3600"
SOLVER_TOLERANCE="2e-7"
VERIFICATION_TOLERANCE="1e-6"
FORCE="false"

usage() {
  printf '%s\n' \
    "Usage: run_robustness.sh [options]" \
    "  --gpus comma,separated,ids  (default: 0)" \
    "  --output DIR" \
    "  --time-limit SECONDS       (default: 3600)" \
    "  --solver-tolerance VALUE   (default: 2e-7)" \
    "  --verification-tolerance VALUE (default: 1e-6)" \
    "  --force true|false"
}

while (($#)); do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --time-limit) TIME_LIMIT="$2"; shift 2 ;;
    --solver-tolerance) SOLVER_TOLERANCE="$2"; shift 2 ;;
    --verification-tolerance) VERIFICATION_TOLERANCE="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "$JULIA" ]] || {
  printf 'Julia not executable: %s\n' "$JULIA" >&2
  exit 2
}
OUTPUT="$(realpath -m "$OUTPUT")"
mkdir -p "$OUTPUT/results" "$OUTPUT/logs" "$OUTPUT/environment"

export JULIA_DEPOT_PATH="$DEPOT"
export JULIA_PKG_OFFLINE=true
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3
export PDCS_SKIP_GPU_PRECOMPILE=1
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64

ARTIFACT_DIR="$OUTPUT/environment/production_sm90"
mkdir -p "$ARTIFACT_DIR"
required_artifacts=(libfew_block_proj.so massive_block_proj.ptx \
  moderate_block_proj.ptx sufficient_block_proj.ptx utils.ptx)
rebuild=false
for artifact in "${required_artifacts[@]}"; do
  [[ -s "$ARTIFACT_DIR/$artifact" ]] || rebuild=true
done
if [[ "$rebuild" == true ]]; then
  make -C "$REPO_ROOT/src/pdcs_gpu/cuda" rebuild-gpu \
    CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90 OUTPUT_DIR="$ARTIFACT_DIR" \
    >"$OUTPUT/logs/build_production_artifacts.log" 2>&1
fi
export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$ARTIFACT_DIR"
sha256sum "$ARTIFACT_DIR"/* >"$OUTPUT/environment/artifact_hashes.sha256"

{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  uname -a
  "$JULIA" --version
  nvidia-smi
  git -C "$REPO_ROOT" rev-parse HEAD
  git -C "$SIBLING_ROOT" rev-parse HEAD
  sha256sum "$CONFIG" "$SCRIPT_DIR/AE11Common.jl" \
    "$SCRIPT_DIR/run_cupdcs.jl" "$0"
} >"$OUTPUT/environment/environment.txt" 2>&1

IFS=',' read -r -a gpu_array <<<"$GPUS"
seeds=(2026 2027 2028 2029 2030)
kappas=(1 1e2 1e4 1e6 1e8)

run_shard() {
  local slot="$1"
  local gpu="${gpu_array[$slot]}"
  local counter=0
  local seed kappa log_name
  for seed in "${seeds[@]}"; do
    for kappa in "${kappas[@]}"; do
      if ((counter % ${#gpu_array[@]} == slot)); then
        log_name="seed${seed}_K${kappa}.log"
        CUDA_VISIBLE_DEVICES="$gpu" "$JULIA" --startup-file=no \
          --project="$REPO_ROOT" "$SCRIPT_DIR/run_cupdcs.jl" \
          --config "$CONFIG" --profile medium --seed "$seed" \
          --kappa "$kappa" --panels B --rescaling on \
          --tolerance "$VERIFICATION_TOLERANCE" \
          --solver-tolerance "$SOLVER_TOLERANCE" \
          --time-limit "$TIME_LIMIT" --warmup true --force "$FORCE" \
          --output-dir "$OUTPUT/results" \
          >"$OUTPUT/logs/$log_name" 2>&1
      fi
      counter=$((counter + 1))
    done
  done
}

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
  "$SCRIPT_DIR/analyze_robustness.jl" "$CONFIG" "$OUTPUT/results" \
  "$OUTPUT/analysis"
