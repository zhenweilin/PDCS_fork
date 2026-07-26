#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PHASE=all
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_warp_divergence_2"
OUTPUT_ROOT="$REPO_ROOT/benchmark/results/rebuttal/additional_experiments_2"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
ARCH="${PDCS_GPU_ARCH:-sm_90}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
SEEDS="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035"
SMOKE=0
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: run_warp_divergence_2.sh [options]' \
    '  --phase snapshot|build|baseline|utilization|parametric|soc-profile|branch-profile|parametric-profile|dimensions|exp|exp-profile|applications|analysis|all' \
    '  --run-id ID --output-root PATH --gpu N --cuda-home PATH --arch sm_XX' \
    '  --julia PATH --julia-depot PATH --smoke --dry-run'
}
while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in snapshot|build|baseline|utilization|parametric|soc-profile|branch-profile|parametric-profile|dimensions|exp|exp-profile|applications|analysis|all) ;; *) usage >&2; exit 2 ;; esac
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
if ((SMOKE)); then
  SEEDS=2026
  COUNT=1024
  DIM_COUNT=1024
  DURATION=5
  EXP_COUNT=1024
else
  COUNT=1048576
  DIM_COUNT=262144
  DURATION=35
  EXP_COUNT=1048576
fi

run() {
  printf 'COMMAND:'; printf ' %q' "$@"; printf '\n'
  ((DRY_RUN)) || "$@"
}
has_phase() {
  [[ "$PHASE" == "$1" ]] && return 0
  [[ "$PHASE" == all ]] || return 1
  case "$1" in
    snapshot|build|baseline|utilization|parametric|dimensions|exp|analysis) return 0 ;;
    *) return 1 ;;
  esac
}
IFS=, read -r -a SEED_ARRAY <<< "$SEEDS"
common=(--gpu "$GPU" --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT")

if ((!DRY_RUN)); then
  mkdir -p "$RUN_DIR"
fi

if has_phase snapshot; then
  run "$REPO_ROOT/benchmark/rebuttal/snapshot_source.sh" \
    --output "$RUN_DIR/source_snapshot"
fi

if has_phase build; then
  run make -C "$REPO_ROOT/src/pdcs_gpu/cuda" rebuild-gpu rebuild-profile \
    CUDA_HOME="$CUDA_ROOT" ARCH="$ARCH"
fi

if has_phase baseline; then
  args=("$REPO_ROOT/benchmark/run_soc_divergence_2x2.sh" "${common[@]}"
    --arch "$ARCH" --cone-count "$COUNT" --cone-dimension 10
    --seeds "$SEEDS" --experiments iteration,branch
    --output-root "$RUN_DIR/baseline" --run-id paired --keep-caches
    --no-build)
  ((SMOKE)) && args+=(--smoke)
  run "${args[@]}"
fi

if has_phase utilization; then
  for seed in "${SEED_ARRAY[@]}"; do
    for experiment in iteration branch; do
      cache="$RUN_DIR/baseline/paired/$experiment/seed$seed/case_cache.jls"
      for layout in grouped interleaved; do
        for strategy in threadWise warpWise; do
          run "$REPO_ROOT/benchmark/rebuttal/run_active_window_utilization.sh" \
            --run-dir "$RUN_DIR/utilization/raw" --case-cache "$cache" \
            --experiment "$experiment" --layout "$layout" --strategy "$strategy" \
            --seed "$seed" --duration "$DURATION" --cone-count "$COUNT" \
            --cone-dimension 10 "${common[@]}"
        done
      done
    done
  done
fi

if has_phase parametric; then
  args=("$REPO_ROOT/benchmark/run_soc_parametric_similarity.sh"
    --run-dir "$RUN_DIR/soc_parametric" "${common[@]}"
    --cone-count "$COUNT" --cone-dimension 10 --seeds "$SEEDS"
    --deltas 0,0.0001,0.001,0.01 --keep-caches)
  ((SMOKE)) && args+=(--smoke)
  run "${args[@]}"
fi

if has_phase branch-profile; then
  run "$REPO_ROOT/benchmark/profile_soc_divergence_2x2_ncu.sh" \
    --run-dir "$RUN_DIR/baseline/paired" "${common[@]}" \
    --cone-count "$COUNT" --cone-dimension 10 --seeds "$SEEDS" \
    --experiments branch
fi

if has_phase soc-profile; then
  run "$REPO_ROOT/benchmark/profile_soc_divergence_2x2_ncu.sh" \
    --run-dir "$RUN_DIR/baseline/paired" "${common[@]}" \
    --cone-count "$COUNT" --cone-dimension 10 --seeds "$SEEDS" \
    --experiments iteration,branch
fi

if has_phase parametric-profile; then
  run "$REPO_ROOT/benchmark/profile_soc_parametric_ncu.sh" \
    --run-dir "$RUN_DIR/soc_parametric" "${common[@]}" \
    --cone-count "$COUNT" --cone-dimension 10 --seeds "$SEEDS" \
    --deltas 0,0.0001,0.001,0.01
fi

if has_phase dimensions; then
  for dimension in 10 32 100; do
    args=("$REPO_ROOT/benchmark/run_soc_divergence_2x2.sh" "${common[@]}"
      --arch "$ARCH" --cone-count "$DIM_COUNT" --cone-dimension "$dimension"
      --seeds "$SEEDS" --experiments iteration,branch
      --output-root "$RUN_DIR/soc_dimensions/dimension_$dimension"
      --run-id paired --no-build)
    ((SMOKE)) && args+=(--smoke)
    run "${args[@]}"
  done
fi

if has_phase exp; then
  for seed in "${SEED_ARRAY[@]}"; do
    for sigma in 0.1 0.5 1 2 5 10; do
      run env CUDA_VISIBLE_DEVICES="$GPU" CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
        JULIA_CONDAPKG_BACKEND=Null JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1 \
        "$JULIA_BIN" -O1 "--project=$REPO_ROOT" "$REPO_ROOT/benchmark/exp_cone_projection.jl" \
        --cone-counts "$EXP_COUNT" --trials 10 --seed "$seed" \
        --strategies threadWise,warpWise,blockWise --variants primalDiagonal \
        --input-distribution heterogeneous --sigma "$sigma" --diagonal-sigma 1 \
        --output "$RUN_DIR/exp_divergence/input_sigma_${sigma}_seed${seed}.csv"
      run env CUDA_VISIBLE_DEVICES="$GPU" CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
        JULIA_CONDAPKG_BACKEND=Null JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1 \
        "$JULIA_BIN" -O1 "--project=$REPO_ROOT" "$REPO_ROOT/benchmark/exp_cone_projection.jl" \
        --cone-counts "$EXP_COUNT" --trials 10 --seed "$seed" \
        --strategies threadWise,warpWise,blockWise --variants primalDiagonal \
        --input-distribution heterogeneous --sigma 1 --diagonal-sigma "$sigma" \
        --output "$RUN_DIR/exp_divergence/diagonal_sigma_${sigma}_seed${seed}.csv"
    done
  done
fi

if has_phase exp-profile; then
  for seed in "${SEED_ARRAY[@]}"; do
    for case_name in similar heterogeneous mixed_grouped mixed_interleaved; do
      for strategy in threadWise warpWise; do
        run "$REPO_ROOT/benchmark/rebuttal/profile_ncu.sh" \
          --kind exp --case "$case_name" --strategy "$strategy" --seed "$seed" \
          --gpu "$GPU" --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" \
          --julia-depot "$JULIA_DEPOT" --cone-count "$EXP_COUNT" \
          --output-root "$RUN_DIR/ncu/exp" \
          --run-id "${case_name}_${strategy}_seed${seed}"
      done
    done
  done
fi

if has_phase applications; then
  if grep -q 'Replace with the manuscript' "$REPO_ROOT/benchmark/rebuttal/application_manifest.csv"; then
    message='Application manifest still contains placeholders; solver commands were not fabricated.'
    if [[ "$PHASE" == applications ]]; then
      printf '%s\n' "$message" >&2
      exit 2
    fi
    printf '%s\n' "$message" >&2
    ((DRY_RUN)) || printf '%s\n' "$message" > "$RUN_DIR/applications_EXTERNAL_INPUT_REQUIRED.txt"
  else
    run env PDCS_APPLICATION_SEEDS="$SEEDS" \
      "$JULIA_BIN" "--project=$REPO_ROOT" "$REPO_ROOT/benchmark/rebuttal/application_trace.jl" \
      --manifest "$REPO_ROOT/benchmark/rebuttal/application_manifest.csv" \
      --output "$RUN_DIR/applications"
  fi
fi

if has_phase analysis; then
  run "$JULIA_BIN" "--project=$REPO_ROOT" \
    "$REPO_ROOT/benchmark/summarize_soc_divergence_utilization.jl" \
    --root "$RUN_DIR/utilization/raw" \
    --ledger-root "$RUN_DIR/baseline/paired" \
    --output "$RUN_DIR/utilization/utilization_summary.csv"
  run "$JULIA_BIN" "--project=$REPO_ROOT" "$REPO_ROOT/benchmark/analyze_soc_divergence_2x2.jl" \
    --root "$RUN_DIR/baseline/paired" --seeds "$SEEDS" --bootstrap 10000 \
    --output "$RUN_DIR"
fi

printf 'PHASE_COMPLETE phase=%s run_dir=%s\n' "$PHASE" "$RUN_DIR"
