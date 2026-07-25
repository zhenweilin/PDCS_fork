#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="$SCRIPT_DIR/profile_nsys.sh"

KINDS="soc,exp"
SOC_CASES="similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved"
EXP_CASES="similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved"
STRATEGIES="threadWise,warpWise,blockWise"
SEEDS="2026,2027,2028"
WARM_MODES="cold,warm"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
DURATION=30
CONE_COUNT=1048576
SOC_DIMENSION=10
HETERO_SIGMA=2.0
INPUT_SIGMA=1.0
DIAGONAL_SIGMA=1.0
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
JULIA_BIN=""
JULIA_DEPOT=""
OUTPUT_ROOT=""
GPU_METRICS="required"
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: run_nsys_matrix.sh [options]' \
    '  --kinds soc,exp' \
    '  --soc-cases LIST' \
    '  --exp-cases LIST' \
    '  --strategies LIST' \
    '  --seeds LIST' \
    '  --warm-modes cold,warm       SOC only' \
    '  --gpu N --duration S --cone-count N --soc-dimension N' \
    '  --hetero-sigma X --sigma X --diagonal-sigma X' \
    '  --cuda-home PATH --julia PATH --julia-depot PATH' \
    '  --output-root PATH --gpu-metrics auto|required|off --dry-run'
}

while (($#)); do
  case "$1" in
    --kinds) KINDS="$2"; shift 2 ;;
    --soc-cases) SOC_CASES="$2"; shift 2 ;;
    --exp-cases) EXP_CASES="$2"; shift 2 ;;
    --strategies) STRATEGIES="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --warm-modes) WARM_MODES="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --cone-count) CONE_COUNT="$2"; shift 2 ;;
    --soc-dimension) SOC_DIMENSION="$2"; shift 2 ;;
    --hetero-sigma) HETERO_SIGMA="$2"; shift 2 ;;
    --sigma) INPUT_SIGMA="$2"; shift 2 ;;
    --diagonal-sigma) DIAGONAL_SIGMA="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu-metrics) GPU_METRICS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/../results/rebuttal/nsys_matrix_$STAMP}"
IFS=',' read -r -a KIND_ARRAY <<< "$KINDS"
IFS=',' read -r -a STRATEGY_ARRAY <<< "$STRATEGIES"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"

COMMON=(--gpu "$GPU" --duration "$DURATION" --cone-count "$CONE_COUNT"
  --cuda-home "$CUDA_ROOT" --output-root "$OUTPUT_ROOT" --gpu-metrics "$GPU_METRICS")
[[ -n "$JULIA_BIN" ]] && COMMON+=(--julia "$JULIA_BIN")
[[ -n "$JULIA_DEPOT" ]] && COMMON+=(--julia-depot "$JULIA_DEPOT")
((DRY_RUN)) && COMMON+=(--dry-run)

for kind in "${KIND_ARRAY[@]}"; do
  if [[ "$kind" == soc ]]; then
    IFS=',' read -r -a CASE_ARRAY <<< "$SOC_CASES"
    IFS=',' read -r -a WARM_ARRAY <<< "$WARM_MODES"
    for case_name in "${CASE_ARRAY[@]}"; do
      for strategy in "${STRATEGY_ARRAY[@]}"; do
        for seed in "${SEED_ARRAY[@]}"; do
          for warm in "${WARM_ARRAY[@]}"; do
            ARGS=(--kind soc --case "$case_name" --strategy "$strategy" --seed "$seed"
              --cone-dimension "$SOC_DIMENSION" --hetero-sigma "$HETERO_SIGMA"
              --run-id "soc_${case_name}_${strategy}_${warm}_seed${seed}")
            [[ "$warm" == warm ]] && ARGS+=(--warm-start)
            "$PROFILE" "${COMMON[@]}" "${ARGS[@]}"
          done
        done
      done
    done
  elif [[ "$kind" == exp ]]; then
    IFS=',' read -r -a CASE_ARRAY <<< "$EXP_CASES"
    for case_name in "${CASE_ARRAY[@]}"; do
      for strategy in "${STRATEGY_ARRAY[@]}"; do
        for seed in "${SEED_ARRAY[@]}"; do
          "$PROFILE" "${COMMON[@]}" --kind exp --case "$case_name" \
            --strategy "$strategy" --seed "$seed" --sigma "$INPUT_SIGMA" \
            --diagonal-sigma "$DIAGONAL_SIGMA" \
            --run-id "exp_${case_name}_${strategy}_seed${seed}"
        done
      done
    done
  else
    printf 'Unknown kind: %s\n' "$kind" >&2
    exit 2
  fi
done

printf 'Nsight Systems matrix complete: %s\n' "$OUTPUT_ROOT"
