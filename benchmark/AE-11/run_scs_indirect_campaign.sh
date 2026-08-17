#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIBLING_ROOT="$(cd "$REPO_ROOT/../PDCS_fork" && pwd)"
JULIA="$SIBLING_ROOT/.julia-bin/julia-1.12.6/bin/julia"
PHASE="pilot"
PANELS="A,B"
TOLERANCES="1e-3"
SEEDS=""
KAPPAS="1,1e2,1e4,1e6,1e8"
OUTPUT="$SCRIPT_DIR/results/formal"
TIME_LIMIT=""
WARMUP="true"
FORCE="false"

while (($#)); do
    case "$1" in
      --phase) PHASE="$2"; shift 2 ;;
      --panels) PANELS="$2"; shift 2 ;;
      --tolerances) TOLERANCES="$2"; shift 2 ;;
      --seeds) SEEDS="$2"; shift 2 ;;
      --kappas) KAPPAS="$2"; shift 2 ;;
      --output) OUTPUT="$2"; shift 2 ;;
      --time-limit) TIME_LIMIT="$2"; shift 2 ;;
      --warmup) WARMUP="$2"; shift 2 ;;
      --force) FORCE="$2"; shift 2 ;;
      *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case "$PHASE" in
  pilot) [[ -n "$SEEDS" ]] || SEEDS="2026,2027" ;;
  medium) [[ -n "$SEEDS" ]] || SEEDS="2026,2027,2028,2029,2030" ;;
  *) printf 'Phase must be pilot or medium\n' >&2; exit 2 ;;
esac

OUTPUT="$(realpath -m "$OUTPUT")"
mkdir -p "$OUTPUT/logs" "$OUTPUT/results"
export JULIA_DEPOT_PATH="$SIBLING_ROOT/.julia-depot"
export JULIA_PKG_OFFLINE=true
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3
export PDCS_SKIP_GPU_PRECOMPILE=1
export AE11_BLAS_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=32

IFS=',' read -r -a seed_array <<<"$SEEDS"
IFS=',' read -r -a kappa_array <<<"$KAPPAS"
IFS=',' read -r -a tolerance_array <<<"$TOLERANCES"
extra_time=()
[[ -z "$TIME_LIMIT" ]] || extra_time=(--time-limit "$TIME_LIMIT")
log="$OUTPUT/logs/scs_indirect_pinned_omp32.log"
: >"$log"

for seed in "${seed_array[@]}"; do
  for kappa in "${kappa_array[@]}"; do
    for tolerance in "${tolerance_array[@]}"; do
      taskset -c 49-80 "$JULIA" --startup-file=no --threads=32 \
        --project="$REPO_ROOT" "$SCRIPT_DIR/run_moi_solver.jl" \
        --solver scs_indirect --profile "$PHASE" --seed "$seed" \
        --kappa "$kappa" --tolerance "$tolerance" --panels "$PANELS" \
        --output-dir "$OUTPUT/results" --warmup "$WARMUP" --force "$FORCE" \
        "${extra_time[@]}" >>"$log" 2>&1
    done
  done
done

python3 "$SCRIPT_DIR/analyze_results.py" \
  --config "$SCRIPT_DIR/experiment.toml" --results-dir "$OUTPUT/results" \
  --output-dir "$OUTPUT/analysis" --allow-incomplete
