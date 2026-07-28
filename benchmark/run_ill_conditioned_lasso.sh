#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PHASE=all
PROFILE=pilot
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_ill_conditioned_lasso"
OUTPUT_ROOT="$REPO_ROOT/benchmark/results/rebuttal/ill_conditioned_lasso"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
SOLVERS="pdcs_cpu,pdcs_gpu,scs,cuclarabel,mosek"
SEEDS=""
PANEL=fixed_lambda
TOLS="1e-3,1e-6"
MOSEK_LICENSE="$REPO_ROOT/mosek_lic/mosek.lic"
DRY_RUN=0
GPU=auto
SOLVERS_SET=0
SEEDS_SET=0

usage() {
  printf '%s\n' \
    'Usage: run_ill_conditioned_lasso.sh [options]' \
    '  --phase snapshot|test|generate|solve|analysis|all' \
    '  --profile smoke|reference|pilot|medium --run-id ID --output-root PATH' \
    '  --julia PATH --julia-depot PATH --solvers LIST --seeds LIST --gpu auto|INDEX' \
    '  --tolerances LIST --mosek-license PATH --panel fixed_lambda|fixed_residual --dry-run' \
    'MOSEK is run only for smoke/reference correctness checks; pilot/medium exclude it by default.'
}
while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --solvers) SOLVERS="$2"; SOLVERS_SET=1; shift 2 ;;
    --seeds) SEEDS="$2"; SEEDS_SET=1; shift 2 ;;
    --tolerances) TOLS="$2"; shift 2 ;;
    --mosek-license) MOSEK_LICENSE="$2"; shift 2 ;;
    --panel) PANEL="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in snapshot|test|generate|solve|analysis|all) ;; *) usage >&2; exit 2 ;; esac
case "$PROFILE" in
  smoke)
    M=1000; N=5000; S=40; D=10; LIMIT=60
    (( SEEDS_SET )) || SEEDS=2026
    ;;
  reference)
    # Medium-sized independent reference: retained for a feasible MOSEK
    # validation, unlike the production-scale pilot and medium profiles.
    M=5000; N=25000; S=100; D=20; LIMIT=1800
    (( SEEDS_SET )) || SEEDS=2026
    ;;
  pilot)
    M=20000; N=100000; S=200; D=20; LIMIT=7200
    (( SEEDS_SET )) || SEEDS=2026,2027
    # The pilot is already too large for a routine reference solve.  Use
    # PDCS/SCS/CuClarabel only unless a user deliberately overrides --solvers.
    (( SOLVERS_SET )) || SOLVERS=pdcs_cpu,pdcs_gpu,scs,cuclarabel
    ;;
  medium)
    M=50000; N=500000; S=500; D=20; LIMIT=7200
    (( SEEDS_SET )) || SEEDS=2026,2027,2028,2029,2030,2031,2032,2033,2034,2035
    # MOSEK is deliberately omitted for the large benchmark.  A user can
    # override this only explicitly, e.g. --solvers pdcs_cpu,mosek.
    (( SOLVERS_SET )) || SOLVERS=pdcs_cpu,pdcs_gpu,scs,cuclarabel
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
IFS=, read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=, read -r -a SOLVER_ARRAY <<< "$SOLVERS"
IFS=, read -r -a TOL_ARRAY <<< "$TOLS"
KVALUES="1,100,10000,1000000"
run() { printf 'COMMAND:'; printf ' %q' "$@"; printf '\n'; ((DRY_RUN)) || "$@"; }
has_phase() { [[ "$PHASE" == all || "$PHASE" == "$1" ]]; }
env_base=(env JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1)
GPU_NOTE="not requested"
if [[ "$GPU" == auto ]]; then
  if selected_gpu="$($REPO_ROOT/benchmark/rebuttal/select_idle_gpu.sh 10 2>&1)"; then
    GPU="$selected_gpu"; GPU_NOTE="auto-selected GPU $GPU (utilization <= 10%)"
  else
    GPU_NOTE="GPU auto-selection unavailable: $selected_gpu"
  fi
fi
if [[ "$GPU" =~ ^[0-9]+$ ]]; then
  env_base=(env CUDA_VISIBLE_DEVICES="$GPU" JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1)
  GPU_NOTE="${GPU_NOTE}; CUDA_VISIBLE_DEVICES=$GPU"
fi

((DRY_RUN)) || mkdir -p "$RUN_DIR"
if ((!DRY_RUN)) && [[ ! -f "$RUN_DIR/run_config.txt" ]]; then
  {
    printf 'run_id=%s\nprofile=%s\npanel=%s\nseeds=%s\nsolvers=%s\ngpu=%s\ngpu_note=%s\n' "$RUN_ID" "$PROFILE" "$PANEL" "$SEEDS" "$SOLVERS" "$GPU" "$GPU_NOTE"
    printf 'm=%s\nn=%s\nsupport=%s\nsparsity=%s\nk_values=%s\n' "$M" "$N" "$S" "$D" "$KVALUES"
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true
    "$JULIA_BIN" --version
    nvidia-smi --query-gpu=index,name,uuid,memory.free,utilization.gpu --format=csv,noheader 2>&1 || true
  } > "$RUN_DIR/environment.txt"
  cp "$RUN_DIR/environment.txt" "$RUN_DIR/run_config.txt"
fi
if has_phase snapshot; then run "$REPO_ROOT/benchmark/rebuttal/snapshot_source.sh" --output "$RUN_DIR/source_snapshot"; fi
if has_phase test; then run "${env_base[@]}" "$JULIA_BIN" "--project=$REPO_ROOT" "$REPO_ROOT/test/rebuttal/ill_conditioned_lasso_cases_test.jl"; fi
if has_phase generate; then
  for seed in "${SEED_ARRAY[@]}"; do
    run "${env_base[@]}" "$JULIA_BIN" "--project=$REPO_ROOT" "$REPO_ROOT/benchmark/ill_conditioned_lasso.jl" \
      --mode generate --m "$M" --n "$N" --support "$S" --sparsity "$D" --k-values "$KVALUES" --seed "$seed" --panel "$PANEL" --output-dir "$RUN_DIR"
  done
fi
if has_phase solve; then
  for seed in "${SEED_ARRAY[@]}"; do for kval in 1 100 10000 1000000; do for solver in "${SOLVER_ARRAY[@]}"; do for tol in "${TOL_ARRAY[@]}"; do
    cache="$RUN_DIR/instances/seed_${seed}_K_${kval}.jls"
    run "${env_base[@]}" "$JULIA_BIN" "--project=$REPO_ROOT" "$REPO_ROOT/benchmark/ill_conditioned_lasso.jl" \
      --mode solve --cache "$cache" --solver "$solver" --tol "$tol" --time-limit "$LIMIT" --output-dir "$RUN_DIR" \
      --mosek-license "$MOSEK_LICENSE"
  done; done; done; done
fi
if has_phase analysis; then
  run "${env_base[@]}" "$JULIA_BIN" "--project=$REPO_ROOT" "$REPO_ROOT/benchmark/analyze_ill_conditioned_lasso.jl" --root "$RUN_DIR"
fi
printf 'PHASE_COMPLETE phase=%s run_dir=%s\n' "$PHASE" "$RUN_DIR"
