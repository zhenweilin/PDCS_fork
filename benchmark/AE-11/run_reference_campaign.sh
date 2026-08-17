#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIBLING_ROOT="$(cd "$REPO_ROOT/../PDCS_fork" && pwd)"
JULIA="$SIBLING_ROOT/.julia-bin/julia-1.12.6/bin/julia"
DEPOT="$SIBLING_ROOT/.julia-depot"
PHASE="pilot"
PANELS=""
SEEDS=""
KAPPAS="1,1e2,1e4,1e6,1e8"
TOLERANCE="1e-9"
AFFINITY="97-128"
OUTPUT=""
TIME_LIMIT=""
WARMUP="true"
FORCE="false"

usage() {
    printf '%s\n' \
      "Usage: run_reference_campaign.sh [options]" \
      "  --phase pilot|medium|large" \
      "  --panels A|B|A,B" \
      "  --seeds comma,separated,list" \
      "  --kappas comma,separated,list" \
      "  --tolerance FLOAT  (default: 1e-9)" \
      "  --affinity CPU-LIST (default: 97-128)" \
      "  --output DIR" \
      "  --time-limit SECONDS  (otherwise use experiment.toml)" \
      "  --warmup true|false" \
      "  --force true|false"
}

while (($#)); do
    case "$1" in
      --phase) PHASE="$2"; shift 2 ;;
      --panels) PANELS="$2"; shift 2 ;;
      --seeds) SEEDS="$2"; shift 2 ;;
      --kappas) KAPPAS="$2"; shift 2 ;;
      --tolerance) TOLERANCE="$2"; shift 2 ;;
      --affinity) AFFINITY="$2"; shift 2 ;;
      --output) OUTPUT="$2"; shift 2 ;;
      --time-limit) TIME_LIMIT="$2"; shift 2 ;;
      --warmup) WARMUP="$2"; shift 2 ;;
      --force) FORCE="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$PHASE" in
  pilot)
    [[ -n "$SEEDS" ]] || SEEDS="2026,2027"
    [[ -n "$PANELS" ]] || PANELS="A,B"
    ;;
  medium)
    [[ -n "$SEEDS" ]] || SEEDS="2026,2027,2028,2029,2030"
    [[ -n "$PANELS" ]] || PANELS="A,B"
    ;;
  large)
    [[ -n "$SEEDS" ]] || SEEDS="2026,2027,2028"
    [[ -n "$PANELS" ]] || PANELS="A"
    ;;
  *) printf 'Invalid phase: %s\n' "$PHASE" >&2; exit 2 ;;
esac

[[ -x "$JULIA" ]] || { printf 'Julia not executable: %s\n' "$JULIA" >&2; exit 2; }
[[ -n "$OUTPUT" ]] || OUTPUT="$SCRIPT_DIR/results/formal_$PHASE"
OUTPUT="$(realpath -m "$OUTPUT")"
mkdir -p "$OUTPUT/logs" "$OUTPUT/environment" "$OUTPUT/results"

export JULIA_DEPOT_PATH="$DEPOT"
export JULIA_PKG_OFFLINE=true
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3
export PDCS_SKIP_GPU_PRECOMPILE=1
export AE11_BLAS_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

IFS=',' read -r -a seed_array <<<"$SEEDS"
IFS=',' read -r -a kappa_array <<<"$KAPPAS"
extra_time=()
[[ -z "$TIME_LIMIT" ]] || extra_time=(--time-limit "$TIME_LIMIT")
tolerance_tag="${TOLERANCE//./p}"
tolerance_tag="${tolerance_tag//-/m}"
log="$OUTPUT/logs/reference_pdcs_cpu_${PHASE}_tol${tolerance_tag}.log"

{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    printf 'phase=%s seeds=%s kappas=%s panels=%s tolerance=%s affinity=%s\n' \
      "$PHASE" "$SEEDS" "$KAPPAS" "$PANELS" "$TOLERANCE" "$AFFINITY"
    "$JULIA" --version
    sha256sum "$SCRIPT_DIR/experiment.toml" "$SCRIPT_DIR/AE11Common.jl" \
      "$SCRIPT_DIR/run_moi_solver.jl" "$0"
} >>"$OUTPUT/environment/reference_environment.txt" 2>&1

for seed in "${seed_array[@]}"; do
  for kappa in "${kappa_array[@]}"; do
    taskset -c "$AFFINITY" "$JULIA" --startup-file=no --threads=32 \
      --project="$REPO_ROOT" "$SCRIPT_DIR/run_moi_solver.jl" \
      --solver pdcs_cpu --profile "$PHASE" --seed "$seed" \
      --kappa "$kappa" --tolerance "$TOLERANCE" --panels "$PANELS" \
      --output-dir "$OUTPUT/results" --warmup "$WARMUP" --force "$FORCE" \
      "${extra_time[@]}" >>"$log" 2>&1
  done
done

python3 "$SCRIPT_DIR/analyze_results.py" \
  --config "$SCRIPT_DIR/experiment.toml" --results-dir "$OUTPUT/results" \
  --output-dir "$OUTPUT/analysis" --allow-incomplete
