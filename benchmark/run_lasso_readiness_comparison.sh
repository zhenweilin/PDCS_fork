#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JULIA="${JULIA:-$ROOT/.julia-bin/julia}"
MAIN_DEPOT="${JULIA_DEPOT_PATH:-$ROOT/.julia-depot}"
CUCLARABEL_DEPOT="${CUCLARABEL_DEPOT:-/tmp/pdcs_cuclarabel_depot}"
SCS_GPU_DEPOT="${SCS_GPU_DEPOT:-/tmp/pdcs_scs_gpu_depot}"
GPU=7
TIME_LIMIT=300
TOL=1e-6
RUN_ID="readiness_$(date -u +%Y%m%dT%H%M%SZ)"
SOLVERS="pdcs_gpu,scs_gpu,cuclarabel"
MOSEK_LICENSE="$ROOT/mosek_lic/mosek.lic"

usage() {
    cat <<'EOF'
Usage: run_lasso_readiness_comparison.sh [options]
  --run-id ID
  --gpu INDEX
  --time-limit SECONDS
  --tol VALUE
  --solvers comma,separated,list
  --mosek-license FILE

The readiness suite uses one seed and is not a publication-statistics run:
  warm-up: m=100,  n=400,  s=10, d=10, K=1
  conditioning sweep: m=200, n=1000, s=20, d=10, K=1,100,10000,1000000
  scale probe: m=1000, n=5000, s=40, d=10, K=1
EOF
}

while (($#)); do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --run-id=*) RUN_ID="${1#*=}"; shift ;;
        --gpu) GPU="$2"; shift 2 ;;
        --gpu=*) GPU="${1#*=}"; shift ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --time-limit=*) TIME_LIMIT="${1#*=}"; shift ;;
        --tol) TOL="$2"; shift 2 ;;
        --tol=*) TOL="${1#*=}"; shift ;;
        --solvers) SOLVERS="$2"; shift 2 ;;
        --solvers=*) SOLVERS="${1#*=}"; shift ;;
        --mosek-license) MOSEK_LICENSE="$2"; shift 2 ;;
        --mosek-license=*) MOSEK_LICENSE="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "--gpu must be an index" >&2; exit 2; }
[[ -x "$JULIA" ]] || { echo "Julia is not executable: $JULIA" >&2; exit 2; }
if [[ ",$SOLVERS," == *,mosek,* ||
      ",$SOLVERS," == *,mosek_nopresolve,* ]]; then
    [[ -f "$MOSEK_LICENSE" ]] ||
        { echo "MOSEK license is missing: $MOSEK_LICENSE" >&2; exit 2; }
fi

RUN_DIR="$ROOT/benchmark/lasso_results/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] ||
    { echo "Refusing to overwrite existing run: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"/{raw_logs,results,warmup,tiny,smoke,source_state}

export CUDA_VISIBLE_DEVICES="$GPU"
export MOSEKLM_LICENSE_FILE="$MOSEK_LICENSE"
export PDCS_SKIP_GPU_PRECOMPILE=1

{
    echo "run_id=$RUN_ID"
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gpu=$GPU"
    echo "tolerance=$TOL"
    echo "time_limit_seconds=$TIME_LIMIT"
    echo "solvers=$SOLVERS"
    echo "main_depot=$MAIN_DEPOT"
    echo "cuclarabel_depot=$CUCLARABEL_DEPOT"
    echo "scs_gpu_depot=$SCS_GPU_DEPOT"
    "$JULIA" --version
    uname -a
    lscpu
    free -h
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu \
        --format=csv,noheader
} > "$RUN_DIR/environment.txt" 2>&1

git -C "$ROOT" rev-parse HEAD > "$RUN_DIR/source_state/git_head.txt" 2>&1 || true
git -C "$ROOT" status --short > "$RUN_DIR/source_state/git_status.txt" 2>&1 || true
git -C "$ROOT" diff --stat > "$RUN_DIR/source_state/git_diff_stat.txt" 2>&1 || true
sha256sum \
    "$ROOT/Project.toml" \
    "$ROOT/Manifest.toml" \
    "$ROOT/benchmark/ill_conditioned_lasso.jl" \
    "$ROOT/benchmark/ill_conditioned_lasso_cuclarabel.jl" \
    "$ROOT/benchmark/ill_conditioned_lasso_scs_gpu.jl" \
    "$ROOT/benchmark/rebuttal/ill_conditioned_lasso_cases.jl" \
    "$ROOT/benchmark/run_lasso_readiness_comparison.sh" \
    "$ROOT/benchmark/run_scs_gpu_lasso.sh" \
    "$ROOT/benchmark/scs_gpu_env/Project.toml" \
    "$ROOT/benchmark/scs_gpu_env/LocalPreferences.toml" \
    "$ROOT/benchmark/scs_gpu_env/Manifest.toml" \
    > "$RUN_DIR/source_state/source_hashes.sha256"

run_logged() {
    local log="$1"
    shift
    "$@" 2>&1 | tee "$log"
    local status=${PIPESTATUS[0]}
    printf '%s\n' "$status" > "${log%.raw.log}.exit_status.txt"
    return 0
}

MAIN_ENV=(
    env
    "CUDA_VISIBLE_DEVICES=$GPU"
    "JULIA_DEPOT_PATH=$MAIN_DEPOT"
    "MOSEKLM_LICENSE_FILE=$MOSEK_LICENSE"
    "PDCS_SKIP_GPU_PRECOMPILE=1"
)

run_logged "$RUN_DIR/raw_logs/generator_test.raw.log" \
    "${MAIN_ENV[@]}" "$JULIA" --project="$ROOT" \
    "$ROOT/test/rebuttal/ill_conditioned_lasso_cases_test.jl"

run_logged "$RUN_DIR/raw_logs/generate_warmup.raw.log" \
    "${MAIN_ENV[@]}" "$JULIA" --project="$ROOT" \
    "$ROOT/benchmark/ill_conditioned_lasso.jl" \
    --mode generate --m 100 --n 400 --support 10 --sparsity 10 \
    --k-values 1 --seed 1901 --panel fixed_lambda --lambda 0.01 \
    --output-dir "$RUN_DIR/warmup"

run_logged "$RUN_DIR/raw_logs/generate_tiny.raw.log" \
    "${MAIN_ENV[@]}" "$JULIA" --project="$ROOT" \
    "$ROOT/benchmark/ill_conditioned_lasso.jl" \
    --mode generate --m 200 --n 1000 --support 20 --sparsity 10 \
    --k-values 1,100,10000,1000000 --seed 2026 --panel fixed_lambda \
    --lambda 0.01 --output-dir "$RUN_DIR/tiny"

run_logged "$RUN_DIR/raw_logs/generate_smoke.raw.log" \
    "${MAIN_ENV[@]}" "$JULIA" --project="$ROOT" \
    "$ROOT/benchmark/ill_conditioned_lasso.jl" \
    --mode generate --m 1000 --n 5000 --support 40 --sparsity 10 \
    --k-values 1 --seed 2026 --panel fixed_lambda --lambda 0.01 \
    --output-dir "$RUN_DIR/smoke"

WARMUP_CACHE="$RUN_DIR/warmup/instances/seed_1901_K_1.jls"
IFS=, read -r -a SOLVER_ARRAY <<< "$SOLVERS"
CASE_NAMES=(tiny_K_1 tiny_K_100 tiny_K_10000 tiny_K_1000000 smoke_K_1)
CASE_CACHES=(
    "$RUN_DIR/tiny/instances/seed_2026_K_1.jls"
    "$RUN_DIR/tiny/instances/seed_2026_K_100.jls"
    "$RUN_DIR/tiny/instances/seed_2026_K_10000.jls"
    "$RUN_DIR/tiny/instances/seed_2026_K_1000000.jls"
    "$RUN_DIR/smoke/instances/seed_2026_K_1.jls"
)
for required_cache in "$WARMUP_CACHE" "${CASE_CACHES[@]}"; do
    [[ -f "$required_cache" ]] || {
        echo "Required generated cache is missing: $required_cache" >&2
        exit 1
    }
done

for solver in "${SOLVER_ARRAY[@]}"; do
    for index in "${!CASE_NAMES[@]}"; do
        case_name="${CASE_NAMES[$index]}"
        cache="${CASE_CACHES[$index]}"
        result_dir="$RUN_DIR/results/$solver/$case_name"
        mkdir -p "$result_dir"
        echo "CASE_START solver=$solver case=$case_name gpu=$GPU"
        if [[ "$solver" == cuclarabel ]]; then
            timeout --signal=INT --kill-after=60 "$((TIME_LIMIT + 300))" \
                env "CUCLARABEL_DEPOT=$CUCLARABEL_DEPOT" \
                    "GPU=$GPU" "TOL=$TOL" "TIME_LIMIT=$TIME_LIMIT" \
                    "VERBOSE_LEVEL=2" \
                "$ROOT/benchmark/run_cuclarabel_lasso.sh" \
                    --cache "$cache" --warmup-cache "$WARMUP_CACHE" \
                    --output-dir "$result_dir"
            status=$?
        elif [[ "$solver" == scs_gpu ]]; then
            timeout --signal=INT --kill-after=60 "$((TIME_LIMIT + 300))" \
                env "SCS_GPU_DEPOT=$SCS_GPU_DEPOT" \
                    "GPU=$GPU" "TOL=$TOL" "TIME_LIMIT=$TIME_LIMIT" \
                    "VERBOSE_LEVEL=2" \
                "$ROOT/benchmark/run_scs_gpu_lasso.sh" \
                    --cache "$cache" --warmup-cache "$WARMUP_CACHE" \
                    --output-dir "$result_dir"
            status=$?
        else
            timeout --signal=INT --kill-after=60 "$((TIME_LIMIT + 300))" \
                "${MAIN_ENV[@]}" "$JULIA" --project="$ROOT" \
                "$ROOT/benchmark/ill_conditioned_lasso.jl" \
                --mode solve --cache "$cache" --warmup-cache "$WARMUP_CACHE" \
                --solver "$solver" --tol "$TOL" --time-limit "$TIME_LIMIT" \
                --output-dir "$result_dir" --mosek-license "$MOSEK_LICENSE" \
                --verbose --verbose-level 2 --print-freq 1000 \
                2>&1 | tee "$result_dir/solver.raw.log"
            status=${PIPESTATUS[0]}
        fi
        printf '%s\n' "$status" > "$result_dir/exit_status.txt"
        echo "CASE_END solver=$solver case=$case_name exit_status=$status"
    done
done

"$ROOT/benchmark/analyze_lasso_readiness_comparison.py" "$RUN_DIR"
echo "LASSO_READINESS_COMPLETE run_dir=$RUN_DIR"
