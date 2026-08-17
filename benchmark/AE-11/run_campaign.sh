#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIBLING_ROOT="$(cd "$REPO_ROOT/../PDCS_fork" && pwd)"
CONFIG="$SCRIPT_DIR/experiment.toml"
JULIA="$SIBLING_ROOT/.julia-bin/julia-1.12.6/bin/julia"
BASE_DEPOT="$SIBLING_ROOT/.julia-depot"
CUCLARABEL_DEPOT="$SIBLING_ROOT/.julia-depot-cuclarabel:$BASE_DEPOT"
SCS_GPU_PROJECT="$SIBLING_ROOT/benchmark/scs_gpu_env"
CUCLARABEL_PROJECT="$SIBLING_ROOT/benchmark/cuclarabel_env"
PHASE="pilot"
SOLVERS="cupdcs_on,cupdcs_off,pdcs_cpu,scs_indirect,clarabel_cpu,mosek,scs_gpu,cuclarabel"
PANELS=""
TOLERANCES="1e-3,1e-6"
SEEDS=""
KAPPAS="1,1e2,1e4,1e6,1e8"
GPUS="0,1,2,3,4,5,6,7"
OUTPUT="$SCRIPT_DIR/results/formal"
TIME_LIMIT=""
WARMUP="true"
FORCE="false"

usage() {
    printf '%s\n' \
      "Usage: run_campaign.sh [options]" \
      "  --phase pilot|medium|large" \
      "  --solvers comma,separated,list" \
      "  --panels A|B|A,B" \
      "  --tolerances comma,separated,list" \
      "  --seeds comma,separated,list" \
      "  --kappas comma,separated,list" \
      "  --gpus comma,separated,list" \
      "  --output DIR" \
      "  --time-limit SECONDS  (otherwise use experiment.toml)" \
      "  --warmup true|false" \
      "  --force true|false"
}

while (($#)); do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --solvers) SOLVERS="$2"; shift 2 ;;
        --panels) PANELS="$2"; shift 2 ;;
        --tolerances) TOLERANCES="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --kappas) KAPPAS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --force) FORCE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$PHASE" in
    pilot) [[ -n "$SEEDS" ]] || SEEDS="2026,2027" ;;
    medium) [[ -n "$SEEDS" ]] || SEEDS="2026,2027,2028,2029,2030" ;;
    large)
        [[ -n "$SEEDS" ]] || SEEDS="2026,2027,2028"
        [[ -n "$PANELS" ]] || PANELS="A"
        ;;
    *) printf 'Invalid phase: %s\n' "$PHASE" >&2; exit 2 ;;
esac
[[ -n "$PANELS" ]] || PANELS="A,B"
[[ -x "$JULIA" ]] || { printf 'Julia not executable: %s\n' "$JULIA" >&2; exit 2; }
OUTPUT="$(realpath -m "$OUTPUT")"
mkdir -p "$OUTPUT/logs" "$OUTPUT/environment"

export JULIA_PKG_OFFLINE=true
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3
export PDCS_SKIP_GPU_PRECOMPILE=1
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
export AE11_BLAS_THREADS=32
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32

ARTIFACT_DIR="$OUTPUT/environment/production_sm90"
if [[ "$SOLVERS" == *cupdcs* ]]; then
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
    else
      printf 'Reusing complete production artifact set in %s\n' \
        "$ARTIFACT_DIR" >"$OUTPUT/logs/build_production_artifacts.log"
    fi
    export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$ARTIFACT_DIR"
    sha256sum "$ARTIFACT_DIR"/* >"$OUTPUT/environment/artifact_hashes.sha256"
fi

{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    uname -a
    "$JULIA" --version
    nvidia-smi
    git -C "$REPO_ROOT" rev-parse HEAD
    git -C "$SIBLING_ROOT" rev-parse HEAD
    sha256sum "$CONFIG" "$SCRIPT_DIR/AE11Common.jl" \
      "$SCRIPT_DIR/run_cupdcs.jl" "$SCRIPT_DIR/run_moi_solver.jl" "$0"
} >"$OUTPUT/environment/environment.txt" 2>&1

IFS=',' read -r -a solver_array <<<"$SOLVERS"
IFS=',' read -r -a seed_array <<<"$SEEDS"
IFS=',' read -r -a kappa_array <<<"$KAPPAS"
IFS=',' read -r -a tolerance_array <<<"$TOLERANCES"
IFS=',' read -r -a gpu_array <<<"$GPUS"

extra_time=()
[[ -z "$TIME_LIMIT" ]] || extra_time=(--time-limit "$TIME_LIMIT")

run_cupdcs_shard() {
    local slot="$1"
    local gpu="${gpu_array[$slot]}"
    local wanted="$2"
    local counter=0
    local seed kappa tolerance mode
    for seed in "${seed_array[@]}"; do
      for kappa in "${kappa_array[@]}"; do
        for tolerance in "${tolerance_array[@]}"; do
          for mode in on off; do
            [[ "cupdcs_$mode" == "$wanted" ]] || continue
            if ((counter % ${#gpu_array[@]} == slot)); then
              CUDA_VISIBLE_DEVICES="$gpu" JULIA_DEPOT_PATH="$BASE_DEPOT" \
                "$JULIA" --startup-file=no --project="$REPO_ROOT" \
                "$SCRIPT_DIR/run_cupdcs.jl" \
                --profile "$PHASE" --seed "$seed" --kappa "$kappa" \
                --tolerance "$tolerance" --rescaling "$mode" \
                --panels "$PANELS" --output-dir "$OUTPUT/results" \
                --warmup "$WARMUP" --force "$FORCE" \
                "${extra_time[@]}"
            fi
            counter=$((counter + 1))
          done
        done
      done
    done
}

run_moi_solver() {
    local solver="$1"
    local project="$REPO_ROOT"
    local depot="$BASE_DEPOT"
    local gpu_prefix=()
    case "$solver" in
      scs_gpu)
        project="$SCS_GPU_PROJECT"
        gpu_prefix=(env CUDA_VISIBLE_DEVICES="${gpu_array[0]}")
        ;;
      clarabel_cpu)
        project="$REPO_ROOT"
        depot="$BASE_DEPOT"
        ;;
      cuclarabel)
        project="$CUCLARABEL_PROJECT"
        depot="$CUCLARABEL_DEPOT"
        gpu_prefix=(env CUDA_VISIBLE_DEVICES="${gpu_array[0]}")
        ;;
    esac
    local seed kappa tolerance
    for seed in "${seed_array[@]}"; do
      for kappa in "${kappa_array[@]}"; do
        for tolerance in "${tolerance_array[@]}"; do
          "${gpu_prefix[@]}" env JULIA_DEPOT_PATH="$depot" \
            "$JULIA" --startup-file=no --threads=32 --project="$project" \
            "$SCRIPT_DIR/run_moi_solver.jl" --solver "$solver" \
            --profile "$PHASE" --seed "$seed" --kappa "$kappa" \
            --tolerance "$tolerance" --panels "$PANELS" \
            --output-dir "$OUTPUT/results" --warmup "$WARMUP" \
            --force "$FORCE" "${extra_time[@]}"
        done
      done
    done
}

for solver in "${solver_array[@]}"; do
    case "$solver" in
      cupdcs_on|cupdcs_off)
        pids=()
        for slot in "${!gpu_array[@]}"; do
          run_cupdcs_shard "$slot" "$solver" \
            >"$OUTPUT/logs/${solver}_gpu${gpu_array[$slot]}.log" 2>&1 &
          pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
        ;;
      pdcs_cpu|scs_indirect|clarabel_cpu|mosek|scs_gpu|cuclarabel)
        run_moi_solver "$solver" >"$OUTPUT/logs/${solver}.log" 2>&1
        ;;
      *) printf 'Unsupported solver: %s\n' "$solver" >&2; exit 2 ;;
    esac
done

python3 "$SCRIPT_DIR/analyze_results.py" \
  --config "$CONFIG" --results-dir "$OUTPUT/results" \
  --output-dir "$OUTPUT/analysis" --allow-incomplete
