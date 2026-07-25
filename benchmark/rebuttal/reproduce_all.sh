#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
ARCH="${PDCS_GPU_ARCH:-}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="$REPO_ROOT/benchmark/results/rebuttal"
MODE=smoke
PROFILES=0
APPLICATIONS=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --full) MODE=full; shift ;;
    --profiles) PROFILES=1; shift ;;
    --applications) APPLICATIONS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      printf '%s\n' 'Usage: reproduce_all.sh [--full] [--profiles] [--applications] [--gpu N] [--arch sm_XX] [--run-id ID] [--dry-run]'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { printf 'Refusing to overwrite %s\n' "$RUN_DIR" >&2; exit 1; }
[[ -x "$JULIA_BIN" ]] || { printf 'Julia not executable: %s\n' "$JULIA_BIN" >&2; exit 1; }
NVCC="$CUDA_ROOT/bin/nvcc"
[[ -x "$NVCC" ]] || { printf 'nvcc not found: %s\n' "$NVCC" >&2; exit 1; }

if [[ -z "$ARCH" ]]; then
  ARCH="$(env CUDA_VISIBLE_DEVICES="$GPU" JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" --project="$REPO_ROOT" -e 'using CUDA; c=CUDA.capability(CUDA.device()); print("sm_",c.major,c.minor)')"
fi
[[ "$ARCH" =~ ^sm_[0-9]+$ ]] || { printf 'Invalid architecture: %s\n' "$ARCH" >&2; exit 2; }

if ((DRY_RUN)); then
  printf 'run_dir=%s\nmode=%s\nprofiles=%s\napplications=%s\ngpu=%s\narch=%s\n' \
    "$RUN_DIR" "$MODE" "$PROFILES" "$APPLICATIONS" "$GPU" "$ARCH"
  exit 0
fi

mkdir -p "$RUN_DIR"/{audit,strategy_map,strategy_validation,soc_root,exp_root,logs}
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"
export JULIA_DEPOT_PATH="$JULIA_DEPOT"
export PDCS_SKIP_GPU_PRECOMPILE=1
export PDCS_GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"

{
  printf 'run_id=%s\nmode=%s\narch=%s\ngpu_index=%s\ngit_commit=%s\n' \
    "$RUN_ID" "$MODE" "$ARCH" "$GPU" "$PDCS_GIT_COMMIT"
  "$JULIA_BIN" --version
  "$NVCC" --version
  nvidia-smi -i "$GPU" --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,memory.free,power.limit,persistence_mode,compute_mode,mig.mode.current --format=csv
} > "$RUN_DIR/environment.txt" 2>&1

"$JULIA_BIN" --project="$REPO_ROOT" -e 'using Pkg; Pkg.instantiate()' \
  > "$RUN_DIR/logs/instantiate.log" 2>&1
make -C "$REPO_ROOT/src/pdcs_gpu/cuda" rebuild-gpu rebuild-profile \
  CUDA_HOME="$CUDA_ROOT" ARCH="$ARCH" > "$RUN_DIR/logs/build.log" 2>&1

for test_file in test_rebuttal_case_generation.jl test_rebuttal_permutations.jl \
                 test_rebuttal_correctness.jl test_rebuttal_timing.jl; do
  "$JULIA_BIN" --project="$REPO_ROOT" "$REPO_ROOT/test/$test_file" \
    > "$RUN_DIR/logs/$test_file.log" 2>&1
done

"$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/audit_existing_results.jl" \
  --root "$REPO_ROOT/rebuttal_plan" \
  --output "$RUN_DIR/audit/summary_inventory.csv"

if [[ "$MODE" == full ]]; then
  COUNTS=3,10,100,1000,10000,100000,1000000
  DIMS=10,32,100,500,2000,10000,50000
  TRIALS=10
  ROOT_COUNT=1048576
  ROOT_DIMS=10,32,128,256
  SEEDS=2026,2027,2028,2029,2030,2031,2032,2033,2034,2035
else
  COUNTS=3,32
  DIMS=10,100
  TRIALS=1
  ROOT_COUNT=1024
  ROOT_DIMS=10
  SEEDS=2026
fi

"$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/soc_strategy_map.jl" \
  --cone-counts "$COUNTS" --dimensions "$DIMS" --trials "$TRIALS" --sigma 2 \
  --output "$RUN_DIR/strategy_map/timings_raw.csv"
"$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/summarize_strategy_map.jl" \
  --raw "$RUN_DIR/strategy_map/timings_raw.csv" \
  --output "$RUN_DIR/strategy_map/strategy_map.csv" \
  --heuristic "$RUN_DIR/strategy_map/heuristic.csv"

if [[ "$MODE" == full ]]; then
  "$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/soc_strategy_map.jl" \
    --cone-counts 30,300,3000,30000,300000 \
    --dimensions 20,64,256,1000,5000,25000 --trials 10 --sigma 2 \
    --output "$RUN_DIR/strategy_validation/timings_raw.csv"
  "$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/summarize_strategy_map.jl" \
    --raw "$RUN_DIR/strategy_validation/timings_raw.csv" \
    --output "$RUN_DIR/strategy_validation/strategy_map.csv" \
    --heuristic "$RUN_DIR/strategy_validation/oracle.csv"
  "$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/validate_strategy_heuristic.jl" \
    --heuristic "$RUN_DIR/strategy_map/heuristic.csv" \
    --validation "$RUN_DIR/strategy_validation/strategy_map.csv" \
    --output "$RUN_DIR/strategy_validation/heuristic_validation.csv"
fi

"$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/soc_root_profile.jl" \
  --dimensions "$ROOT_DIMS" --cone-count "$ROOT_COUNT" --seeds "$SEEDS" \
  --trials 1 --output "$RUN_DIR/soc_root"
"$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/exp_root_profile.jl" \
  --cone-count "$ROOT_COUNT" --seeds "$SEEDS" --trials 1 \
  --output "$RUN_DIR/exp_root"
"$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/summarize_root_profiles.jl" \
  --root "$RUN_DIR" --output "$RUN_DIR/root_profile_summary.csv"

if ((PROFILES)); then
  "$SCRIPT_DIR/run_nsys_matrix.sh" --kinds soc,exp --gpu "$GPU" \
    --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT" \
    --cone-count "$ROOT_COUNT" --seeds 2026,2027,2028 --duration 30 \
    --output-root "$RUN_DIR/nsys"
fi

if ((APPLICATIONS)); then
  "$JULIA_BIN" --project="$REPO_ROOT" "$SCRIPT_DIR/application_trace.jl" \
    --manifest "$SCRIPT_DIR/application_manifest.csv" \
    --output "$RUN_DIR/application"
fi

printf 'Rebuttal experiment package complete: %s\n' "$RUN_DIR"
