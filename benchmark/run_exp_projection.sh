#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
JULIA_BIN="${PDCS_JULIA:-}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
CUDA_ROOT="${CUDA_HOME:-}"
GPU_ARCH="${PDCS_GPU_ARCH:-}"
COUNTS="3,10,100,1000,10000,100000,1000000"
VARIANTS="primalDiagonal"
INPUT_DISTRIBUTION="heterogeneous"
SIGMA="1.0"
STRATEGIES="gridWise,blockWise,warpWise,threadWise"
TRIALS=10
SEED=2026
MAX_GRIDWISE_CONES=10000
RUN_LABEL=""
OUTPUT_ROOT="$REPO_ROOT/benchmark/results/exp_projection"
CUDA_RUNTIME_SOURCE="${PDCS_CUDA_RUNTIME_SOURCE:-local}"
NO_BUILD=0
SMOKE=0

usage() {
  sed -n '10,35p' "$0" | sed -n 's/^# //p'
}

# GPU exponential-cone projection benchmark.
#
# Options:
#   --julia PATH
#   --julia-depot PATH
#   --cuda-home PATH
#   --cuda-runtime local|artifact   (default: local)
#   --arch sm_XX
#   --cone-counts LIST
#   --variants LIST                 primalDiagonal (default)
#   --input-distribution MODE       similar|heterogeneous (default: heterogeneous)
#   --sigma FLOAT                   std. dev. of heterogeneous N(0,sigma^2) inputs
#   --strategies LIST               gridWise,blockWise,warpWise,threadWise
#   --trials N
#   --seed N
#   --max-gridwise-cones N
#   --output-dir PATH
#   --run-label NAME
#   --smoke                         counts 3,100,10000; two trials
#   --no-build
#   --help

while (($#)); do
  case "$1" in
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --cuda-runtime) CUDA_RUNTIME_SOURCE="$2"; shift 2 ;;
    --arch) GPU_ARCH="$2"; shift 2 ;;
    --cone-counts) COUNTS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --input-distribution) INPUT_DISTRIBUTION="$2"; shift 2 ;;
    --sigma) SIGMA="$2"; shift 2 ;;
    --strategies) STRATEGIES="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-gridwise-cones) MAX_GRIDWISE_CONES="$2"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-label) RUN_LABEL="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --no-build) NO_BUILD=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$JULIA_BIN" ]]; then
  [[ -x "$REPO_ROOT/.julia-bin/julia" ]] && JULIA_BIN="$REPO_ROOT/.julia-bin/julia"
  [[ -n "$JULIA_BIN" ]] || JULIA_BIN="$(command -v julia || true)"
fi
[[ -x "$JULIA_BIN" ]] || { printf 'Julia is not executable: %s\n' "$JULIA_BIN" >&2; exit 1; }

if [[ -z "$CUDA_ROOT" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_ROOT="$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_ROOT=/usr/local/cuda
  else
    printf 'CUDA toolkit not found; pass --cuda-home.\n' >&2; exit 1
  fi
fi
CUDA_ROOT="$(cd -- "$CUDA_ROOT" && pwd)"
NVCC="$CUDA_ROOT/bin/nvcc"
[[ -x "$NVCC" ]] || { printf 'nvcc not found under %s.\n' "$CUDA_ROOT" >&2; exit 1; }
[[ "$CUDA_RUNTIME_SOURCE" == local || "$CUDA_RUNTIME_SOURCE" == artifact ]] || {
  printf 'Expected --cuda-runtime local or artifact.\n' >&2; exit 1
}

CUDA_TOOLKIT="$($NVCC --version | awk '/release/{version=$5; gsub(/,/,"",version); print version; exit}')"
[[ -n "$CUDA_TOOLKIT" ]] || { printf 'Cannot parse the CUDA toolkit version.\n' >&2; exit 1; }
RUN_LABEL="${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_LABEL"
mkdir -p "$JULIA_DEPOT" "$RUN_DIR"

if ((SMOKE)); then
  COUNTS="3,100,10000"
  TRIALS=2
fi

# Configure the runtime before importing CUDA, avoiding CUDA_Runtime artifact
# downloads in local mode.
"$JULIA_BIN" --startup-file=no -e '
  using TOML
  root, version, source = ARGS
  path = joinpath(root, "LocalPreferences.toml")
  preferences = isfile(path) ? TOML.parsefile(path) : Dict{String,Any}()
  runtime = Dict{String,Any}("version" => version)
  source == "local" && (runtime["local"] = "true")
  preferences["CUDA_Runtime_jll"] = runtime
  open(path, "w") do io
      TOML.print(io, preferences; sorted=true)
  end
' "$REPO_ROOT" "$CUDA_TOOLKIT" "$CUDA_RUNTIME_SOURCE"

env JULIA_DEPOT_PATH="$JULIA_DEPOT" "$JULIA_BIN" --project="$REPO_ROOT" \
  -e 'using Pkg; Pkg.instantiate()'

env PATH="$CUDA_ROOT/bin:$PATH" CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
  JULIA_DEPOT_PATH="$JULIA_DEPOT" "$JULIA_BIN" --project="$REPO_ROOT" -e '
    using CUDA, LinearAlgebra
    CUDA.functional() || error("CUDA is not functional")
    value = norm(CUDA.ones(Float64, 33), 2)
    isfinite(value) || error("cuBLAS preflight failed")
    println("CUDA.jl cuBLAS preflight PASS: ", value)
    CUDA.versioninfo()
  '

if [[ -z "$GPU_ARCH" ]]; then
  GPU_ARCH="$(env PATH="$CUDA_ROOT/bin:$PATH" CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
    JULIA_DEPOT_PATH="$JULIA_DEPOT" "$JULIA_BIN" --project="$REPO_ROOT" -e '
      using CUDA
      capability = CUDA.capability(CUDA.device())
      print("sm_", capability.major, capability.minor)
    ')"
fi
[[ "$GPU_ARCH" =~ ^sm_[0-9]+$ ]] || { printf 'Invalid architecture: %s\n' "$GPU_ARCH" >&2; exit 1; }

if ((!NO_BUILD)); then
  make -C "$REPO_ROOT/src/pdcs_gpu/cuda" rebuild-gpu CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
fi

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')"
COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || printf unknown)"
RAW="$RUN_DIR/raw.csv"
SUMMARY="$RUN_DIR/summary.csv"
REPORT="$RUN_DIR/report.md"

{
  printf 'run_label=%s\n' "$RUN_LABEL"
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'cuda_home=%s\n' "$CUDA_ROOT"
  printf 'cuda_toolkit=%s\n' "$CUDA_TOOLKIT"
  printf 'cuda_runtime_source=%s\n' "$CUDA_RUNTIME_SOURCE"
  printf 'gpu_arch=%s\n' "$GPU_ARCH"
  printf 'variants=%s\n' "$VARIANTS"
  printf 'input_distribution=%s\n' "$INPUT_DISTRIBUTION"
  printf 'input_sigma=%s\n' "$SIGMA"
  printf 'seed=%s\n' "$SEED"
  printf 'strategies=%s\n' "$STRATEGIES"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
} > "$RUN_DIR/environment.txt"

env PATH="$CUDA_ROOT/bin:$PATH" CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
  JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1 \
  PDCS_RUN_ID="$RUN_LABEL" PDCS_CUDA_TOOLKIT="$CUDA_TOOLKIT" \
  PDCS_NVIDIA_DRIVER="$DRIVER" PDCS_GIT_COMMIT="$COMMIT" \
  "$JULIA_BIN" --project="$REPO_ROOT" "$REPO_ROOT/benchmark/exp_cone_projection.jl" \
    --cone-counts "$COUNTS" --variants "$VARIANTS" \
    --input-distribution "$INPUT_DISTRIBUTION" --sigma "$SIGMA" \
    --strategies "$STRATEGIES" \
    --trials "$TRIALS" --seed "$SEED" \
    --max-gridwise-cones "$MAX_GRIDWISE_CONES" --output "$RAW" \
    2> "$RUN_DIR/benchmark.log"

env JULIA_DEPOT_PATH="$JULIA_DEPOT" "$JULIA_BIN" --project="$REPO_ROOT" \
  "$REPO_ROOT/benchmark/summarize_exp_projection.jl" \
  --raw "$RAW" --summary "$SUMMARY" --report "$REPORT"

printf 'Raw CSV: %s\nSummary CSV: %s\nReport: %s\n' "$RAW" "$SUMMARY" "$REPORT"
