#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

JULIA_BIN="${PDCS_JULIA:-}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
CUDA_ROOT="${CUDA_HOME:-}"
GPU_ARCH="${PDCS_GPU_ARCH:-}"
OUTPUT_ROOT=""
RUN_LABEL=""
SMOKE=0
NO_BUILD=0
NO_PLOT=0
DRY_RUN=0
CUDA_RUNTIME_SOURCE="${PDCS_CUDA_RUNTIME_SOURCE:-local}"
CUSTOM_DIMENSIONS=""
CUSTOM_TRIALS=""
CUSTOM_CONE_COUNT="100"
CUSTOM_STRATEGIES="gridWise,blockWise,warpWise,threadWise"
SKIPPED_STRATEGIES=""

usage() {
  sed -n '2,60p' "$0" | sed -n 's/^# //p'
}

# Portable runner for the fixed-count SOC dimension experiment.
#
# Usage:
#   benchmark/run_soc_dimension_projection.sh [options]
#
# Options:
#   --julia PATH          Julia executable (local .julia-bin/julia is preferred)
#   --julia-depot PATH    Julia depot (default: repository .julia-depot)
#   --cuda-home PATH      CUDA toolkit root containing bin/nvcc
#   --arch sm_XX          GPU architecture; detected with CUDA.jl by default
#   --output-dir PATH     Parent output directory
#   --run-label NAME      Stable name for this run (default: UTC timestamp)
#   --dimensions LIST     Comma-separated full SOC dimensions
#   --trials N            Independent trials per dimension and strategy
#   --cone-count N        Fixed number of SOC blocks (default: 100)
#   --strategies LIST     Names to run: gridWise,blockWise,warpWise,threadWise
#   --skip-strategies LIST  Record omitted methods as SKIPPED_TIMEOUT_RISK
#   --smoke               Run dimensions 4,64,1024 with two trials
#   --no-build            Use existing PTX/shared-library artifacts
#   --no-plot             Do not invoke pdflatex/pdftoppm
#   --dry-run             Print resolved configuration and commands only
#   --cuda-runtime MODE   local (default) or artifact
#   --help                Show this help

while (($#)); do
  case "$1" in
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --arch) GPU_ARCH="$2"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-label) RUN_LABEL="$2"; shift 2 ;;
    --dimensions) CUSTOM_DIMENSIONS="$2"; shift 2 ;;
    --trials) CUSTOM_TRIALS="$2"; shift 2 ;;
    --cone-count) CUSTOM_CONE_COUNT="$2"; shift 2 ;;
    --strategies) CUSTOM_STRATEGIES="$2"; shift 2 ;;
    --skip-strategies) SKIPPED_STRATEGIES="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --no-build) NO_BUILD=1; shift ;;
    --no-plot) NO_PLOT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --cuda-runtime) CUDA_RUNTIME_SOURCE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$JULIA_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.julia-bin/julia" ]]; then
    JULIA_BIN="$REPO_ROOT/.julia-bin/julia"
  elif command -v julia >/dev/null 2>&1; then
    JULIA_BIN="$(command -v julia)"
  else
    printf 'Julia was not found. Pass --julia PATH or set PDCS_JULIA.\n' >&2
    exit 1
  fi
fi
[[ -x "$JULIA_BIN" ]] || { printf 'Julia is not executable: %s\n' "$JULIA_BIN" >&2; exit 1; }

if [[ -z "$CUDA_ROOT" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_ROOT="$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_ROOT=/usr/local/cuda
  else
    printf 'nvcc was not found. Pass --cuda-home PATH or set CUDA_HOME.\n' >&2
    exit 1
  fi
fi
CUDA_ROOT="$(cd -- "$CUDA_ROOT" && pwd)"
NVCC="$CUDA_ROOT/bin/nvcc"
[[ -x "$NVCC" ]] || { printf 'nvcc is not executable: %s\n' "$NVCC" >&2; exit 1; }
[[ "$CUDA_RUNTIME_SOURCE" == local || "$CUDA_RUNTIME_SOURCE" == artifact ]] || {
  printf 'Invalid --cuda-runtime value: %s (expected local or artifact)\n' "$CUDA_RUNTIME_SOURCE" >&2
  exit 1
}

CUDA_TOOLKIT="$($NVCC --version | awk '/release/{version=$5; gsub(/,/,"",version); print version; exit}')"
[[ -n "$CUDA_TOOLKIT" ]] || { printf 'Could not determine CUDA version from %s.\n' "$NVCC" >&2; exit 1; }

RUN_LABEL="${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/benchmark/results/soc_dimension}"
RUN_DIR="$OUTPUT_ROOT/$RUN_LABEL"
RAW_CSV="$RUN_DIR/raw.csv"
SUMMARY_CSV="$RUN_DIR/summary.csv"
REPORT="$RUN_DIR/report.md"
FIGURE_DIR="$REPO_ROOT/rebuttal_plan/figures"
FIGURE_TEX="$FIGURE_DIR/soc_dimension.tex"

printf 'Repository: %s\nJulia: %s\nJulia depot: %s\nCUDA_HOME: %s\nRun directory: %s\n' \
  "$REPO_ROOT" "$JULIA_BIN" "$JULIA_DEPOT" "$CUDA_ROOT" "$RUN_DIR"

if ((DRY_RUN)); then
  printf 'Architecture: %s\n' "${GPU_ARCH:-auto-detect with CUDA.jl}"
  printf 'Dry run: no packages, binaries, results, or figures were changed.\n'
  exit 0
fi

command -v make >/dev/null 2>&1 || { printf 'GNU Make is required.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { printf 'nvidia-smi is required.\n' >&2; exit 1; }
nvidia-smi >/dev/null || { printf 'nvidia-smi cannot communicate with the NVIDIA driver.\n' >&2; exit 1; }

mkdir -p "$JULIA_DEPOT" "$RUN_DIR" "$FIGURE_DIR"

# Write the CUDA runtime preference before Pkg.instantiate() or `using CUDA`.
# This prevents CUDA_Runtime artifact downloads when local mode is selected.
"$JULIA_BIN" --startup-file=no -e '
  using TOML
  project_root, version, source = ARGS
  path = joinpath(project_root, "LocalPreferences.toml")
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

# This must run in a fresh process because CUDA runtime preferences take effect
# only after Julia restarts. Test a real cuBLAS operation, not just allocation.
env PATH="$CUDA_ROOT/bin:$PATH" \
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
  JULIA_DEPOT_PATH="$JULIA_DEPOT" \
  "$JULIA_BIN" --project="$REPO_ROOT" -e '
    using CUDA, LinearAlgebra
    CUDA.functional() || error("CUDA.functional() is false")
    x = CUDA.ones(Float64, 33)
    value = norm(x, 2)
    isfinite(value) || error("CUDA.jl cuBLAS norm returned a non-finite value")
    println("CUDA.jl cuBLAS preflight PASS: norm=", value)
    CUDA.versioninfo()
  '

if [[ -z "$GPU_ARCH" ]]; then
  GPU_ARCH="$(env PATH="$CUDA_ROOT/bin:$PATH" \
    CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" --project="$REPO_ROOT" -e '
    using CUDA
    CUDA.functional() || error("CUDA.functional() is false")
    capability = CUDA.capability(CUDA.device())
    print("sm_", capability.major, capability.minor)
  ')"
fi
[[ "$GPU_ARCH" =~ ^sm_[0-9]+$ ]] || { printf 'Invalid GPU architecture: %s\n' "$GPU_ARCH" >&2; exit 1; }

NVIDIA_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
printf 'Detected architecture: %s\nCUDA toolkit/runtime: %s (%s)\n' \
  "$GPU_ARCH" "$CUDA_TOOLKIT" "$CUDA_RUNTIME_SOURCE"

if ((!NO_BUILD)); then
  make -C "$REPO_ROOT/src/pdcs_gpu/cuda" rebuild-gpu CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
fi

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || printf unknown)"
{
  printf 'run_label=%s\n' "$RUN_LABEL"
  printf 'repository=%s\n' "$REPO_ROOT"
  printf 'git_commit=%s\n' "$GIT_COMMIT"
  printf 'julia=%s\n' "$($JULIA_BIN --version)"
  printf 'cuda_home=%s\n' "$CUDA_ROOT"
  printf 'cuda_toolkit=%s\n' "$CUDA_TOOLKIT"
  printf 'gpu_arch=%s\n' "$GPU_ARCH"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
} > "$RUN_DIR/environment.txt"

DIMENSIONS="4,16,64,256,1024,4096,16384,65536,262144,1048576,4194304"
TRIALS=10
if ((SMOKE)); then
  DIMENSIONS="4,64,1024"
  TRIALS=2
fi
[[ -n "$CUSTOM_DIMENSIONS" ]] && DIMENSIONS="$CUSTOM_DIMENSIONS"
[[ -n "$CUSTOM_TRIALS" ]] && TRIALS="$CUSTOM_TRIALS"

env JULIA_DEPOT_PATH="$JULIA_DEPOT" \
  PATH="$CUDA_ROOT/bin:$PATH" \
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
  PDCS_SKIP_GPU_PRECOMPILE=1 \
  PDCS_RUN_ID="$RUN_LABEL" \
  PDCS_CUDA_TOOLKIT="$CUDA_TOOLKIT" \
  PDCS_NVIDIA_DRIVER="$NVIDIA_DRIVER" \
  PDCS_GIT_COMMIT="$GIT_COMMIT" \
  PDCS_SOC_TRIALS="$TRIALS" \
  "$JULIA_BIN" --project="$REPO_ROOT" "$REPO_ROOT/benchmark/soc_dimension_projection.jl" \
  --cone-count "$CUSTOM_CONE_COUNT" --dimensions "$DIMENSIONS" --trials "$TRIALS" \
  --strategies "$CUSTOM_STRATEGIES" --skip-strategies "$SKIPPED_STRATEGIES" --output "$RAW_CSV" \
  2>"$RUN_DIR/benchmark.log"

env JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SOC_TRIALS="$TRIALS" \
  PATH="$CUDA_ROOT/bin:$PATH" \
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
  "$JULIA_BIN" --project="$REPO_ROOT" \
  "$REPO_ROOT/benchmark/summarize_soc_dimension.jl" \
  --raw "$RAW_CSV" --summary "$SUMMARY_CSV" --report "$REPORT" --figure-tex "$FIGURE_TEX" \
  2>"$RUN_DIR/summary.log"

if ((!NO_PLOT)); then
  if command -v pdflatex >/dev/null 2>&1 && kpsewhich pgfplots.sty >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$FIGURE_DIR" "$FIGURE_TEX" \
      >"$RUN_DIR/pdflatex.log"
    if command -v pdftoppm >/dev/null 2>&1; then
      pdftoppm -png -r 300 -singlefile "$FIGURE_DIR/soc_dimension.pdf" "$FIGURE_DIR/soc_dimension" \
        >"$RUN_DIR/pdftoppm.log" 2>&1
    else
      printf 'pdftoppm not found; PDF was created but PNG was skipped.\n' >&2
    fi
  else
    printf 'pdflatex/PGFPlots not found; CSV, report, and figure TeX were still created.\n' >&2
  fi
fi

printf 'Raw CSV: %s\nSummary CSV: %s\nReport: %s\nFigure source: %s\n' \
  "$RAW_CSV" "$SUMMARY_CSV" "$REPORT" "$FIGURE_TEX"
