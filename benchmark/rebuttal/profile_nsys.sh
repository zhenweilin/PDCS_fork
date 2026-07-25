#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

KIND="soc"
CASE_NAME="similar"
STRATEGY="threadWise"
SEED=2026
GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU_METRICS="auto"
GPU_METRICS_FREQUENCY=10000
DURATION=30
CONE_COUNT=1048576
CONE_DIMENSION=10
HETERO_SIGMA=2.0
INPUT_SIGMA=1.0
DIAGONAL_SIGMA=1.0
WARM_START=0
SAMPLE_MS=200
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
OUTPUT_ROOT="$REPO_ROOT/benchmark/results/rebuttal/nsys"
RUN_ID=""
DRY_RUN=0

usage() {
  sed -n '4,55p' "$0" | sed -n 's/^# //p'
}

# Collect one sustained Nsight Systems profile without overwriting prior data.
#
# Usage:
#   benchmark/rebuttal/profile_nsys.sh [options]
#
# Options:
#   --kind soc|exp
#   --case NAME
#   --strategy threadWise|warpWise|blockWise
#   --seed N
#   --gpu PHYSICAL_INDEX
#   --gpu-metrics auto|required|off
#   --gpu-metrics-frequency HZ
#   --duration SECONDS                 default: 30
#   --cone-count N                     default: 1048576
#   --cone-dimension N                 SOC only; default: 10
#   --hetero-sigma FLOAT               SOC only; default: 2.0
#   --sigma FLOAT                      exponential input sigma
#   --diagonal-sigma FLOAT             exponential diagonal sigma
#   --warm-start                       SOC only
#   --sample-ms N                      nvidia-smi interval; default: 200
#   --cuda-home PATH
#   --julia PATH
#   --julia-depot PATH
#   --output-root PATH
#   --run-id NAME                      must not already exist
#   --dry-run                          validate and print commands only
#   --help

while (($#)); do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --case) CASE_NAME="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --gpu-metrics) GPU_METRICS="$2"; shift 2 ;;
    --gpu-metrics-frequency) GPU_METRICS_FREQUENCY="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --cone-count) CONE_COUNT="$2"; shift 2 ;;
    --cone-dimension) CONE_DIMENSION="$2"; shift 2 ;;
    --hetero-sigma) HETERO_SIGMA="$2"; shift 2 ;;
    --sigma) INPUT_SIGMA="$2"; shift 2 ;;
    --diagonal-sigma) DIAGONAL_SIGMA="$2"; shift 2 ;;
    --warm-start) WARM_START=1; shift ;;
    --sample-ms) SAMPLE_MS="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$KIND" == soc || "$KIND" == exp ]] || { printf 'Expected --kind soc or exp.\n' >&2; exit 2; }
case "$STRATEGY" in threadWise|warpWise|blockWise) ;; *) printf 'Invalid strategy: %s\n' "$STRATEGY" >&2; exit 2 ;; esac
case "$GPU_METRICS" in auto|required|off) ;; *) printf 'Invalid --gpu-metrics mode.\n' >&2; exit 2 ;; esac
[[ "$CONE_COUNT" =~ ^[0-9]+$ && "$CONE_COUNT" -ge 32 ]] || { printf 'Cone count must be at least 32.\n' >&2; exit 2; }
((CONE_COUNT % 32 == 0)) || { printf 'Cone count must be divisible by 32.\n' >&2; exit 2; }
[[ "$CONE_DIMENSION" =~ ^[0-9]+$ && "$CONE_DIMENSION" -ge 3 ]] || { printf 'SOC dimension must be at least 3.\n' >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { printf 'Seed must be a nonnegative integer.\n' >&2; exit 2; }
[[ "$SAMPLE_MS" =~ ^[0-9]+$ && "$SAMPLE_MS" -ge 100 ]] || { printf 'Sample interval must be at least 100 ms.\n' >&2; exit 2; }

case "$KIND:$CASE_NAME" in
  soc:uniform|soc:similar|soc:heterogeneous|soc:mixed_grouped|soc:mixed_random|soc:mixed_interleaved) ;;
  exp:similar|exp:heterogeneous|exp:mixed_grouped|exp:mixed_random|exp:mixed_interleaved) ;;
  *) printf 'Unsupported case %s for %s.\n' "$CASE_NAME" "$KIND" >&2; exit 2 ;;
esac
if ((WARM_START)) && [[ "$KIND" != soc ]]; then
  printf '%s\n' '--warm-start is supported only for SOC.' >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${KIND}_${CASE_NAME}_${STRATEGY}_seed${SEED}}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { printf 'Refusing to overwrite existing run: %s\n' "$RUN_DIR" >&2; exit 1; }

NSYS="$CUDA_ROOT/bin/nsys"
[[ -x "$NSYS" ]] || NSYS="$(command -v nsys || true)"
[[ -x "$NSYS" ]] || { printf 'Nsight Systems (nsys) was not found.\n' >&2; exit 1; }
[[ -x "$JULIA_BIN" ]] || { printf 'Julia is not executable: %s\n' "$JULIA_BIN" >&2; exit 1; }

if [[ "$KIND" == soc ]]; then
  HARNESS="$REPO_ROOT/benchmark/rescaled_soc_warp_profile.jl"
  HARNESS_ARGS=(--case "$CASE_NAME" --strategy "$STRATEGY" --cone-count "$CONE_COUNT"
    --cone-dimension "$CONE_DIMENSION" --hetero-sigma "$HETERO_SIGMA"
    --seed "$SEED" --duration "$DURATION")
  ((WARM_START)) && HARNESS_ARGS+=(--warm-start)
else
  HARNESS="$REPO_ROOT/benchmark/exp_warp_profile.jl"
  HARNESS_ARGS=(--case "$CASE_NAME" --strategy "$STRATEGY" --cone-count "$CONE_COUNT"
    --sigma "$INPUT_SIGMA" --diagonal-sigma "$DIAGONAL_SIGMA"
    --seed "$SEED" --duration "$DURATION")
fi

NSYS_ARGS=(profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
  --cuda-memory-usage=true --force-overwrite=false --show-output=true
  --wait=all --output "$RUN_DIR/trace")
if [[ "$GPU_METRICS" != off ]]; then
  NSYS_ARGS+=(--gpu-metrics-devices=cuda-visible
    --gpu-metrics-frequency "$GPU_METRICS_FREQUENCY")
fi

printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Command: env CUDA_VISIBLE_DEVICES=%q JULIA_DEPOT_PATH=%q PDCS_SKIP_GPU_PRECOMPILE=1 %q' \
  "$GPU" "$JULIA_DEPOT" "$NSYS"
printf ' %q' "${NSYS_ARGS[@]}" "$JULIA_BIN" "--project=$REPO_ROOT" "$HARNESS" "${HARNESS_ARGS[@]}"
printf '\n'
((DRY_RUN)) && exit 0

mkdir -p "$RUN_DIR"
NSYS_VERSION="$("$NSYS" --version | head -1)"
{
  printf 'run_id=%s\nkind=%s\ncase=%s\nstrategy=%s\nseed=%s\n' \
    "$RUN_ID" "$KIND" "$CASE_NAME" "$STRATEGY" "$SEED"
  printf 'duration_seconds=%s\ncone_count=%s\ncone_dimension=%s\n' \
    "$DURATION" "$CONE_COUNT" "$([[ "$KIND" == soc ]] && printf '%s' "$CONE_DIMENSION" || printf '3')"
  printf 'warm_start=%s\nhetero_sigma=%s\ninput_sigma=%s\ndiagonal_sigma=%s\n' \
    "$WARM_START" "$HETERO_SIGMA" "$INPUT_SIGMA" "$DIAGONAL_SIGMA"
  printf 'gpu_physical_index=%s\ngpu_metrics_mode=%s\ngpu_metrics_frequency_hz=%s\n' \
    "$GPU" "$GPU_METRICS" "$GPU_METRICS_FREQUENCY"
  printf 'nsys=%s\njulia=%s\ncuda_home=%s\ngit_commit=%s\n' \
    "$NSYS_VERSION" "$("$JULIA_BIN" --version)" "$CUDA_ROOT" \
    "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
  nvidia-smi -i "$GPU" --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,memory.free,power.limit,persistence_mode,compute_mode,mig.mode.current --format=csv,noheader 2>&1 || true
} > "$RUN_DIR/environment.txt"

SMI_PID=""
cleanup() {
  if [[ -n "$SMI_PID" ]] && kill -0 "$SMI_PID" 2>/dev/null; then
    kill "$SMI_PID" 2>/dev/null || true
    wait "$SMI_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv,noheader,nounits --loop-ms="$SAMPLE_MS" \
  > "$RUN_DIR/nvidia_smi.csv" 2> "$RUN_DIR/nvidia_smi.log" &
SMI_PID=$!

set +e
env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH" \
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
  JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1 \
  "$NSYS" "${NSYS_ARGS[@]}" \
  "$JULIA_BIN" "--project=$REPO_ROOT" "$HARNESS" "${HARNESS_ARGS[@]}" \
  > "$RUN_DIR/application.log" 2> "$RUN_DIR/nsys.log"
STATUS=$?
set -e
cleanup
SMI_PID=""
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"

if grep -Eqi 'permission|GPU metrics|not supported|not available|ERR_NVGPUCTRPERM' "$RUN_DIR/nsys.log"; then
  printf '%s\n' 'GPU metrics were unavailable. The failure is preserved; rerun with --gpu-metrics off only for trace debugging, or use an authorized machine.' \
    > "$RUN_DIR/PROFILE_INCOMPLETE.txt"
fi

REPORT="$RUN_DIR/trace.nsys-rep"
if [[ -f "$REPORT" ]]; then
  "$NSYS" export --type sqlite --force-overwrite=false \
    --output "$RUN_DIR/trace.sqlite" "$REPORT" > "$RUN_DIR/export.log" 2>&1
  "$NSYS" stats --force-export=false --force-overwrite=true --format csv \
    --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_kern_exec_sum \
    --output "$RUN_DIR/stats" "$REPORT" > "$RUN_DIR/stats.log" 2>&1
fi

printf 'status=%s\nreport=%s\n' "$STATUS" "$REPORT"
exit "$STATUS"
