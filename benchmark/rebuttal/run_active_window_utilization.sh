#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUN_DIR=""
CACHE=""
EXPERIMENT=iteration
LAYOUT=grouped
STRATEGY=threadWise
SEED=2026
COUNT=1048576
DIMENSION=10
DURATION=35
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
JULIA_OPT_LEVEL=1
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: run_active_window_utilization.sh --run-dir PATH --case-cache PATH [options]' \
    '  --experiment iteration|branch|parametric --layout grouped|interleaved' \
    '  --strategy threadWise|warpWise --seed N --duration 35 --gpu N' \
    '  --cone-count N --cone-dimension N --julia-opt-level 0|1|2|3 --dry-run'
}
while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --case-cache) CACHE="$2"; shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --layout) LAYOUT="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --julia-opt-level) JULIA_OPT_LEVEL="$2"; shift 2 ;;
    --cone-count) COUNT="$2"; shift 2 ;;
    --cone-dimension) DIMENSION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" && -n "$CACHE" ]] || { usage >&2; exit 2; }
case "$EXPERIMENT" in iteration|branch|parametric) ;; *) exit 2 ;; esac
case "$LAYOUT" in grouped|interleaved) ;; *) exit 2 ;; esac
case "$STRATEGY" in threadWise|warpWise) ;; *) exit 2 ;; esac
[[ "$JULIA_OPT_LEVEL" =~ ^[0-3]$ ]] || exit 2
[[ -f "$CACHE" || $DRY_RUN -eq 1 ]] || { printf 'Missing cache: %s\n' "$CACHE" >&2; exit 1; }

stem="${EXPERIMENT}_${LAYOUT}_${STRATEGY}_seed${SEED}"
out="$RUN_DIR/$stem"
cmd=(env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH"
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
  JULIA_CONDAPKG_BACKEND=Null JULIA_DEPOT_PATH="$JULIA_DEPOT"
  PDCS_SKIP_GPU_PRECOMPILE=1
  "$JULIA_BIN" "-O$JULIA_OPT_LEVEL" "--project=$REPO_ROOT"
  "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
  --experiment "$EXPERIMENT" --mode duration --duration "$DURATION"
  --wait-for-start --layout "$LAYOUT" --strategy "$STRATEGY" --seed "$SEED"
  --cone-count "$COUNT" --cone-dimension "$DIMENSION"
  --case-cache "$CACHE" --output-dir "$(dirname "$CACHE")")
printf 'COMMAND:'; printf ' %q' "${cmd[@]}"; printf '\n'
((DRY_RUN)) && exit 0

mkdir -p "$RUN_DIR"
[[ ! -e "$out.application.log" ]] || { printf 'Refusing to overwrite %s\n' "$out.application.log" >&2; exit 1; }
temp_dir="$(mktemp -d "/tmp/pdcs_active_window_${SEED}_XXXXXX")"
fifo="$temp_dir/control.fifo"
mkfifo "$fifo"
exec 3<>"$fifo"
APP_PID= SMI_PID=
cleanup() {
  [[ -z "${SMI_PID:-}" ]] || { kill "$SMI_PID" 2>/dev/null || true; wait "$SMI_PID" 2>/dev/null || true; }
  [[ -z "${APP_PID:-}" ]] || { kill "$APP_PID" 2>/dev/null || true; wait "$APP_PID" 2>/dev/null || true; }
  exec 3>&- 3<&-
  rm -f -- "$fifo"
  rmdir -- "$temp_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${cmd[@]}" <&3 > "$out.application.log" 2> "$out.application.err" &
APP_PID=$!
for _ in {1..1800}; do
  grep -q '^READY ' "$out.application.log" 2>/dev/null && break
  kill -0 "$APP_PID" 2>/dev/null || { wait "$APP_PID"; exit $?; }
  sleep 0.1
done
grep -q '^READY ' "$out.application.log" ||
  { printf '%s\n' 'Timed out waiting for READY' >&2; exit 1; }

nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv,noheader,nounits --loop-ms=1000 \
  > "$out.gpu.csv" 2> "$out.gpu.log" &
SMI_PID=$!
printf '%s\n' START >&3
set +e
wait "$APP_PID"
status=$?
set -e
APP_PID=
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true
SMI_PID=
printf '%s\n' "$status" > "$out.exit_status.txt"
grep -q '^DONE ' "$out.application.log" || status=1
trap - EXIT INT TERM
cleanup
exit "$status"
