#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_DIR=""
CACHE=""
SEED=2026
DELTA=0
STRATEGY=threadWise
COUNT=1048576
DIMENSION=10
DURATION=35
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: run_soc_parametric_similarity_utilization.sh [options]' \
    '  --run-dir PATH --case-cache PATH --seed N --delta 0|0.01' \
    '  --strategy threadWise|warpWise --duration 35 --gpu N' \
    '  --cone-count N --cone-dimension N --cuda-home PATH' \
    '  --julia PATH --julia-depot PATH --dry-run'
}
while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --case-cache) CACHE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --delta) DELTA="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --cone-count) COUNT="$2"; shift 2 ;;
    --cone-dimension) DIMENSION="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" && -n "$CACHE" ]] || { usage >&2; exit 2; }
case "$DELTA" in 0|0.0|0.01|1e-2|1E-2) ;; *) printf '%s\n' 'Endpoint delta must be 0 or 0.01.' >&2; exit 2 ;; esac
case "$STRATEGY" in threadWise|warpWise) ;; *) exit 2 ;; esac
[[ -f "$CACHE" || $DRY_RUN -eq 1 ]] || { printf 'Missing cache: %s\n' "$CACHE" >&2; exit 1; }

label="${DELTA//./p}"
stem="seed${SEED}_delta${label}_${STRATEGY}"
out="$RUN_DIR/$stem"
cmd=(env CUDA_VISIBLE_DEVICES="$GPU" PDCS_GPU_PHYSICAL="$GPU"
  PATH="$CUDA_ROOT/bin:$PATH" CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
  JULIA_DEPOT_PATH="$JULIA_DEPOT"
  PDCS_SKIP_GPU_PRECOMPILE=1
  "$JULIA_BIN" -O1 "--project=$REPO_ROOT"
  "$REPO_ROOT/benchmark/rescaled_soc_parametric_similar.jl"
  --mode duration --wait-for-start --seed "$SEED" --delta "$DELTA"
  --strategy "$STRATEGY" --duration "$DURATION" --cone-count "$COUNT"
  --cone-dimension "$DIMENSION" --case-cache "$CACHE"
  --output-dir "$(dirname "$CACHE")")
printf 'COMMAND:'; printf ' %q' "${cmd[@]}"; printf '\n'
((DRY_RUN)) && exit 0

mkdir -p "$RUN_DIR"
[[ ! -e "$out.application.log" ]] ||
  { printf 'Refusing to overwrite %s\n' "$out.application.log" >&2; exit 1; }
tmp="$(mktemp -d "/tmp/pdcs_parametric_util_${SEED}_XXXXXX")"
fifo="$tmp/control.fifo"
mkfifo "$fifo"
exec 3<>"$fifo"
APP_PID= SMI_PID=
cleanup() {
  [[ -z "${SMI_PID:-}" ]] || { kill "$SMI_PID" 2>/dev/null || true; wait "$SMI_PID" 2>/dev/null || true; }
  [[ -z "${APP_PID:-}" ]] || { kill "$APP_PID" 2>/dev/null || true; wait "$APP_PID" 2>/dev/null || true; }
  exec 3>&- 3<&-
  rm -f -- "$fifo"
  rmdir -- "$tmp" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${cmd[@]}" <&3 >"$out.application.log" 2>"$out.application.err" &
APP_PID=$!
for _ in {1..1800}; do
  grep -q '^READY ' "$out.application.log" 2>/dev/null && break
  kill -0 "$APP_PID" 2>/dev/null || { wait "$APP_PID"; exit $?; }
  sleep 0.1
done
grep -q '^READY ' "$out.application.log" ||
  { printf '%s\n' 'Timed out waiting for READY.' >&2; exit 1; }

nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv,noheader,nounits --loop-ms=1000 \
  >"$out.gpu.csv" 2>"$out.gpu.log" &
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
printf '%s\n' "$status" >"$out.exit_status.txt"
grep -q '^DONE ' "$out.application.log" || status=1
trap - EXIT INT TERM
cleanup
exit "$status"
