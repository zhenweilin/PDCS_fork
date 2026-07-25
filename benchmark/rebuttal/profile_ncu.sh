#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
KIND="soc"
CASE_NAME="similar"
STRATEGY="threadWise"
SEED=2026
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CONE_COUNT=1048576
CONE_DIMENSION=10
HETERO_SIGMA=2.0
INPUT_SIGMA=1.0
DIAGONAL_SIGMA=1.0
WARM_START=0
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
OUTPUT_ROOT="$REPO_ROOT/benchmark/results/rebuttal/ncu"
RUN_ID=""
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: profile_ncu.sh [options]' \
    '  --kind soc|exp --case NAME --strategy threadWise|warpWise|blockWise' \
    '  --seed N --gpu N --cone-count N --cone-dimension N' \
    '  --hetero-sigma X --sigma X --diagonal-sigma X --warm-start' \
    '  --cuda-home PATH --julia PATH --julia-depot PATH' \
    '  --output-root PATH --run-id NAME --dry-run'
}

while (($#)); do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --case) CASE_NAME="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --cone-count) CONE_COUNT="$2"; shift 2 ;;
    --cone-dimension) CONE_DIMENSION="$2"; shift 2 ;;
    --hetero-sigma) HETERO_SIGMA="$2"; shift 2 ;;
    --sigma) INPUT_SIGMA="$2"; shift 2 ;;
    --diagonal-sigma) DIAGONAL_SIGMA="$2"; shift 2 ;;
    --warm-start) WARM_START=1; shift ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ "$KIND" == soc || "$KIND" == exp ]] || { printf 'Expected --kind soc or exp.\n' >&2; exit 2; }
case "$STRATEGY" in
  threadWise) KERNEL=massive_block_proj ;;
  warpWise) KERNEL=sufficient_block_proj ;;
  blockWise) KERNEL=moderate_block_proj ;;
  *) printf 'Invalid strategy: %s\n' "$STRATEGY" >&2; exit 2 ;;
esac
if ((WARM_START)) && [[ "$KIND" != soc ]]; then
  printf '%s\n' '--warm-start is supported only for SOC.' >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${KIND}_${CASE_NAME}_${STRATEGY}_seed${SEED}}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { printf 'Refusing to overwrite existing run: %s\n' "$RUN_DIR" >&2; exit 1; }
NCU="$CUDA_ROOT/bin/ncu"
[[ -x "$NCU" ]] || NCU="$(command -v ncu || true)"
[[ -x "$NCU" ]] || { printf 'Nsight Compute (ncu) was not found.\n' >&2; exit 1; }

if [[ "$KIND" == soc ]]; then
  HARNESS="$REPO_ROOT/benchmark/rescaled_soc_warp_profile.jl"
  HARNESS_ARGS=(--profile-one --case "$CASE_NAME" --strategy "$STRATEGY"
    --cone-count "$CONE_COUNT" --cone-dimension "$CONE_DIMENSION"
    --hetero-sigma "$HETERO_SIGMA" --seed "$SEED")
  ((WARM_START)) && HARNESS_ARGS+=(--warm-start)
else
  HARNESS="$REPO_ROOT/benchmark/exp_warp_profile.jl"
  HARNESS_ARGS=(--profile-one --case "$CASE_NAME" --strategy "$STRATEGY"
    --cone-count "$CONE_COUNT" --sigma "$INPUT_SIGMA"
    --diagonal-sigma "$DIAGONAL_SIGMA" --seed "$SEED")
fi

NCU_ARGS=(--target-processes all --kernel-name "regex:$KERNEL"
  --launch-skip 1 --launch-count 1
  --section SpeedOfLight --section Occupancy --section SchedulerStats
  --section WarpStateStats --section SourceCounters
  --force-overwrite=false --export "$RUN_DIR/profile")

printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Command: env CUDA_VISIBLE_DEVICES=%q JULIA_DEPOT_PATH=%q PDCS_SKIP_GPU_PRECOMPILE=1 %q' \
  "$GPU" "$JULIA_DEPOT" "$NCU"
printf ' %q' "${NCU_ARGS[@]}" "$JULIA_BIN" "--project=$REPO_ROOT" "$HARNESS" "${HARNESS_ARGS[@]}"
printf '\n'
((DRY_RUN)) && exit 0

mkdir -p "$RUN_DIR"
{
  printf 'run_id=%s\nkind=%s\ncase=%s\nstrategy=%s\nseed=%s\nkernel=%s\n' \
    "$RUN_ID" "$KIND" "$CASE_NAME" "$STRATEGY" "$SEED" "$KERNEL"
  printf 'cone_count=%s\ncone_dimension=%s\nwarm_start=%s\n' \
    "$CONE_COUNT" "$([[ "$KIND" == soc ]] && printf '%s' "$CONE_DIMENSION" || printf '3')" "$WARM_START"
  printf 'ncu=%s\ngit_commit=%s\n' "$("$NCU" --version | head -1)" \
    "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
  nvidia-smi -i "$GPU" --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,memory.free --format=csv,noheader 2>&1 || true
} > "$RUN_DIR/environment.txt"

set +e
env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH" \
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
  JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1 \
  "$NCU" "${NCU_ARGS[@]}" \
  "$JULIA_BIN" "--project=$REPO_ROOT" "$HARNESS" "${HARNESS_ARGS[@]}" \
  > "$RUN_DIR/application.log" 2> "$RUN_DIR/ncu.log"
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
if grep -Eqi 'ERR_NVGPUCTRPERM|permission' "$RUN_DIR/ncu.log"; then
  cp "$RUN_DIR/ncu.log" "$RUN_DIR/PROFILE_INCOMPLETE.txt"
fi
if [[ -f "$RUN_DIR/profile.ncu-rep" ]]; then
  "$NCU" --import "$RUN_DIR/profile.ncu-rep" --page raw --csv \
    > "$RUN_DIR/profile_raw.csv" 2> "$RUN_DIR/import.log"
fi
exit "$STATUS"
