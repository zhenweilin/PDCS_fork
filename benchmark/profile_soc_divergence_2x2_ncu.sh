#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_DIR=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
COUNT=1048576
DIMENSION=10
SEEDS="2026,2027,2028"
EXPERIMENTS="iteration,branch"
DRY_RUN=0

usage() {
  printf '%s\n' \
    'Usage: profile_soc_divergence_2x2_ncu.sh --run-dir PATH [options]' \
    '  --gpu N --cuda-home PATH --julia PATH --julia-depot PATH' \
    '  --cone-count N --cone-dimension N --seeds LIST --experiments LIST --dry-run'
}
while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --cone-count) COUNT="$2"; shift 2 ;;
    --cone-dimension) DIMENSION="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" ]] || { usage >&2; exit 2; }
RUN_DIR="$(cd -- "$RUN_DIR" 2>/dev/null && pwd || printf '%s' "$RUN_DIR")"
NCU="$CUDA_ROOT/bin/ncu"
[[ -x "$NCU" ]] || { printf 'ncu not found: %s\n' "$NCU" >&2; exit 1; }
IFS=, read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=, read -r -a EXP_ARRAY <<< "$EXPERIMENTS"

if ((!DRY_RUN)); then
  mkdir -p "$RUN_DIR/ncu"
  "$NCU" --query-metrics > "$RUN_DIR/ncu/available_metrics.txt"
fi

for experiment in "${EXP_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    cache="$RUN_DIR/$experiment/seed$seed/case_cache.jls"
    [[ -f "$cache" || $DRY_RUN -eq 1 ]] ||
      { printf 'Missing case cache: %s\n' "$cache" >&2; exit 1; }
    for layout in grouped interleaved; do
      for strategy in threadWise warpWise; do
        kernel=massive_block_proj
        [[ "$strategy" == warpWise ]] && kernel=sufficient_block_proj
        stem="${experiment}_${layout}_${strategy}_seed${seed}"
        out="$RUN_DIR/ncu/$stem"
        [[ ! -e "$out.ncu-rep" ]] ||
          { printf 'Refusing to overwrite %s.ncu-rep\n' "$out" >&2; exit 1; }
        cmd=(env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH"
          CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
          JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1
          JULIA_CONDAPKG_OFFLINE=true
          "$NCU" --kernel-name "regex:${kernel}" --launch-count 1
          --section SpeedOfLight --section Occupancy --section SchedulerStats
          --section WarpStateStats --section SourceCounters
          --export "$out"
          "$JULIA_BIN" "--project=$REPO_ROOT"
          "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
          --experiment "$experiment" --mode profile-one --layout "$layout"
          --strategy "$strategy" --seed "$seed" --cone-count "$COUNT"
          --cone-dimension "$DIMENSION" --case-cache "$cache"
          --output-dir "$RUN_DIR/$experiment/seed$seed")
        printf 'COMMAND:'; printf ' %q' "${cmd[@]}"; printf '\n'
        ((DRY_RUN)) && continue
        set +e
        "${cmd[@]}" > "$out.application.log" 2> "$out.ncu.log"
        status=$?
        set -e
        printf '%s\n' "$status" > "$out.exit_status.txt"
        if grep -Eqi 'ERR_NVGPUCTRPERM|permission' "$out.ncu.log"; then
          cp "$out.ncu.log" "$out.PROFILE_INCOMPLETE.txt"
        fi
        ((status==0)) || continue
        "$NCU" --import "$out.ncu-rep" --page raw --csv \
          > "$out.csv" 2> "$out.import.log"
      done
    done
  done
done
