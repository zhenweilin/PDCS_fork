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
SEEDS="2026:2035"
DELTAS="0,0.0001,0.001,0.01"
SMOKE=0
DRY_RUN=0
KEEP_CACHES=0
SAVE_LARGE_INTERMEDIATES=0
RETAIN_CACHE_SEEDS="2026,2027,2028"

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
    --deltas) DELTAS="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --keep-caches) KEEP_CACHES=1; shift ;;
    --save-large-intermediates) SAVE_LARGE_INTERMEDIATES=1; shift ;;
    --retain-cache-seeds) RETAIN_CACHE_SEEDS="$2"; shift 2 ;;
    --help|-h)
      printf '%s\n' \
        'Usage: run_soc_parametric_similarity.sh --run-dir PATH [options]' \
        '  --gpu N --seeds A:B --deltas LIST --retain-cache-seeds LIST' \
        '  --keep-caches --save-large-intermediates --smoke --dry-run'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" ]] || { printf '%s\n' '--run-dir is required' >&2; exit 2; }
if ((SMOKE)); then COUNT=1024; SEEDS=2026; DELTAS="0,0.001"; fi
if [[ "$SEEDS" == *:* ]]; then
  first="${SEEDS%%:*}"; last="${SEEDS##*:}"
  mapfile -t SEED_ARRAY < <(seq "$first" "$last")
else
  IFS=, read -r -a SEED_ARRAY <<< "$SEEDS"
fi
IFS=, read -r -a DELTA_ARRAY <<< "$DELTAS"
BASE_ENV=(env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH"
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
  JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1)

for delta in "${DELTA_ARRAY[@]}"; do
  label="${delta//./p}"
  for seed in "${SEED_ARRAY[@]}"; do
    out="$RUN_DIR/parametric/delta_${label}/seed${seed}"
    cache="$out/case_cache.jls"
    generate=("${BASE_ENV[@]}" "$JULIA_BIN" "--project=$REPO_ROOT"
      "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
      --experiment parametric --mode generate --delta "$delta" --seed "$seed"
      --cone-count "$COUNT" --cone-dimension "$DIMENSION" --output-dir "$out")
    ((SAVE_LARGE_INTERMEDIATES)) && generate+=(--save-large-intermediates)
    timing=("${BASE_ENV[@]}" "$JULIA_BIN" "--project=$REPO_ROOT"
      "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
      --experiment parametric --mode timing --delta "$delta" --seed "$seed"
      --cone-count "$COUNT" --cone-dimension "$DIMENSION"
      --case-cache "$cache" --warmups 5 --rounds 10 --output-dir "$out")
    if ((DRY_RUN)); then
      printf 'COMMAND:'; printf ' %q' "${generate[@]}"; printf '\n'
      printf 'COMMAND:'; printf ' %q' "${timing[@]}"; printf '\n'
    else
      [[ ! -e "$out" ]] || { printf 'Refusing to overwrite %s\n' "$out" >&2; exit 1; }
      mkdir -p "$out"
      "${generate[@]}" > "$out/generate.log" 2>&1
      "${timing[@]}" > "$out/timing.log" 2>&1
      if ((!KEEP_CACHES)) && [[ ",$RETAIN_CACHE_SEEDS," != *",$seed,"* ]]; then
        rm -f -- "$cache"
        printf 'Removed intermediate cache after successful timing: %s\n' "$cache"
      fi
    fi
  done
done
((DRY_RUN)) && exit 0

if ((SAVE_LARGE_INTERMEDIATES)) && command -v zstd >/dev/null 2>&1; then
  while IFS= read -r file; do
    zstd -q --rm -- "$file"
  done < <(find "$RUN_DIR/parametric" -type f \
    \( -name root_work_raw.csv -o -name permutations.csv \))
fi

seed_csv="$(IFS=,; printf '%s' "${SEED_ARRAY[*]}")"
"$JULIA_BIN" --project="$REPO_ROOT" "$REPO_ROOT/benchmark/analyze_soc_parametric_similarity.jl" \
  --root "$RUN_DIR/parametric" --seeds "$seed_csv" \
  --output "$RUN_DIR/parametric/parametric_summary.csv"
printf 'PARAMETRIC_COMPLETE: %s/parametric\n' "$RUN_DIR"
