#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
ARCH="${PDCS_GPU_ARCH:-sm_90}"
COUNT=1048576
DIMENSION=10
SEEDS="2026:2035"
EXPERIMENTS="iteration,branch"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="$REPO_ROOT/benchmark/results/rebuttal/soc_divergence_2x2"
SMOKE=0
DRY_RUN=0
NO_BUILD=0
KEEP_CACHES=0
SAVE_LARGE_INTERMEDIATES=0
RETAIN_CACHE_SEEDS="2026,2027,2028"

usage() {
  printf '%s\n' \
    'Usage: run_soc_divergence_2x2.sh [options]' \
    '  --julia PATH --julia-depot PATH --cuda-home PATH --gpu N --arch sm_XX' \
    '  --cone-count N --cone-dimension N --seeds A:B|A,B --experiments LIST' \
    '  --output-root PATH --run-id NAME --retain-cache-seeds LIST' \
    '  --keep-caches --save-large-intermediates --smoke --dry-run --no-build'
}

while (($#)); do
  case "$1" in
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --cone-count) COUNT="$2"; shift 2 ;;
    --cone-dimension) DIMENSION="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-build) NO_BUILD=1; shift ;;
    --keep-caches) KEEP_CACHES=1; shift ;;
    --save-large-intermediates) SAVE_LARGE_INTERMEDIATES=1; shift ;;
    --retain-cache-seeds) RETAIN_CACHE_SEEDS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if ((SMOKE)); then
  COUNT=1024
  SEEDS=2026
fi
((COUNT % 128 == 0)) || { printf '%s\n' 'Cone count must be divisible by 128.' >&2; exit 2; }
[[ "$ARCH" =~ ^sm_[0-9]+$ ]] || { printf 'Invalid architecture: %s\n' "$ARCH" >&2; exit 2; }
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { printf 'Refusing to overwrite %s\n' "$RUN_DIR" >&2; exit 1; }

if [[ "$SEEDS" == *:* ]]; then
  first="${SEEDS%%:*}"; last="${SEEDS##*:}"
  mapfile -t SEED_ARRAY < <(seq "$first" "$last")
else
  IFS=, read -r -a SEED_ARRAY <<< "$SEEDS"
fi
IFS=, read -r -a EXP_ARRAY <<< "$EXPERIMENTS"

BASE_ENV=(env CUDA_VISIBLE_DEVICES="$GPU" PATH="$CUDA_ROOT/bin:$PATH"
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
  JULIA_DEPOT_PATH="$JULIA_DEPOT" PDCS_SKIP_GPU_PRECOMPILE=1)

printf 'Run directory: %s\n' "$RUN_DIR"
if ((!NO_BUILD)); then
  if ((DRY_RUN)); then
    printf 'COMMAND: make -C %q rebuild-gpu CUDA_HOME=%q ARCH=%q\n' \
      "$REPO_ROOT/src/pdcs_gpu/cuda" "$CUDA_ROOT" "$ARCH"
    printf 'COMMAND: make -C %q rebuild-profile CUDA_HOME=%q ARCH=%q\n' \
      "$REPO_ROOT/src/pdcs_gpu/cuda" "$CUDA_ROOT" "$ARCH"
  else
    make -C "$REPO_ROOT/src/pdcs_gpu/cuda" rebuild-gpu CUDA_HOME="$CUDA_ROOT" ARCH="$ARCH"
    make -C "$REPO_ROOT/src/pdcs_gpu/cuda" rebuild-profile CUDA_HOME="$CUDA_ROOT" ARCH="$ARCH"
  fi
fi
if ((!DRY_RUN)); then
  mkdir -p "$RUN_DIR"
  {
    printf 'run_id=%s\ngit_commit=%s\ngit_dirty=%s\n' "$RUN_ID" \
      "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)" \
      "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l)"
    printf 'gpu_physical_index=%s\ncuda_home=%s\narch=%s\ncone_count=%s\ncone_dimension=%s\nseeds=%s\nexperiments=%s\n' \
      "$GPU" "$CUDA_ROOT" "$ARCH" "$COUNT" "$DIMENSION" "$SEEDS" "$EXPERIMENTS"
    nvidia-smi -i "$GPU" --query-gpu=name,uuid,pci.bus_id,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,compute_mode,memory.total,memory.free --format=csv,noheader
    "$CUDA_ROOT/bin/nvcc" --version
    "$JULIA_BIN" --version
  } > "$RUN_DIR/environment.txt"
fi
for experiment in "${EXP_ARRAY[@]}"; do
  for seed in "${SEED_ARRAY[@]}"; do
    seed_dir="$RUN_DIR/$experiment/seed$seed"
    cache="$seed_dir/case_cache.jls"
    generate=("${BASE_ENV[@]}" "$JULIA_BIN" "--project=$REPO_ROOT"
      "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
      --experiment "$experiment" --mode generate --seed "$seed"
      --cone-count "$COUNT" --cone-dimension "$DIMENSION"
      --candidate-factor 4 --output-dir "$seed_dir")
    ((SAVE_LARGE_INTERMEDIATES)) && generate+=(--save-large-intermediates)
    timing=("${BASE_ENV[@]}" "$JULIA_BIN" "--project=$REPO_ROOT"
      "$REPO_ROOT/benchmark/rescaled_soc_divergence_2x2.jl"
      --experiment "$experiment" --mode timing --seed "$seed"
      --cone-count "$COUNT" --cone-dimension "$DIMENSION"
      --case-cache "$cache" --warmups 5 --rounds 10 --output-dir "$seed_dir")
    if ((DRY_RUN)); then
      printf 'COMMAND:'; printf ' %q' "${generate[@]}"; printf '\n'
      printf 'COMMAND:'; printf ' %q' "${timing[@]}"; printf '\n'
    else
      mkdir -p "$seed_dir"
      "${generate[@]}" > "$seed_dir/generate.log" 2>&1
      "${timing[@]}" > "$seed_dir/timing.log" 2>&1
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
  done < <(find "$RUN_DIR" -type f \( -name root_work_raw.csv -o -name permutations.csv \))
fi

seed_csv="$(IFS=,; printf '%s' "${SEED_ARRAY[*]}")"
"$JULIA_BIN" --project="$REPO_ROOT" "$REPO_ROOT/benchmark/analyze_soc_divergence_2x2.jl" \
  --root "$RUN_DIR" --seeds "$seed_csv" --bootstrap 10000 --output "$RUN_DIR"
printf 'COMPLETE: %s\n' "$RUN_DIR"
