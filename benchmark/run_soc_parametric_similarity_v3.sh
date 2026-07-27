#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PHASE=all
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_soc_parametric_similarity"
OUTPUT_ROOT="$REPO_ROOT/benchmark/results/rebuttal/soc_parametric_similarity"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-12.6}"
ARCH="${PDCS_GPU_ARCH:-sm_90}"
JULIA_BIN="${PDCS_JULIA:-$REPO_ROOT/.julia-bin/julia}"
JULIA_DEPOT="${JULIA_DEPOT_PATH:-$REPO_ROOT/.julia-depot}"
SEEDS="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035"
COUNT=1048576
DIMENSION=10
DURATION=35
SMOKE=0
DRY_RUN=0
SAVE_PER_CONE=1

usage() {
  printf '%s\n' \
    'Usage: run_soc_parametric_similarity_v3.sh [options]' \
    '  --phase snapshot|build|test|pilot|generate|timing|utilization|analysis|all' \
    '  --run-id ID --output-root PATH --gpu N --cuda-home PATH --arch sm_XX' \
    '  --julia PATH --julia-depot PATH --smoke --dry-run --no-per-cone'
}
while (($#)); do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --cuda-home) CUDA_ROOT="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --julia) JULIA_BIN="$2"; shift 2 ;;
    --julia-depot) JULIA_DEPOT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-per-cone) SAVE_PER_CONE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in snapshot|build|test|pilot|generate|timing|utilization|analysis|all) ;; *) usage >&2; exit 2 ;; esac
if ((SMOKE)); then
  SEEDS=2026
  COUNT=1024
  DURATION=5
fi
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
IFS=, read -r -a SEED_ARRAY <<<"$SEEDS"

run() {
  printf 'COMMAND:'; printf ' %q' "$@"; printf '\n'
  ((DRY_RUN)) || "$@"
}
has_phase() { [[ "$PHASE" == all || "$PHASE" == "$1" ]]; }
common_env=(env CUDA_VISIBLE_DEVICES="$GPU" PDCS_GPU_PHYSICAL="$GPU"
  CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT"
  JULIA_DEPOT_PATH="$JULIA_DEPOT"
  PDCS_SKIP_GPU_PRECOMPILE=1)

((DRY_RUN)) || mkdir -p "$RUN_DIR"
if ((!DRY_RUN)) && [[ ! -e "$RUN_DIR/manifest.json" ]]; then
  printf '{\n  "run_id": "%s",\n  "seeds": "%s",\n  "deltas": [0, 0.0001, 0.001, 0.01],\n  "cone_count": %s,\n  "cone_dimension": %s,\n  "abs_tol": 1e-12,\n  "rel_tol": 1e-12,\n  "warmups": 5,\n  "measured_launches": 10,\n  "duration_seconds": %s,\n  "precision": "Float64",\n  "projection_type": 22,\n  "root_initialization": 0,\n  "gpu_physical_index": "%s",\n  "cuda_home": "%s",\n  "gpu_arch": "%s"\n}\n' \
    "$RUN_ID" "$SEEDS" "$COUNT" "$DIMENSION" "$DURATION" "$GPU" \
    "$CUDA_ROOT" "$ARCH" >"$RUN_DIR/manifest.json"
  {
    date -u '+timestamp_utc=%Y-%m-%dT%H:%M:%SZ'
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true
    git -C "$REPO_ROOT" status --short 2>/dev/null || true
    "$JULIA_BIN" --version
    "$CUDA_ROOT/bin/nvcc" --version
    nvidia-smi -i "$GPU" \
      --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free,power.limit,compute_mode \
      --format=csv
  } >"$RUN_DIR/environment.txt" 2>&1
fi
if has_phase snapshot; then
  run "$REPO_ROOT/benchmark/rebuttal/snapshot_source.sh" --output "$RUN_DIR/source_snapshot"
fi
if has_phase build; then
  run make -C "$REPO_ROOT/src/pdcs_gpu/cuda" rebuild-gpu rebuild-profile \
    CUDA_HOME="$CUDA_ROOT" ARCH="$ARCH"
fi
if has_phase test; then
  run env JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" "--project=$REPO_ROOT" \
    "$REPO_ROOT/test/rebuttal/soc_parametric_similarity_test.jl"
fi
if has_phase pilot; then
  args=("${common_env[@]}" "$JULIA_BIN" -O1 "--project=$REPO_ROOT"
    "$REPO_ROOT/benchmark/rescaled_soc_parametric_similar.jl"
    --mode pilot --seed 2025 --cone-count "$COUNT" --cone-dimension "$DIMENSION"
    --output-dir "$RUN_DIR/pilot/seed_2025")
  run "${args[@]}"
fi
if has_phase generate; then
  for seed in "${SEED_ARRAY[@]}"; do
    out="$RUN_DIR/seeds/seed_$seed"
    args=("${common_env[@]}" "$JULIA_BIN" -O1 "--project=$REPO_ROOT"
      "$REPO_ROOT/benchmark/rescaled_soc_parametric_similar.jl"
      --mode generate --seed "$seed" --cone-count "$COUNT"
      --cone-dimension "$DIMENSION" --output-dir "$out")
    ((SAVE_PER_CONE)) && args+=(--save-per-cone)
    run "${args[@]}"
    if ((!DRY_RUN && SAVE_PER_CONE)) && command -v zstd >/dev/null 2>&1; then
      while IFS= read -r raw; do
        zstd -q --rm -- "$raw"
      done < <(find "$out/diagnostic/per_cone" -type f -name '*.csv' -print)
    fi
  done
fi
if has_phase timing; then
  for seed in "${SEED_ARRAY[@]}"; do
    out="$RUN_DIR/seeds/seed_$seed"
    run "${common_env[@]}" "$JULIA_BIN" -O1 "--project=$REPO_ROOT" \
      "$REPO_ROOT/benchmark/rescaled_soc_parametric_similar.jl" \
      --mode timing --seed "$seed" --cone-count "$COUNT" \
      --cone-dimension "$DIMENSION" --case-cache "$out/case_cache.jls" \
      --output-dir "$out"
  done
fi
if has_phase utilization; then
  for seed in "${SEED_ARRAY[@]}"; do
    cache="$RUN_DIR/seeds/seed_$seed/case_cache.jls"
    for delta in 0 0.01; do
      for strategy in threadWise warpWise; do
        run "$REPO_ROOT/benchmark/run_soc_parametric_similarity_utilization.sh" \
          --run-dir "$RUN_DIR/utilization/raw" --case-cache "$cache" \
          --seed "$seed" --delta "$delta" --strategy "$strategy" \
          --duration "$DURATION" --cone-count "$COUNT" \
          --cone-dimension "$DIMENSION" --gpu "$GPU" \
          --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" \
          --julia-depot "$JULIA_DEPOT"
      done
    done
  done
fi
if has_phase analysis; then
  run env JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" "--project=$REPO_ROOT" \
    "$REPO_ROOT/benchmark/analyze_soc_parametric_similar.jl" \
    --root "$RUN_DIR" --seeds "$SEEDS" --bootstrap 10000
fi
printf 'PHASE_COMPLETE phase=%s run_dir=%s\n' "$PHASE" "$RUN_DIR"
