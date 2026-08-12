#!/usr/bin/env bash
# Multi-period portfolio baseline driver (scs_gpu | cuclarabel).
# Runs size groups in ascending order; if ANY seed of a size fails, the
# solver is aborted and all larger sizes are skipped (per user rule).
# cuPDCS is NOT driven here — it must attempt every instance.
set -uo pipefail

SOLVER="${1:?usage: run_mpp_baseline_driver.sh scs_gpu|cuclarabel}"
ROOT=/home/zhenwei/PDCS_fork
SIZES_DIR="$ROOT/benchmark/results/multi_period_port/size_folders"
OUT="$ROOT/benchmark/results/multi_period_port/$SOLVER"
JULIA=/home/zhenwei/.juliaup/bin/julia
TIME_LIMIT=18000          # 5 hours per instance, solver-internal
LOAD_MARGIN=2400          # generous CBF load margin per instance (10GB gz)
SIZE_TIMEOUT=$(( 5 * (TIME_LIMIT + LOAD_MARGIN) ))
LOG="$ROOT/benchmark/results/multi_period_port/driver_${SOLVER}.log"

case "$SOLVER" in
    scs_gpu)
        PROJECT="$ROOT/benchmark/scs_gpu_env"
        DEPOT="/tmp/pdcs_scs_gpu_depot"
        SCRIPT="$ROOT/benchmark/multi_period_port_scs_gpu.jl" ;;
    cuclarabel)
        PROJECT="$ROOT/benchmark/cuclarabel_env"
        DEPOT="/tmp/pdcs_cuclarabel_depot:$ROOT/.julia-depot"
        SCRIPT="$ROOT/benchmark/multi_period_port_cuclarabel.jl" ;;
    *) echo "unknown solver: $SOLVER" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
log(){ printf 'MPP_DRIVER %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

size_ok(){
    local d="$1" f stem term raw
    for f in "$d"/*.cbf.gz; do
        stem="$(basename "$f" .cbf.gz)"
        raw="$OUT/$stem.raw.log"
        [[ -f "$raw" ]] || { echo "fail:missing_log($stem)"; return; }
        if grep -q '^RUN_STATUS=EXCEPTION' "$raw"; then
            echo "fail:exception($stem)"; return
        fi
        term="$(grep -m1 '^TERMINATION_STATUS=' "$raw" | cut -d= -f2)"
        case "$term" in
            OPTIMAL*|ALMOST_OPTIMAL*) ;;
            *) echo "fail:termination=${term:-none}($stem)"; return ;;
        esac
    done
    echo success
}

log "DRIVER_START solver=$SOLVER time_limit=$TIME_LIMIT"

for d in "$SIZES_DIR"/*/; do
    size="$(basename "$d")"
    log "SIZE_START solver=$SOLVER size=$size"
    setsid env -u LD_LIBRARY_PATH JULIA_PKG_OFFLINE=true "JULIA_DEPOT_PATH=$DEPOT" \
        "$JULIA" --startup-file=no --project="$PROJECT" "$SCRIPT" \
        --input_folder "$d" --output_folder "$OUT" --time_limit "$TIME_LIMIT" \
        >> "$LOG" 2>&1 &
    pid=$!
    ( sleep "$SIZE_TIMEOUT"
      kill -INT -- -"$pid" 2>/dev/null
      sleep 60
      kill -KILL -- -"$pid" 2>/dev/null ) &
    wd=$!
    wait "$pid"
    rc=$?
    kill "$wd" 2>/dev/null
    wait "$wd" 2>/dev/null
    outcome="$(size_ok "$d")"
    log "SIZE_DONE solver=$SOLVER size=$size rc=$rc outcome=$outcome"
    if [[ "$outcome" != success ]]; then
        log "SOLVER_ABORT solver=$SOLVER at size=$size -> skipping all larger sizes"
        break
    fi
done
log "DRIVER_COMPLETE solver=$SOLVER"
