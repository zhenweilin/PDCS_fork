#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_DIR="${ROOT_DIR}/benchmark/represent_data"
JULIA_BIN="${ROOT_DIR}/.julia-bin/julia-1.12.6/bin/julia"
GPU_SELECTION="auto"
TIME_LIMIT="600"
TOLERANCE="1e-6"
ORDER_SEED="20260730"
RUN_ID="progressive_ablation_$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR=""
RESUME="true"
IDLE_MEMORY_LIMIT_MIB="2048"
IDLE_UTILIZATION_LIMIT="10"
CONFIGS="pdhg,pdhg_restart,pdhg_restart_scaling,pdhg_restart_scaling_reflection,pdhg_restart_scaling_reflection_adaptive_primal_weight,pdhg_restart_scaling_reflection_adaptive"
CONFIG_COUNT="6"

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "  --input-dir PATH" \
        "  --output-dir PATH" \
        "  --run-id ID" \
        "  --gpus auto|INDEX[,INDEX...]" \
        "  --julia PATH" \
        "  --time-limit SECONDS       default: 600" \
        "  --tolerance VALUE          default: 1e-6" \
        "  --order-seed INTEGER       default: 20260730" \
        "  --resume true|false        default: true" \
        "" \
        "With --gpus auto, every GPU satisfying memory.used < 2048 MiB and" \
        "utilization <= 10% at launch is used. One worker runs per selected GPU."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --gpus) GPU_SELECTION="$2"; shift 2 ;;
        --julia) JULIA_BIN="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --tolerance) TOLERANCE="$2"; shift 2 ;;
        --order-seed) ORDER_SEED="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${RESUME}" != "true" && "${RESUME}" != "false" ]]; then
    printf -- '--resume must be true or false\n' >&2
    exit 2
fi
if ! [[ "${TIME_LIMIT}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf -- '--time-limit must be a nonnegative number\n' >&2
    exit 2
fi

INPUT_DIR="$(readlink -f "${INPUT_DIR}")"
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/benchmark/results/rebuttal/progressive_ablation/${RUN_ID}"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(readlink -f "${OUTPUT_DIR}")"

if [[ ! -x "${JULIA_BIN}" ]]; then
    printf 'Julia executable not found: %s\n' "${JULIA_BIN}" >&2
    exit 2
fi

declare -a SELECTED_GPUS=()
if [[ "${GPU_SELECTION}" == "auto" ]]; then
    mapfile -t SELECTED_GPUS < <(
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits |
        awk -F, \
            -v memory_limit="${IDLE_MEMORY_LIMIT_MIB}" \
            -v utilization_limit="${IDLE_UTILIZATION_LIMIT}" '
                {
                    gsub(/ /, "", $1);
                    gsub(/ /, "", $2);
                    gsub(/ /, "", $3);
                    if ($2 < memory_limit && $3 <= utilization_limit) {
                        print $1;
                    }
                }
            '
    )
else
    IFS=',' read -r -a SELECTED_GPUS <<< "${GPU_SELECTION}"
fi

if (( ${#SELECTED_GPUS[@]} == 0 )); then
    printf 'No idle GPU satisfies the launch thresholds.\n' >&2
    exit 3
fi

declare -A SEEN_GPUS=()
for gpu in "${SELECTED_GPUS[@]}"; do
    if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
        printf 'Invalid GPU index: %s\n' "${gpu}" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        printf 'Duplicate GPU index: %s\n' "${gpu}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu}"]=1
    gpu_state="$(
        nvidia-smi -i "${gpu}" \
            --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits
    )"
    gpu_used="$(printf '%s\n' "${gpu_state}" |
        awk -F, '{gsub(/ /, "", $4); print $4}')"
    gpu_util="$(printf '%s\n' "${gpu_state}" |
        awk -F, '{gsub(/ /, "", $6); print $6}')"
    if (( gpu_used >= IDLE_MEMORY_LIMIT_MIB || gpu_util > IDLE_UTILIZATION_LIMIT )); then
        printf 'Selected GPU is not idle: %s\n' "${gpu_state}" >&2
        exit 3
    fi
done

GPU_LIST="$(IFS=,; printf '%s' "${SELECTED_GPUS[*]}")"
MANIFEST="${OUTPUT_DIR}/manifest.csv"
RUN_ORDER="${OUTPUT_DIR}/run_order.csv"
if [[ ! -f "${MANIFEST}" ]]; then
    python3 "${ROOT_DIR}/benchmark/ablation/inspect_represent_data.py" \
        --input-dir "${INPUT_DIR}" \
        --output "${MANIFEST}" \
        --order-seed "${ORDER_SEED}" \
        --expected-count 63
fi
if [[ ! -f "${RUN_ORDER}" ]]; then
    python3 \
        "${ROOT_DIR}/benchmark/progressive_ablation/make_run_order.py" \
        --manifest "${MANIFEST}" \
        --output "${RUN_ORDER}" \
        --order-seed "${ORDER_SEED}" \
        --expected-count 63
fi

run_order_rows="$(awk 'END {print NR - 1}' "${RUN_ORDER}")"
if [[ "${run_order_rows}" != "378" ]]; then
    printf 'Expected 378 run-order rows, found %s\n' "${run_order_rows}" >&2
    exit 2
fi

if [[ ! -f "${OUTPUT_DIR}/environment.txt" ]]; then
    {
        printf 'run_id=%s\n' "${RUN_ID}"
        printf 'experiment=progressive_cumulative_ablation\n'
        printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'root_dir=%s\n' "${ROOT_DIR}"
        printf 'input_dir=%s\n' "${INPUT_DIR}"
        printf 'output_dir=%s\n' "${OUTPUT_DIR}"
        printf 'configurations=%s\n' "${CONFIGS}"
        printf 'time_limit_seconds=%s\n' "${TIME_LIMIT}"
        printf 'tolerance=%s\n' "${TOLERANCE}"
        printf 'order_seed=%s\n' "${ORDER_SEED}"
        printf 'physical_gpus=%s\n' "${GPU_LIST}"
        printf 'idle_memory_limit_mib=%s\n' "${IDLE_MEMORY_LIMIT_MIB}"
        printf 'idle_utilization_limit_percent=%s\n' \
            "${IDLE_UTILIZATION_LIMIT}"
        printf 'julia_bin=%s\n' "${JULIA_BIN}"
        "${JULIA_BIN}" --version
        nvidia-smi
        /usr/local/cuda/bin/nvcc --version
        git -C "${ROOT_DIR}" rev-parse HEAD
        git -C "${ROOT_DIR}" status --short
        sha256sum \
            "${ROOT_DIR}/Project.toml" \
            "${ROOT_DIR}/Manifest.toml" \
            "${ROOT_DIR}/src/pdcs_gpu/cuda/utils.cu" \
            "${ROOT_DIR}/src/pdcs_gpu/cuda/utils.ptx" \
            "${ROOT_DIR}/src/pdcs_gpu/rpdhg_alg_gpu_gen_scaling.jl" \
            "${ROOT_DIR}/benchmark/ablation/run_ablation_case.jl"
    } > "${OUTPUT_DIR}/environment.txt" 2>&1
fi

analyze_results() {
    python3 "${ROOT_DIR}/benchmark/ablation/analyze_ablation.py" \
        --run-dir "${OUTPUT_DIR}" \
        --configs "${CONFIGS}" \
        --tolerance "${TOLERANCE}" \
        --timeout-value "${TIME_LIMIT}" \
        --sgm-shift 10 \
        --bootstrap-samples 10000 \
        --bootstrap-seed "${ORDER_SEED}" \
        --report "${OUTPUT_DIR}/report.md" \
        >> "${OUTPUT_DIR}/analysis.log" 2>&1 || true
    if [[ -f "${OUTPUT_DIR}/raw_results.csv" ]]; then
        python3 \
            "${ROOT_DIR}/benchmark/progressive_ablation/analyze_progressive_ablation.py" \
            --run-dir "${OUTPUT_DIR}" \
            --time-limit "${TIME_LIMIT}" \
            --tolerance "${TOLERANCE}" \
            --bootstrap-samples 10000 \
            --bootstrap-seed "${ORDER_SEED}" \
            --report "${OUTPUT_DIR}/progressive_report.md" \
            >> "${OUTPUT_DIR}/analysis.log" 2>&1 || true
    fi
}

trap analyze_results EXIT

mapfile -t INSTANCES < <(
    awk -F, 'NR > 1 && !seen[$2]++ {print $2}' "${RUN_ORDER}"
)
if (( ${#INSTANCES[@]} != 63 )); then
    printf 'Expected 63 unique instances, found %s\n' \
        "${#INSTANCES[@]}" >&2
    exit 2
fi

SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
SESSION_DIR="${OUTPUT_DIR}/scheduler_sessions/${SESSION_ID}"
CLAIM_DIR="${SESSION_DIR}/claims"
mkdir -p "${CLAIM_DIR}"
LOCK_FILE="${SESSION_DIR}/claim.lock"
ASSIGNMENTS="${OUTPUT_DIR}/gpu_assignments.csv"
if [[ ! -f "${ASSIGNMENTS}" ]]; then
    printf 'session_id,instance_id,gpu,claimed_utc,driver_log\n' \
        > "${ASSIGNMENTS}"
fi

INSTANCE_TIMEOUT="$(
    python3 -c "print(int(float('${TIME_LIMIT}') * ${CONFIG_COUNT} + 1200))"
)"

all_configurations_done() {
    local instance_id="$1"
    local configuration
    IFS=',' read -r -a configurations <<< "${CONFIGS}"
    for configuration in "${configurations[@]}"; do
        if [[ ! -f \
            "${OUTPUT_DIR}/cases/${instance_id}/${configuration}/DONE" ]]
        then
            return 1
        fi
    done
    return 0
}

claim_next_instance() {
    local gpu="$1"
    local instance_id
    local driver_log
    local driver_number
    local claimed=""
    exec 9> "${LOCK_FILE}"
    flock -x 9
    for instance_id in "${INSTANCES[@]}"; do
        if [[ "${RESUME}" == "true" ]] &&
            all_configurations_done "${instance_id}"
        then
            continue
        fi
        if [[ -f "${CLAIM_DIR}/${instance_id}" ]]; then
            continue
        fi
        : > "${CLAIM_DIR}/${instance_id}"
        driver_number=1
        while :; do
            driver_log="$(
                printf '%s/instance_%s.driver_attempt_%03d.log' \
                    "${OUTPUT_DIR}" "${instance_id}" "${driver_number}"
            )"
            [[ -e "${driver_log}" ]] || break
            driver_number=$((driver_number + 1))
        done
        printf '%s,%s,%s,%s,%s\n' \
            "${SESSION_ID}" "${instance_id}" "${gpu}" \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${driver_log}" \
            >> "${ASSIGNMENTS}"
        printf '%s|%s' "${instance_id}" "${driver_log}"
        claimed="true"
        break
    done
    flock -u 9
    exec 9>&-
    [[ "${claimed}" == "true" ]]
}

run_worker() {
    local gpu="$1"
    local claim
    local instance_id
    local driver_log
    local return_code
    local failures=0
    while claim="$(claim_next_instance "${gpu}")"; do
        instance_id="${claim%%|*}"
        driver_log="${claim#*|}"
        printf 'RUN_INSTANCE instance=%s gpu=%s utc=%s\n' \
            "${instance_id}" "${gpu}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PDCS_SKIP_GPU_PRECOMPILE=1 \
        JULIA_PKG_PRECOMPILE_AUTO=0 \
        LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
        timeout --signal=TERM --kill-after=60 "${INSTANCE_TIMEOUT}" \
            "${JULIA_BIN}" \
            --startup-file=no \
            --compiled-modules=existing \
            -O1 \
            --project="${ROOT_DIR}" \
            "${ROOT_DIR}/benchmark/ablation/run_ablation_case.jl" \
            --input-dir "${INPUT_DIR}" \
            --run-order "${RUN_ORDER}" \
            --output-dir "${OUTPUT_DIR}" \
            --instance-id "${instance_id}" \
            --expected-config-count "${CONFIG_COUNT}" \
            --tolerance "${TOLERANCE}" \
            --time-limit "${TIME_LIMIT}" \
            --check-frequency 2000 \
            --print-frequency 20000 \
            --warmup true \
            >> "${driver_log}" 2>&1
        return_code=$?
        if (( return_code == 0 )); then
            printf 'INSTANCE_COMPLETE instance=%s gpu=%s\n' \
                "${instance_id}" "${gpu}"
        else
            failures=$((failures + 1))
            printf 'INSTANCE_FAILED instance=%s gpu=%s return_code=%s log=%s\n' \
                "${instance_id}" "${gpu}" "${return_code}" "${driver_log}" >&2
        fi
    done
    printf '%s\n' "${failures}" > "${SESSION_DIR}/worker_gpu${gpu}.failures"
    printf 'WORKER_COMPLETE gpu=%s failures=%s\n' "${gpu}" "${failures}"
}

declare -a WORKER_PIDS=()
terminate_workers() {
    local pid
    for pid in "${WORKER_PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
}
trap 'terminate_workers; exit 130' INT TERM

for gpu in "${SELECTED_GPUS[@]}"; do
    run_worker "${gpu}" \
        > "${SESSION_DIR}/worker_gpu${gpu}.log" 2>&1 &
    WORKER_PIDS+=("$!")
done

worker_wait_failures=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "${pid}"; then
        worker_wait_failures=$((worker_wait_failures + 1))
    fi
done

driver_failures=0
for gpu in "${SELECTED_GPUS[@]}"; do
    failure_file="${SESSION_DIR}/worker_gpu${gpu}.failures"
    if [[ -f "${failure_file}" ]]; then
        failures="$(< "${failure_file}")"
        driver_failures=$((driver_failures + failures))
    else
        driver_failures=$((driver_failures + 1))
    fi
done

analyze_results
trap - EXIT INT TERM

{
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'last_session_id=%s\n' "${SESSION_ID}"
    printf 'last_session_gpus=%s\n' "${GPU_LIST}"
    printf 'last_session_driver_failures=%s\n' "${driver_failures}"
    printf 'last_session_worker_wait_failures=%s\n' "${worker_wait_failures}"
} >> "${OUTPUT_DIR}/environment.txt"

printf 'PROGRESSIVE_ABLATION_COMPLETE run_dir=%s gpus=%s driver_failures=%s\n' \
    "${OUTPUT_DIR}" "${GPU_LIST}" "${driver_failures}"

if (( driver_failures != 0 || worker_wait_failures != 0 )); then
    exit 4
fi
