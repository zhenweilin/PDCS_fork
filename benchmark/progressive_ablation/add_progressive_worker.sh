#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR=""
SESSION_ID=""
GPU=""
IDLE_MEMORY_LIMIT_MIB="2048"
IDLE_UTILIZATION_LIMIT="10"
CONFIGS="pdhg,pdhg_restart,pdhg_restart_scaling,pdhg_restart_scaling_reflection,pdhg_restart_scaling_reflection_adaptive_primal_weight,pdhg_restart_scaling_reflection_adaptive"
CONFIG_COUNT="6"

usage() {
    printf '%s\n' \
        "Usage: $0 --run-dir PATH --session-id ID --gpu INDEX" \
        "" \
        "Adds one worker to an active progressive-ablation scheduler session." \
        "The worker shares that session's lock and claims, so an instance cannot" \
        "be assigned to both the original workers and this added worker." \
        "" \
        "The run directory must contain environment.txt and run_order.csv."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --session-id) SESSION_ID="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${RUN_DIR}" || -z "${SESSION_ID}" || -z "${GPU}" ]]; then
    usage >&2
    exit 2
fi
if ! [[ "${GPU}" =~ ^[0-9]+$ ]]; then
    printf 'Invalid GPU index: %s\n' "${GPU}" >&2
    exit 2
fi

RUN_DIR="$(readlink -f "${RUN_DIR}")"
ENVIRONMENT="${RUN_DIR}/environment.txt"
RUN_ORDER="${RUN_DIR}/run_order.csv"
SESSION_DIR="${RUN_DIR}/scheduler_sessions/${SESSION_ID}"
CLAIM_DIR="${SESSION_DIR}/claims"
LOCK_FILE="${SESSION_DIR}/claim.lock"
ASSIGNMENTS="${RUN_DIR}/gpu_assignments.csv"

for required in \
    "${ENVIRONMENT}" \
    "${RUN_ORDER}" \
    "${LOCK_FILE}" \
    "${ASSIGNMENTS}"
do
    if [[ ! -e "${required}" ]]; then
        printf 'Required scheduler file is missing: %s\n' "${required}" >&2
        exit 2
    fi
done

read_environment_value() {
    local key="$1"
    awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' \
        "${ENVIRONMENT}"
}

INPUT_DIR="$(read_environment_value input_dir)"
JULIA_BIN="$(read_environment_value julia_bin)"
TIME_LIMIT="$(read_environment_value time_limit_seconds)"
TOLERANCE="$(read_environment_value tolerance)"
RECORDED_CONFIGS="$(read_environment_value configurations)"

if [[ "${JULIA_BIN}" != /* ]]; then
    JULIA_BIN="${ROOT_DIR}/${JULIA_BIN#./}"
fi

if [[ "${RECORDED_CONFIGS}" != "${CONFIGS}" ]]; then
    printf 'Configuration mismatch in %s\n' "${ENVIRONMENT}" >&2
    printf 'Expected: %s\nRecorded: %s\n' "${CONFIGS}" "${RECORDED_CONFIGS}" >&2
    exit 2
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
    printf 'Recorded input directory is unavailable: %s\n' "${INPUT_DIR}" >&2
    exit 2
fi
if [[ ! -x "${JULIA_BIN}" ]]; then
    printf 'Recorded Julia executable is unavailable: %s\n' "${JULIA_BIN}" >&2
    exit 2
fi

gpu_state="$(
    nvidia-smi -i "${GPU}" \
        --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
)"
gpu_used="$(printf '%s\n' "${gpu_state}" |
    awk -F, '{gsub(/ /, "", $4); print $4}')"
gpu_util="$(printf '%s\n' "${gpu_state}" |
    awk -F, '{gsub(/ /, "", $6); print $6}')"
if (( gpu_used >= IDLE_MEMORY_LIMIT_MIB || gpu_util > IDLE_UTILIZATION_LIMIT )); then
    printf 'GPU is not idle; added worker was not started: %s\n' "${gpu_state}" >&2
    exit 3
fi

exec 8> "${SESSION_DIR}/added_worker_gpu${GPU}.lock"
if ! flock -n 8; then
    printf 'An added worker for GPU %s is already active in session %s.\n' \
        "${GPU}" "${SESSION_ID}" >&2
    exit 3
fi

mapfile -t INSTANCES < <(
    awk -F, 'NR > 1 && !seen[$2]++ {print $2}' "${RUN_ORDER}"
)
if (( ${#INSTANCES[@]} != 63 )); then
    printf 'Expected 63 unique instances, found %s\n' \
        "${#INSTANCES[@]}" >&2
    exit 2
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
            "${RUN_DIR}/cases/${instance_id}/${configuration}/DONE" ]]
        then
            return 1
        fi
    done
    return 0
}

claim_next_instance() {
    local instance_id
    local driver_log
    local driver_number
    local claimed=""
    exec 9> "${LOCK_FILE}"
    flock -x 9
    for instance_id in "${INSTANCES[@]}"; do
        if all_configurations_done "${instance_id}"; then
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
                    "${RUN_DIR}" "${instance_id}" "${driver_number}"
            )"
            [[ -e "${driver_log}" ]] || break
            driver_number=$((driver_number + 1))
        done
        printf '%s,%s,%s,%s,%s\n' \
            "${SESSION_ID}" "${instance_id}" "${GPU}" \
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

failures=0
while claim="$(claim_next_instance)"; do
    instance_id="${claim%%|*}"
    driver_log="${claim#*|}"
    printf 'RUN_INSTANCE instance=%s gpu=%s utc=%s added_worker=true\n' \
        "${instance_id}" "${GPU}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    CUDA_VISIBLE_DEVICES="${GPU}" \
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
        --output-dir "${RUN_DIR}" \
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
        printf 'INSTANCE_COMPLETE instance=%s gpu=%s added_worker=true\n' \
            "${instance_id}" "${GPU}"
    else
        failures=$((failures + 1))
        printf 'INSTANCE_FAILED instance=%s gpu=%s return_code=%s log=%s\n' \
            "${instance_id}" "${GPU}" "${return_code}" "${driver_log}" >&2
    fi
done

printf '%s\n' "${failures}" \
    > "${SESSION_DIR}/added_worker_gpu${GPU}.failures"
printf 'ADDED_WORKER_COMPLETE gpu=%s failures=%s\n' "${GPU}" "${failures}"
exit "$(( failures == 0 ? 0 : 4 ))"
