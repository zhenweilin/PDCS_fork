#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_DIR="${ROOT_DIR}/benchmark/represent_data"
JULIA_BIN="${ROOT_DIR}/.julia-bin/julia-1.12.6/bin/julia"
GPU_INDEX="5"
TIME_LIMIT="600"
TOLERANCE="1e-6"
ORDER_SEED="20260730"
RUN_ID="halpern_candidate_$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR=""
RESUME="true"
CONFIGS="with_halpern_candidate,without_halpern_candidate"

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "  --input-dir PATH" \
        "  --output-dir PATH" \
        "  --run-id ID" \
        "  --julia PATH" \
        "  --time-limit SECONDS       default: 600" \
        "  --tolerance VALUE          default: 1e-6" \
        "  --order-seed INTEGER       default: 20260730" \
        "  --resume true|false        default: true" \
        "" \
        "This formal experiment is deliberately pinned to physical GPU 5."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --julia) JULIA_BIN="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --tolerance) TOLERANCE="$2"; shift 2 ;;
        --order-seed) ORDER_SEED="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --gpu)
            printf 'GPU selection is fixed: use physical GPU 5 only.\n' >&2
            exit 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

INPUT_DIR="$(readlink -f "${INPUT_DIR}")"
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/benchmark/results/rebuttal/halpern_candidate/${RUN_ID}"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(readlink -f "${OUTPUT_DIR}")"

if [[ ! -x "${JULIA_BIN}" ]]; then
    printf 'Julia executable not found: %s\n' "${JULIA_BIN}" >&2
    exit 2
fi

GPU_STATE="$(
    nvidia-smi -i "${GPU_INDEX}" \
        --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
)"
GPU_USED="$(printf '%s\n' "${GPU_STATE}" | awk -F, '{gsub(/ /, "", $4); print $4}')"
GPU_UTIL="$(printf '%s\n' "${GPU_STATE}" | awk -F, '{gsub(/ /, "", $6); print $6}')"
if (( GPU_USED >= 2048 || GPU_UTIL > 10 )); then
    printf 'Physical GPU 5 is not idle: %s\n' "${GPU_STATE}" >&2
    exit 3
fi

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
        "${ROOT_DIR}/benchmark/halpern_candidate_ablation/make_run_order.py" \
        --manifest "${MANIFEST}" \
        --output "${RUN_ORDER}" \
        --order-seed "${ORDER_SEED}" \
        --expected-count 63
fi

if [[ ! -f "${OUTPUT_DIR}/environment.txt" ]]; then
    {
        printf 'run_id=%s\n' "${RUN_ID}"
        printf 'experiment=halpern_restart_candidate_ablation\n'
        printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'root_dir=%s\n' "${ROOT_DIR}"
        printf 'input_dir=%s\n' "${INPUT_DIR}"
        printf 'output_dir=%s\n' "${OUTPUT_DIR}"
        printf 'configurations=%s\n' "${CONFIGS}"
        printf 'time_limit_seconds=%s\n' "${TIME_LIMIT}"
        printf 'tolerance=%s\n' "${TOLERANCE}"
        printf 'order_seed=%s\n' "${ORDER_SEED}"
        printf 'physical_gpu=%s\n' "${GPU_INDEX}"
        printf 'gpu_state=%s\n' "${GPU_STATE}"
        printf 'julia_bin=%s\n' "${JULIA_BIN}"
        "${JULIA_BIN}" --version
        nvidia-smi -i "${GPU_INDEX}"
        /usr/local/cuda/bin/nvcc --version
        git -C "${ROOT_DIR}" rev-parse HEAD
        git -C "${ROOT_DIR}" status --short
        sha256sum \
            "${ROOT_DIR}/Project.toml" \
            "${ROOT_DIR}/Manifest.toml" \
            "${ROOT_DIR}/src/pdcs_gpu/cuda/utils.cu" \
            "${ROOT_DIR}/src/pdcs_gpu/cuda/utils.ptx" \
            "${ROOT_DIR}/src/pdcs_gpu/rpdhg_alg_gpu_gen_scaling.jl"
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
}
trap analyze_results EXIT

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PDCS_SKIP_GPU_PRECOMPILE=1
export JULIA_PKG_PRECOMPILE_AUTO=0
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

INSTANCE_TIMEOUT="$(python3 -c "print(int(float('${TIME_LIMIT}') * 2 + 900))")"
mapfile -t INSTANCES < <(
    awk -F, 'NR > 1 && !seen[$2]++ {print $2}' "${RUN_ORDER}"
)

overall_failures=0
for instance_id in "${INSTANCES[@]}"; do
    if [[ "${RESUME}" == "true" ]]; then
        complete_count=0
        for configuration in \
            with_halpern_candidate without_halpern_candidate
        do
            if [[ -f "${OUTPUT_DIR}/cases/${instance_id}/${configuration}/DONE" ]]; then
                complete_count=$((complete_count + 1))
            fi
        done
        if (( complete_count == 2 )); then
            printf 'SKIP_INSTANCE_DONE instance=%s\n' "${instance_id}"
            continue
        fi
    fi

    printf 'RUN_INSTANCE instance=%s gpu=%s utc=%s\n' \
        "${instance_id}" "${GPU_INDEX}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    driver_log="${OUTPUT_DIR}/instance_${instance_id}.driver.log"
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
        --expected-config-count 2 \
        --tolerance "${TOLERANCE}" \
        --time-limit "${TIME_LIMIT}" \
        --check-frequency 2000 \
        --print-frequency 20000 \
        --warmup true \
        >> "${driver_log}" 2>&1
    return_code=$?
    if (( return_code != 0 )); then
        overall_failures=$((overall_failures + 1))
        printf 'INSTANCE_FAILED instance=%s return_code=%s log=%s\n' \
            "${instance_id}" "${return_code}" "${driver_log}" >&2
    else
        printf 'INSTANCE_COMPLETE instance=%s\n' "${instance_id}"
    fi
done

analyze_results
trap - EXIT

printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "${OUTPUT_DIR}/environment.txt"
printf 'instance_driver_failures=%s\n' "${overall_failures}" \
    >> "${OUTPUT_DIR}/environment.txt"
printf 'HALPERN_CANDIDATE_ABLATION_COMPLETE run_dir=%s driver_failures=%s\n' \
    "${OUTPUT_DIR}" "${overall_failures}"

if (( overall_failures != 0 )); then
    exit 4
fi
