#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${BASE_DIR}/../.." && pwd)"

JULIA_BIN="${JULIA_BIN:-julia}"
ENV_DIR="${ENV_DIR:-${BASE_DIR}}"
CONFIG="${CONFIG:-${BASE_DIR}/lasso_table5.toml}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/table5_logs}"
SOLVER_DIR="${SOLVER_DIR:-${BASE_DIR}/table5_solver_results}"
REPORT_MD="${REPORT_MD:-${BASE_DIR}/table5_report.md}"
REPORT_CSV="${REPORT_CSV:-${BASE_DIR}/table5_report.csv}"
SOLVER_REPORT_CSV="${SOLVER_REPORT_CSV:-${BASE_DIR}/table5_solver_report.csv}"

GENERATOR="${BASE_DIR}/large_scale_lasso.jl"
MANIFEST_GENERATOR="${BASE_DIR}/generate_table5_manifest.jl"
SOLVE_SCRIPT="${BASE_DIR}/solve_table5_instance.jl"
REPORT_SCRIPT="${BASE_DIR}/table5_report.py"
MASTER_SEED="20260728"
REQUIRED_JULIA_VERSION="${REQUIRED_JULIA_VERSION:-1.10.4}"
GPU_INDEX="${GPU_INDEX:-auto}"
SOLVERS="${SOLVERS:-cupdcs,scs_gpu,cuclarabel}"
INSTANCES="${INSTANCES:-all}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
TOLERANCE="${TOLERANCE:-1e-6}"
PRINT_FREQUENCY="${PRINT_FREQUENCY:-1000}"
VERBOSE_LEVEL="${VERBOSE_LEVEL:-2}"
PDCS_PROFILE="${PDCS_PROFILE:-full}"
MAX_START_UTIL="${MAX_START_UTIL:-10}"
MIN_FREE_MIB="${MIN_FREE_MIB:-70000}"
PDCS_DEPOT="${PDCS_DEPOT:-${ROOT_DIR}/.julia-depot}"
SCS_GPU_DEPOT="${SCS_GPU_DEPOT:-/tmp/pdcs_scs_gpu_depot}"
CUCLARABEL_DEPOT="${CUCLARABEL_DEPOT:-/tmp/pdcs_cuclarabel_depot}"

usage() {
    cat <<EOF
Usage: $(basename "$0") COMMAND [options]

Commands:
  prepare   Check Julia/environment/resources and create or validate the manifest.
  solve     Regenerate each case in memory and solve it with each GPU solver.
  report    Validate current logs/results and write Markdown/CSV reports.
  all       Run prepare, solve, and report.

Options:
  --julia PATH
  --env-dir PATH
  --config PATH
  --log-dir PATH
  --solver-dir PATH
  --required-julia-version VERSION
  --gpu INDEX|auto
  --solvers cupdcs,scs_gpu,cuclarabel
  --instances all|id1,id2
  --time-limit SECONDS
  --tolerance VALUE
  --print-frequency ITERATIONS
  --verbose-level 0|1|2
  --pdcs-profile full|baseline
  --max-start-util PERCENT
  --min-free-mib MIB

No CBF file is created. Every solver/case runs in a fresh Julia process,
deterministically regenerates the matrix from the manifest seed, solves it,
then retains only its raw log and a small TOML result. Strict reproduction
defaults to Julia 1.10.4 and the exact hashed generator Project/Manifest/script.
Use --required-julia-version only for an explicit smoke test; formal runs must
keep the 1.10.4 default.
EOF
}

[[ $# -gt 0 ]] || {
    usage >&2
    exit 2
}
COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --julia) JULIA_BIN="$2"; shift 2 ;;
        --env-dir) ENV_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --solver-dir) SOLVER_DIR="$2"; shift 2 ;;
        --required-julia-version) REQUIRED_JULIA_VERSION="$2"; shift 2 ;;
        --gpu) GPU_INDEX="$2"; shift 2 ;;
        --solvers) SOLVERS="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --tolerance) TOLERANCE="$2"; shift 2 ;;
        --print-frequency) PRINT_FREQUENCY="$2"; shift 2 ;;
        --verbose-level) VERBOSE_LEVEL="$2"; shift 2 ;;
        --pdcs-profile) PDCS_PROFILE="$2"; shift 2 ;;
        --max-start-util) MAX_START_UTIL="$2"; shift 2 ;;
        --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${COMMAND}" in
    prepare|solve|report|all) ;;
    *) printf 'Unknown command: %s\n' "${COMMAND}" >&2; usage >&2; exit 2 ;;
esac

for value in \
    "${TIME_LIMIT}" "${PRINT_FREQUENCY}" "${VERBOSE_LEVEL}" \
    "${MAX_START_UTIL}" "${MIN_FREE_MIB}"
do
    [[ "${value}" =~ ^[0-9]+$ ]] || {
        printf 'Expected a nonnegative integer, got: %s\n' "${value}" >&2
        exit 2
    }
done
(( TIME_LIMIT > 0 && PRINT_FREQUENCY > 0 )) || {
    printf 'time-limit and print-frequency must be positive\n' >&2
    exit 2
}
(( VERBOSE_LEVEL >= 0 && VERBOSE_LEVEL <= 2 )) || {
    printf 'verbose-level must be 0, 1, or 2\n' >&2
    exit 2
}
[[ "${PDCS_PROFILE}" == "full" || "${PDCS_PROFILE}" == "baseline" ]] || {
    printf 'pdcs-profile must be full or baseline\n' >&2
    exit 2
}
[[ "${TOLERANCE}" =~ ^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$ ]] || {
    printf 'Invalid tolerance: %s\n' "${TOLERANCE}" >&2
    exit 2
}
[[ -x "${JULIA_BIN}" ]] || command -v "${JULIA_BIN}" >/dev/null 2>&1 || {
    printf 'Julia executable not found: %s\n' "${JULIA_BIN}" >&2
    exit 2
}
[[ -f "${GENERATOR}" && -x "${MANIFEST_GENERATOR}" &&
    -f "${SOLVE_SCRIPT}" && -x "${REPORT_SCRIPT}" ]] || {
    printf 'Required Table 5 scripts are missing or not executable in %s\n' \
        "${BASE_DIR}" >&2
    exit 2
}
[[ -f "${ENV_DIR}/Project.toml" && -f "${ENV_DIR}/Manifest.toml" ]] || {
    printf 'ENV_DIR must contain Project.toml and Manifest.toml: %s\n' \
        "${ENV_DIR}" >&2
    exit 2
}

mkdir -p "${LOG_DIR}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DRIVER_LOG="${LOG_DIR}/driver_${RUN_ID}.log"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

resource_snapshot() {
    local label="$1"
    printf 'RESOURCE_SNAPSHOT label=%s utc=%s\n' "${label}" "$(timestamp)"
    free -h
    df -h "${BASE_DIR}"
}

require_julia_version() {
    local version
    version="$("${JULIA_BIN}" --version 2>&1)" || {
        printf 'Cannot run Julia: %s\n' "${JULIA_BIN}" >&2
        return 1
    }
    printf 'JULIA_VERSION %s\n' "${version}"
    [[ "${version}" == "julia version ${REQUIRED_JULIA_VERSION}" ]] || {
        printf 'This run requires exactly Julia %s.\n' \
            "${REQUIRED_JULIA_VERSION}" >&2
        return 1
    }
}

check_environment() {
    require_julia_version || return 1
    local expected_project actual_project
    expected_project="$(readlink -f "${ENV_DIR}/Project.toml")"
    actual_project="$(
        "${JULIA_BIN}" --project="${ENV_DIR}" \
            -e 'print(abspath(Base.active_project()))'
    )" || return 1
    [[ "${actual_project}" == "${expected_project}" ]] || {
        printf 'Active project mismatch: expected %s, got %s\n' \
            "${expected_project}" "${actual_project}" >&2
        return 1
    }
    "${JULIA_BIN}" --project="${ENV_DIR}" -e '
        using Pkg
        println("active project = ", Base.active_project())
        Pkg.status()
    ' || return 1
}

validate_manifest() {
    "${REPORT_SCRIPT}" validate-manifest --manifest "${CONFIG}"
}

ensure_manifest() {
    if [[ ! -f "${CONFIG}" ]]; then
        printf 'GENERATE_MANIFEST config=%s master_seed=%s\n' \
            "${CONFIG}" "${MASTER_SEED}"
        "${JULIA_BIN}" \
            --startup-file=no \
            "${MANIFEST_GENERATOR}" \
            --output "${CONFIG}" \
            --master-seed "${MASTER_SEED}" \
            --generator-script "${GENERATOR}" \
            --generator-project "${ENV_DIR}/Project.toml" \
            --generator-manifest "${ENV_DIR}/Manifest.toml" || return 1
    else
        printf 'KEEP_EXISTING_MANIFEST config=%s\n' "${CONFIG}"
    fi
    validate_manifest
}

prepare_phase() {
    printf 'TABLE5_PREPARE_START utc=%s\n' "$(timestamp)"
    check_environment || return 1
    resource_snapshot "prepare"
    {
        printf 'run_id=%s\n' "${RUN_ID}"
        printf 'recorded_utc=%s\n' "$(timestamp)"
        printf 'root_dir=%s\n' "${ROOT_DIR}"
        printf 'env_dir=%s\n' "${ENV_DIR}"
        printf 'config=%s\n' "${CONFIG}"
        printf 'solver_dir=%s\n' "${SOLVER_DIR}"
        printf 'required_julia_version=%s\n' "${REQUIRED_JULIA_VERSION}"
        printf 'storage_mode=regenerate_in_memory_no_cbf\n'
        "${JULIA_BIN}" --version
        uname -a
        free -h
        df -h "${BASE_DIR}"
        git -C "${ROOT_DIR}" rev-parse HEAD
        git -C "${ROOT_DIR}" status --short
        sha256sum \
            "${GENERATOR}" \
            "${ENV_DIR}/Project.toml" \
            "${ENV_DIR}/Manifest.toml" \
            "${MANIFEST_GENERATOR}" \
            "${SOLVE_SCRIPT}" \
            "${REPORT_SCRIPT}" \
            "$0"
    } > "${LOG_DIR}/environment_${RUN_ID}.txt" 2>&1
    ensure_manifest || return 1
    printf 'TABLE5_PREPARE_COMPLETE utc=%s\n' "$(timestamp)"
}

collect_kernel_diagnostics() {
    local output="$1"
    if dmesg -T > /dev/null 2>&1; then
        dmesg -T | tail -100 > "${output}" 2>&1
    elif command -v journalctl >/dev/null 2>&1; then
        journalctl -k -n 100 --no-pager > "${output}" 2>&1 || true
    else
        printf 'Neither dmesg nor journalctl is available.\n' > "${output}"
    fi
}

select_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf 'nvidia-smi is required for solve mode.\n' >&2
        return 1
    }
    if [[ "${GPU_INDEX}" == "auto" ]]; then
        GPU_INDEX="$(
            nvidia-smi \
                --query-gpu=index,memory.free,utilization.gpu \
                --format=csv,noheader,nounits |
            awk -F, \
                -v min_free="${MIN_FREE_MIB}" \
                -v max_util="${MAX_START_UTIL}" '
                    {
                        gsub(/ /, "", $1)
                        gsub(/ /, "", $2)
                        gsub(/ /, "", $3)
                        if ($2 >= min_free && $3 <= max_util) {
                            print $1
                            exit
                        }
                    }
                '
        )"
    fi
    [[ "${GPU_INDEX}" =~ ^[0-9]+$ ]] || {
        printf 'No GPU meets free-memory/utilization thresholds.\n' >&2
        return 1
    }
}

gpu_preflight() {
    local state free_mib utilization
    state="$(
        nvidia-smi -i "${GPU_INDEX}" \
            --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits
    )" || return 1
    free_mib="$(printf '%s\n' "${state}" | awk -F, '{gsub(/ /, "", $5); print $5}')"
    utilization="$(printf '%s\n' "${state}" | awk -F, '{gsub(/ /, "", $6); print $6}')"
    (( free_mib >= MIN_FREE_MIB && utilization <= MAX_START_UTIL )) || {
        printf 'GPU preflight failed: %s\n' "${state}" >&2
        return 1
    }
    printf 'GPU_PREFLIGHT %s\n' "${state}"
}

solver_project() {
    case "$1" in
        cupdcs) printf '%s\n' "${ROOT_DIR}" ;;
        scs_gpu) printf '%s\n' "${ROOT_DIR}/benchmark/scs_gpu_env" ;;
        cuclarabel) printf '%s\n' "${ROOT_DIR}/benchmark/cuclarabel_env" ;;
        *) return 1 ;;
    esac
}

solver_depot() {
    case "$1" in
        cupdcs) printf '%s\n' "${PDCS_DEPOT}" ;;
        scs_gpu) printf '%s\n' "${SCS_GPU_DEPOT}" ;;
        cuclarabel)
            printf '%s:%s\n' "${CUCLARABEL_DEPOT}" "${PDCS_DEPOT}"
            ;;
        *) return 1 ;;
    esac
}

solve_phase() {
    printf 'TABLE5_SOLVE_PHASE_START storage=no_cbf utc=%s\n' "$(timestamp)"
    require_julia_version || return 1
    ensure_manifest || return 1
    select_gpu || return 1

    IFS=, read -r -a solver_names <<< "${SOLVERS}"
    ((${#solver_names[@]} > 0)) || {
        printf 'At least one solver is required.\n' >&2
        return 2
    }
    local solver
    local -A seen_solvers=()
    for solver in "${solver_names[@]}"; do
        case "${solver}" in
            cupdcs|scs_gpu|cuclarabel) ;;
            *)
                printf 'Unsupported solver: %s\n' "${solver}" >&2
                return 2
                ;;
        esac
        [[ -z "${seen_solvers[${solver}]+present}" ]] || {
            printf 'Duplicate solver: %s\n' "${solver}" >&2
            return 2
        }
        seen_solvers["${solver}"]=1
    done
    mapfile -t instance_ids < <(
        "${REPORT_SCRIPT}" validate-manifest \
            --manifest "${CONFIG}" \
            --ids-only
    )
    [[ ${#instance_ids[@]} -eq 25 ]] || {
        printf 'Expected 25 manifest cases, got %s\n' "${#instance_ids[@]}" >&2
        return 2
    }
    if [[ "${INSTANCES}" != "all" ]]; then
        IFS=, read -r -a requested_instances <<< "${INSTANCES}"
        ((${#requested_instances[@]} > 0)) || {
            printf 'At least one instance ID is required.\n' >&2
            return 2
        }
        local requested known
        for requested in "${requested_instances[@]}"; do
            known=false
            for instance_id in "${instance_ids[@]}"; do
                if [[ "${instance_id}" == "${requested}" ]]; then
                    known=true
                    break
                fi
            done
            [[ "${known}" == true ]] || {
                printf 'Unknown instance ID: %s\n' "${requested}" >&2
                return 2
            }
        done
        instance_ids=("${requested_instances[@]}")
    fi

    export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
    export PDCS_SKIP_GPU_PRECOMPILE=1
    export JULIA_PKG_PRECOMPILE_AUTO=0

    local instance_id case_dir attempt result_path raw_log
    local project depot return_code timeout_seconds
    timeout_seconds=$(( TIME_LIMIT + ${TABLE5_MARGIN_SECONDS:-7200} ))
    for instance_id in "${instance_ids[@]}"; do
        for solver in "${solver_names[@]}"; do
            case_dir="${SOLVER_DIR}/${solver}/${instance_id}"
            if [[ -f "${case_dir}/DONE" ]]; then
                printf 'SOLVE_SKIP_DONE solver=%s id=%s\n' \
                    "${solver}" "${instance_id}"
                continue
            fi
            gpu_preflight || return 3
            attempt="${case_dir}/attempt_${RUN_ID}"
            if [[ -e "${attempt}" ]]; then
                attempt="${case_dir}/attempt_${RUN_ID}_$$"
            fi
            mkdir -p "${attempt}"
            result_path="${attempt}/result.toml"
            raw_log="${attempt}/solver.raw.log"
            project="$(solver_project "${solver}")" || return 2
            depot="$(solver_depot "${solver}")" || return 2
            [[ -f "${project}/Project.toml" ]] || {
                printf 'Solver project is missing: %s\n' "${project}" >&2
                return 2
            }
            printf 'SOLVE_CASE_START solver=%s id=%s gpu=%s utc=%s\n' \
                "${solver}" "${instance_id}" "${GPU_INDEX}" "$(timestamp)"
            timeout --signal=INT --kill-after=60 "${timeout_seconds}" \
                env \
                    "JULIA_DEPOT_PATH=${depot}" \
                    "CUDA_VISIBLE_DEVICES=${GPU_INDEX}" \
                    "PDCS_SKIP_GPU_PRECOMPILE=1" \
                    "JULIA_PKG_PRECOMPILE_AUTO=0" \
                "${JULIA_BIN}" \
                    --startup-file=no \
                    --project="${project}" \
                    "${SOLVE_SCRIPT}" \
                    --solver "${solver}" \
                    --manifest "${CONFIG}" \
                    --instance-id "${instance_id}" \
                    --generator-script "${GENERATOR}" \
                    --generator-project "${ENV_DIR}/Project.toml" \
                    --generator-manifest "${ENV_DIR}/Manifest.toml" \
                    --result "${result_path}" \
                    --tolerance "${TOLERANCE}" \
                    --time-limit "${TIME_LIMIT}" \
                    --print-frequency "${PRINT_FREQUENCY}" \
                    --verbose-level "${VERBOSE_LEVEL}" \
                    --pdcs-profile "${PDCS_PROFILE}" \
                    --required-julia-version "${REQUIRED_JULIA_VERSION}" \
                    2>&1 | tee "${raw_log}"
            return_code=${PIPESTATUS[0]}
            printf '%s\n' "${return_code}" > "${attempt}/exit_status.txt"
            nvidia-smi -i "${GPU_INDEX}" \
                --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu \
                --format=csv,noheader > "${attempt}/gpu_after.txt" 2>&1 || true
            resource_snapshot "solve_${solver}_${instance_id}"
            if (( return_code != 0 )); then
                printf 'SOLVE_CASE_FAILED solver=%s id=%s exit_code=%s log=%s\n' \
                    "${solver}" "${instance_id}" "${return_code}" "${raw_log}" >&2
                collect_kernel_diagnostics "${attempt}/kernel.log"
                return "${return_code}"
            fi
            printf '%s\n' "$(basename "${attempt}")" > "${case_dir}/DONE"
            printf 'SOLVE_CASE_COMPLETE solver=%s id=%s utc=%s\n' \
                "${solver}" "${instance_id}" "$(timestamp)"
        done
    done
    if [[ "${INSTANCES}" == "all" ]]; then
        "${REPORT_SCRIPT}" validate-runs \
            --manifest "${CONFIG}" \
            --solver-dir "${SOLVER_DIR}" \
            --solvers "${SOLVERS}" \
            --require-complete || return 1
    fi
    report_phase || return 1
    printf 'TABLE5_SOLVE_PHASE_COMPLETE cases=%s storage=no_cbf utc=%s\n' \
        "${#instance_ids[@]}" "$(timestamp)"
}

report_phase() {
    "${REPORT_SCRIPT}" report \
        --manifest "${CONFIG}" \
        --solver-dir "${SOLVER_DIR}" \
        --solvers "${SOLVERS}" \
        --markdown "${REPORT_MD}" \
        --csv "${REPORT_CSV}" \
        --solver-csv "${SOLVER_REPORT_CSV}"
}

main_status=0
case "${COMMAND}" in
    prepare)
        prepare_phase || main_status=$?
        ;;
    solve)
        solve_phase || main_status=$?
        ;;
    report)
        report_phase || main_status=$?
        ;;
    all)
        prepare_phase || main_status=$?
        if (( main_status == 0 )); then
            solve_phase || main_status=$?
        fi
        report_phase || {
            report_status=$?
            (( main_status != 0 )) || main_status=${report_status}
        }
        ;;
esac

printf 'TABLE5_DRIVER_FINISH command=%s status=%s log=%s utc=%s\n' \
    "${COMMAND}" "${main_status}" "${DRIVER_LOG}" "$(timestamp)"
exit "${main_status}"
