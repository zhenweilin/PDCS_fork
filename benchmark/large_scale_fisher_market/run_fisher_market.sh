#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${BASE_DIR}/../.." && pwd)"

LOCAL_JULIA="${ROOT_DIR}/benchmark/large_scale_lasso/.tools/julia-1.10.4/bin/julia"
JULIA_BIN="${JULIA_BIN:-${LOCAL_JULIA}}"
CONFIG="${CONFIG:-${BASE_DIR}/fisher_market_cases.toml}"
SOLVE_SCRIPT="${BASE_DIR}/solve_fisher_instance.jl"
REPORT_SCRIPT="${BASE_DIR}/fisher_market_report.jl"
DRIVER_TIMEOUT_SCRIPT="${BASE_DIR}/write_driver_timeout_result.jl"
RESULT_ROOT="${RESULT_ROOT:-${BASE_DIR}/results}"
REQUIRED_JULIA_VERSION="${REQUIRED_JULIA_VERSION:-1.10.4}"
GPU_INDEX="${GPU_INDEX:-0}"
SOLVERS="${SOLVERS:-cupdcs,scs_gpu,cuclarabel}"
INSTANCES="${INSTANCES:-all}"
TIME_LIMIT="${TIME_LIMIT:-18000}"
SETUP_GRACE="${SETUP_GRACE:-600}"
TOLERANCE="${TOLERANCE:-1e-6}"
PRINT_FREQUENCY="${PRINT_FREQUENCY:-1000}"
VERBOSE_LEVEL="${VERBOSE_LEVEL:-2}"
RERUN="${RERUN:-0}"
RUN_ID="${RUN_ID_OVERRIDE:-$(date -u +%Y%m%dT%H%M%SZ)}"

FISHER_DEPOT="${FISHER_DEPOT:-${BASE_DIR}/.julia-depot}"
PYTHONCALL_EXE="${JULIA_PYTHONCALL_EXE:-/usr/bin/python3.12}"
PDCS_DEPOT="${PDCS_DEPOT:-${FISHER_DEPOT}}"
CUPDCS_PROJECT="${CUPDCS_PROJECT:-${BASE_DIR}/.envs/cupdcs}"
SCS_GPU_PROJECT="${SCS_GPU_PROJECT:-${BASE_DIR}/.envs/scs_gpu}"
CUCLARABEL_PROJECT="${CUCLARABEL_PROJECT:-${BASE_DIR}/.envs/cuclarabel}"
SCS_GPU_DEPOT="${SCS_GPU_DEPOT:-${FISHER_DEPOT}}"
CUCLARABEL_DEPOT="${CUCLARABEL_DEPOT:-${FISHER_DEPOT}}"

usage() {
    cat <<EOF
Usage: $(basename "$0") COMMAND [options]

Commands:
  prepare  Validate the manifest, environments, GPU, disk, and no-data policy.
  smoke    Run the small correctness instance with all selected GPU solvers.
  solve    Run the 30 formal cases (six sizes, five seeds per size).
  report   Summarize existing result TOMLs without loading problem data.
  all      Run prepare, smoke, formal solve, and report.

Options:
  --gpu INDEX
  --solvers cupdcs,scs_gpu,cuclarabel
  --instances all|id1,id2
  --time-limit SECONDS
  --setup-grace SECONDS
  --tolerance VALUE
  --print-frequency ITERATIONS
  --verbose-level 0|1|2
  --result-root PATH
  --rerun

Each solver/case runs in a fresh Julia process. The instance is regenerated
deterministically in memory, passed directly to the solver source API, solved,
and released. Only a compact result TOML and the solver stdout/stderr log are
retained. No CBF, JLD2, NPZ, or matrix data file is created.
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
        --gpu) GPU_INDEX="$2"; shift 2 ;;
        --solvers) SOLVERS="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --time-limit) TIME_LIMIT="$2"; shift 2 ;;
        --setup-grace) SETUP_GRACE="$2"; shift 2 ;;
        --tolerance) TOLERANCE="$2"; shift 2 ;;
        --print-frequency) PRINT_FREQUENCY="$2"; shift 2 ;;
        --verbose-level) VERBOSE_LEVEL="$2"; shift 2 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --rerun) RERUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${COMMAND}" in
    prepare|smoke|solve|report|all) ;;
    *) printf 'Unknown command: %s\n' "${COMMAND}" >&2; exit 2 ;;
esac

for value in \
    "${GPU_INDEX}" "${TIME_LIMIT}" "${SETUP_GRACE}" \
    "${PRINT_FREQUENCY}" "${VERBOSE_LEVEL}" "${RERUN}"
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
(( VERBOSE_LEVEL <= 2 )) || {
    printf 'verbose-level must be 0, 1, or 2\n' >&2
    exit 2
}
[[ "${TOLERANCE}" =~ ^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$ ]] || {
    printf 'Invalid tolerance: %s\n' "${TOLERANCE}" >&2
    exit 2
}
[[ -f "${CONFIG}" && -f "${SOLVE_SCRIPT}" ]] || {
    printf 'Missing Fisher manifest or solve script in %s\n' "${BASE_DIR}" >&2
    exit 2
}

solver_project() {
    case "$1" in
        cupdcs) printf '%s\n' "${CUPDCS_PROJECT}" ;;
        scs_gpu) printf '%s\n' "${SCS_GPU_PROJECT}" ;;
        cuclarabel) printf '%s\n' "${CUCLARABEL_PROJECT}" ;;
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

validate_solver_list() {
    local solver
    IFS=, read -r -a requested_solvers <<< "${SOLVERS}"
    ((${#requested_solvers[@]} > 0)) || return 1
    for solver in "${requested_solvers[@]}"; do
        case "${solver}" in
            cupdcs|scs_gpu|cuclarabel) ;;
            *) printf 'Unknown solver: %s\n' "${solver}" >&2; return 1 ;;
        esac
    done
}

check_no_problem_data() {
    local count
    count="$(
        find "${BASE_DIR}" -type f \
            \( -iname '*.cbf' -o -iname '*.cbf.gz' -o \
               -iname '*.jld2' -o -iname '*.npz' -o \
               -iname '*.mtx' -o -iname '*.mps' \) \
            -print | wc -l
    )"
    [[ "${count}" == "0" ]] || {
        printf 'Persistent Fisher problem-data files found:\n' >&2
        find "${BASE_DIR}" -type f \
            \( -iname '*.cbf' -o -iname '*.cbf.gz' -o \
               -iname '*.jld2' -o -iname '*.npz' -o \
               -iname '*.mtx' -o -iname '*.mps' \) \
            -print >&2
        return 1
    }
    printf 'FISHER_NO_PERSISTED_DATA verified=1\n'
}

prepare() {
    local version formal_count smoke_count gpu_state gpu_names gpu_count
    version="$("${JULIA_BIN}" --version 2>&1)" || return 1
    printf 'JULIA_VERSION %s\n' "${version}"
    [[ "${version}" == "julia version ${REQUIRED_JULIA_VERSION}" ]] || {
        printf 'Expected Julia %s\n' "${REQUIRED_JULIA_VERSION}" >&2
        return 1
    }
    validate_solver_list || return 1
    read -r formal_count smoke_count < <(
        "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" -e '
            using TOML
            manifest = TOML.parsefile(ARGS[1])
            @assert manifest["replicates"] == 5
            @assert manifest["julia_version"] == "1.10.4"
            instances = manifest["instances"]
            @assert length(instances) == 30
            @assert length(unique(entry["id"] for entry in instances)) == 30
            for (m, n) in (
                (100, 5000),
                (100000, 1000),
                (150000, 1000),
                (200000, 1000),
                (250000, 1000),
                (280000, 1000),
            )
                @assert count(
                    entry -> entry["m"] == m && entry["n"] == n,
                    instances,
                ) == 5
            end
            println(length(instances), " ",
                    length(manifest["smoke_instances"]))
        ' "${CONFIG}"
    ) || return 1
    printf 'FISHER_MANIFEST_VALID formal=%s smoke=%s\n' \
        "${formal_count}" "${smoke_count}"
    check_no_problem_data || return 1
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)" ||
        return 1
    gpu_count="$(printf '%s\n' "${gpu_names}" | sed '/^[[:space:]]*$/d' | wc -l)"
    [[ "${gpu_count}" == "1" && "${gpu_names}" == *H100* ]] || {
        printf 'Expected exactly one H100, found: %s\n' "${gpu_names}" >&2
        return 1
    }
    export FISHER_GPU_NAME="${gpu_names}"
    gpu_state="$(
        nvidia-smi \
            --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
    )" || return 1
    printf 'GPU_PREFLIGHT %s\n' "${gpu_state}"
    df -h "${BASE_DIR}"
    return 0
}

list_instances() {
    local group="$1"
    "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" -e '
        using TOML
        manifest = TOML.parsefile(ARGS[1])
        entries = manifest[ARGS[2]]
        requested = ARGS[3]
        allowed = requested == "all" ? nothing : Set(split(requested, ","))
        selected = [
            entry["id"] for entry in entries
            if allowed === nothing || entry["id"] in allowed
        ]
        if allowed !== nothing && Set(selected) != allowed
            missing = sort!(collect(setdiff(allowed, Set(selected))))
            error("unknown instance IDs: " * join(missing, ","))
        end
        println(join(selected, "\n"))
    ' "${CONFIG}" "${group}" "${INSTANCES}"
}

result_status() {
    local path="$1"
    "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" -e '
        using TOML
        result = TOML.parsefile(ARGS[1])
        print(
            get(result, "run_status", "MISSING"), "|",
            get(result, "termination_status", "MISSING"),
        )
    ' "${path}"
}

successful_status() {
    case "$1" in
        passed\|OPTIMAL|passed\|ALMOST_OPTIMAL) return 0 ;;
        *) return 1 ;;
    esac
}

formal_scale_key() {
    local instance_id="$1"
    if [[ "${instance_id}" =~ ^fisher-m([0-9]+)-n([0-9]+)-r[0-9]+$ ]]; then
        printf 'm%s-n%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return 0
    fi
    printf '%s\n' "${instance_id}"
}

run_one() {
    local mode="$1"
    local solver="$2"
    local instance_id="$3"
    local case_dir="${RESULT_ROOT}/${mode}/${solver}/${instance_id}"
    local project depot attempt attempt_name result raw_log external_limit status
    if [[ -f "${case_dir}/DONE" && "${RERUN}" == "0" ]]; then
        attempt_name="$(<"${case_dir}/DONE")"
        result="${case_dir}/${attempt_name}/result.toml"
        if [[ -f "${result}" ]]; then
            status="$(result_status "${result}")"
            printf 'FISHER_SKIP_DONE mode=%s solver=%s instance=%s status=%s\n' \
                "${mode}" "${solver}" "${instance_id}" \
                "${status}"
            successful_status "${status}" && return 0
            return 3
        fi
    fi

    project="$(solver_project "${solver}")" || return 2
    depot="$(solver_depot "${solver}")" || return 2
    [[ -f "${project}/Project.toml" ]] || {
        printf 'Missing solver project: %s\n' "${project}" >&2
        return 2
    }
    attempt_name="attempt_${RUN_ID}"
    attempt="${case_dir}/${attempt_name}"
    [[ ! -e "${attempt}" ]] || attempt="${case_dir}/${attempt_name}_$$"
    attempt_name="${attempt##*/}"
    mkdir -p "${attempt}"
    result="${attempt}/result.toml"
    raw_log="${attempt}/solver.raw.log"
    external_limit=$((TIME_LIMIT + SETUP_GRACE))
    printf 'FISHER_CASE_START mode=%s solver=%s instance=%s gpu=%s utc=%s\n' \
        "${mode}" "${solver}" "${instance_id}" "${GPU_INDEX}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    timeout --signal=INT --kill-after=60 "${external_limit}" \
        env \
            "JULIA_DEPOT_PATH=${depot}" \
            "CUDA_VISIBLE_DEVICES=${GPU_INDEX}" \
            "FISHER_GPU_NAME=${FISHER_GPU_NAME:-}" \
            "PDCS_SKIP_GPU_PRECOMPILE=1" \
            "JULIA_PKG_PRECOMPILE_AUTO=0" \
            "JULIA_CONDAPKG_BACKEND=Null" \
            "JULIA_PYTHONCALL_EXE=${PYTHONCALL_EXE}" \
        "${JULIA_BIN}" \
            --startup-file=no \
            --project="${project}" \
            "${SOLVE_SCRIPT}" \
            --solver "${solver}" \
            --manifest "${CONFIG}" \
            --instance-id "${instance_id}" \
            --result "${result}" \
            --tolerance "${TOLERANCE}" \
            --time-limit "${TIME_LIMIT}" \
            --print-frequency "${PRINT_FREQUENCY}" \
            --verbose-level "${VERBOSE_LEVEL}" \
            --required-julia-version "${REQUIRED_JULIA_VERSION}" \
            2>&1 | tee "${raw_log}"
    local return_code=${PIPESTATUS[0]}
    printf '%s\n' "${return_code}" > "${attempt}/exit_status.txt"
    nvidia-smi \
        --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader > "${attempt}/gpu_after.txt" 2>&1 || true
    if [[ ! -f "${result}" && "${return_code}" == "124" ]]; then
        "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" \
            "${DRIVER_TIMEOUT_SCRIPT}" \
            --solver "${solver}" \
            --manifest "${CONFIG}" \
            --instance-id "${instance_id}" \
            --raw-log "${raw_log}" \
            --result "${result}" \
            --time-limit "${TIME_LIMIT}" \
            --setup-grace "${SETUP_GRACE}" \
            --tolerance "${TOLERANCE}" \
            --gpu "${GPU_INDEX}" \
            --verbose-level "${VERBOSE_LEVEL}" || true
    fi

    if [[ -f "${result}" ]]; then
        printf '%s\n' "${attempt_name}" > "${case_dir}/DONE"
        printf 'FISHER_CASE_FINISH mode=%s solver=%s instance=%s status=%s exit=%s\n' \
            "${mode}" "${solver}" "${instance_id}" \
            "$(result_status "${result}")" "${return_code}"
    else
        printf 'FISHER_CASE_NO_RESULT mode=%s solver=%s instance=%s exit=%s\n' \
            "${mode}" "${solver}" "${instance_id}" "${return_code}" >&2
    fi
    return "${return_code}"
}

run_group() {
    local mode="$1"
    local manifest_group="$2"
    local solver instance_id scale_key stop_key return_code failures=0
    local -A failed_scales=()
    mapfile -t selected_instances < <(list_instances "${manifest_group}") ||
        return 2
    validate_solver_list || return 2
    for instance_id in "${selected_instances[@]}"; do
        for solver in "${requested_solvers[@]}"; do
            scale_key="$(formal_scale_key "${instance_id}")"
            stop_key="${solver}|${scale_key}"
            if [[ "${mode}" == "formal" &&
                  "${solver}" != "cupdcs" &&
                  "${failed_scales[$stop_key]+present}" == "present" ]]
            then
                printf 'FISHER_SKIP_SCALE_FAILURE solver=%s instance=%s scale=%s\n' \
                    "${solver}" "${instance_id}" "${scale_key}"
                continue
            fi
            run_one "${mode}" "${solver}" "${instance_id}"
            return_code=$?
            if (( return_code != 0 )); then
                failures=$((failures + 1))
                if [[ "${mode}" == "formal" &&
                      "${solver}" != "cupdcs" ]]
                then
                    failed_scales["${stop_key}"]=1
                    printf 'FISHER_EARLY_STOP_SCALE solver=%s scale=%s after=%s exit=%s\n' \
                        "${solver}" "${scale_key}" "${instance_id}" \
                        "${return_code}"
                fi
            fi
        done
    done
    check_no_problem_data || return 1
    (( failures == 0 )) || {
        printf 'FISHER_GROUP_COMPLETED_WITH_FAILURES count=%s\n' "${failures}" >&2
        return 3
    }
}

report() {
    [[ -f "${REPORT_SCRIPT}" ]] || {
        printf 'Missing report script: %s\n' "${REPORT_SCRIPT}" >&2
        return 2
    }
    "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" \
        "${REPORT_SCRIPT}" \
        --manifest "${CONFIG}" \
        --result-root "${RESULT_ROOT}" \
        --output "${BASE_DIR}/fisher_market_report.md"
}

case "${COMMAND}" in
    prepare)
        prepare
        ;;
    smoke)
        prepare && run_group smoke smoke_instances
        ;;
    solve)
        prepare && run_group formal instances
        ;;
    report)
        report
        ;;
    all)
        prepare &&
        run_group smoke smoke_instances &&
        run_group formal instances
        formal_status=$?
        report
        report_status=$?
        (( formal_status == 0 && report_status == 0 ))
        ;;
esac
