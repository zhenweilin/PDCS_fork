#!/usr/bin/env bash

set -uo pipefail

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$benchmark_dir/../.." && pwd)"
julia_bin="${LASSO_JULIA_BIN:-$benchmark_dir/.tools/julia-1.10.4/bin/julia}"
solver_env="${LASSO_SOLVER_ENV:-$benchmark_dir/.gpu_solver_env}"
scs_solver_env="${LASSO_SCS_SOLVER_ENV:-$benchmark_dir/.gpu_scs_env}"
config="${LASSO_CONFIG:-$benchmark_dir/lasso_table5.toml}"
result_root="${LASSO_RESULT_ROOT:-$benchmark_dir/results/manual}"
solver=""
time_limit="${LASSO_TIME_LIMIT:-3600}"
tolerance="${LASSO_TOLERANCE:-1e-6}"
replicates="${LASSO_REPLICATES:-5}"
max_scales="${LASSO_MAX_SCALES:-5}"
workers="${LASSO_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
verbose="${LASSO_VERBOSE:-1}"

usage() {
    printf '%s\n' \
        "Usage: run_gpu_campaign.sh --solver cupdcs|cuscs|cuclarabel [options]" \
        "  --result-root DIR" \
        "  --time-limit SECONDS" \
        "  --tolerance VALUE" \
        "  --replicates 1..5" \
        "  --max-scales 1..5" \
        "  --workers N" \
        "  --verbose 0..2" \
        "" \
        "CuClarabel and cuSCS stop before larger scales after any failed case" \
        "in a scale. cuPDCS always attempts every requested scale and replicate."
}

while (($#)); do
    case "$1" in
        --solver) solver="$2"; shift 2 ;;
        --result-root) result_root="$2"; shift 2 ;;
        --time-limit) time_limit="$2"; shift 2 ;;
        --tolerance) tolerance="$2"; shift 2 ;;
        --replicates) replicates="$2"; shift 2 ;;
        --max-scales) max_scales="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --verbose) verbose="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$solver" in
    cupdcs|cuscs|cuclarabel) ;;
    *) printf 'A valid --solver is required.\n' >&2; usage >&2; exit 2 ;;
esac
[[ "$time_limit" =~ ^[0-9]+$ ]] && ((time_limit > 0)) || {
    printf 'time-limit must be a positive integer.\n' >&2
    exit 2
}
[[ "$replicates" =~ ^[0-9]+$ ]] && ((replicates >= 1 && replicates <= 5)) || {
    printf 'replicates must be between 1 and 5.\n' >&2
    exit 2
}
[[ "$max_scales" =~ ^[0-9]+$ ]] && ((max_scales >= 1 && max_scales <= 5)) || {
    printf 'max-scales must be between 1 and 5.\n' >&2
    exit 2
}
[[ "$workers" =~ ^[0-9]+$ ]] && ((workers > 0)) || {
    printf 'workers must be positive.\n' >&2
    exit 2
}
[[ "$verbose" =~ ^[0-9]+$ ]] && ((verbose >= 0 && verbose <= 2)) || {
    printf 'verbose must be between 0 and 2.\n' >&2
    exit 2
}
[[ -x "$julia_bin" ]] || {
    printf 'Julia 1.10.4 is missing: %s\nRun prepare_gpu_solver_env.sh first.\n' \
        "$julia_bin" >&2
    exit 2
}
[[ -f "$solver_env/Manifest.toml" ]] || {
    printf 'Solver environment is missing: %s\nRun prepare_gpu_solver_env.sh first.\n' \
        "$solver_env" >&2
    exit 2
}
if [[ "$solver" == "cuscs" ]]; then
    [[ -f "$scs_solver_env/Manifest.toml" ]] || {
        printf 'cuSCS environment is missing: %s\nRun prepare_gpu_solver_env.sh first.\n' \
            "$scs_solver_env" >&2
        exit 2
    }
    solver_env="$scs_solver_env"
fi
[[ -f "$config" ]] || { printf 'Missing config: %s\n' "$config" >&2; exit 2; }

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || exit 2
gpu_count="$(printf '%s\n' "$gpu_names" | sed '/^[[:space:]]*$/d' | wc -l)"
[[ "$gpu_count" == "1" && "$gpu_names" == *H100* ]] || {
    printf 'This campaign requires exactly one Slurm-visible H100; found: %s\n' \
        "$gpu_names" >&2
    exit 2
}

write_process_failure() {
    local result_path="$1"
    local instance_id="$2"
    local scale="$3"
    local replicate="$4"
    local return_code="$5"
    local dimensions="${scale#m}"
    local m="${dimensions%%-n*}"
    local n="${dimensions##*-n}"
    local temporary_path="${result_path}.tmp.$$"

    # The Julia runner normally writes the case record, including ordinary
    # solver failures.  Preserve campaign completeness when the whole process
    # instead crashes or is killed before it can do so.
    {
        printf 'schema_version = 1\n'
        printf 'config = "%s"\n' "$config"
        printf 'solver = "%s"\n' "$solver"
        printf 'instance_id = "%s"\n' "$instance_id"
        printf 'm = %s\n' "$m"
        printf 'n = %s\n' "$n"
        printf 'replicate = %s\n' "$replicate"
        printf 'run_status = "failed"\n'
        printf 'termination_status = "PROCESS_ERROR"\n'
        printf 'error = "solver process exited with code %s before writing a result"\n' \
            "$return_code"
        printf 'gpu_name = "%s"\n' "$gpu_names"
        printf 'cuda_visible_devices = "%s"\n' "${CUDA_VISIBLE_DEVICES:-}"
        printf 'slurm_job_id = "%s"\n' "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
        printf 'slurm_array_task_id = "%s"\n' "${SLURM_ARRAY_TASK_ID:-}"
        printf 'tolerance = %s\n' "$tolerance"
        printf 'validation_tolerance = %s\n' "$tolerance"
        printf 'time_limit_seconds = %s.0\n' "$time_limit"
        printf 'status_accepted = false\n'
        printf 'validation_accepted = false\n'
        printf 'solver_tolerance_accepted = false\n'
        printf 'finished_utc = "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"$temporary_path"
    mv -f "$temporary_path" "$result_path"
}

mkdir -p "$result_root/$solver/logs"
printf 'LASSO_CAMPAIGN_START solver=%s gpu=%s max_scales=%s replicates=%s utc=%s\n' \
    "$solver" "$gpu_names" "$max_scales" "$replicates" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$solver" == "cupdcs" ]]; then
    cuda_root="$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)"
    artifact_dir="$result_root/artifacts/cupdcs-sm90"
    mkdir -p "$artifact_dir"
    make -C "$repo_root/src/pdcs_gpu/cuda" rebuild-gpu \
        CUDA_HOME="$cuda_root" ARCH=sm_90 OUTPUT_DIR="$artifact_dir" \
        >"$result_root/$solver/logs/build_cuda_artifacts.log" 2>&1 || {
            printf 'cuPDCS CUDA artifact build failed.\n' >&2
            exit 1
        }
    export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir"
fi

scales=(
    "m10000-n100000"
    "m70000-n700000"
    "m400000-n7000000"
    "m700000-n7000000"
    "m750000-n7500000"
)
overall_failed=0
stopped_after=""
case_timeout=$((time_limit + 1800))

for ((scale_index = 0; scale_index < max_scales; scale_index++)); do
    scale="${scales[scale_index]}"
    scale_failed=0
    printf 'LASSO_SCALE_START solver=%s scale=%s utc=%s\n' \
        "$solver" "$scale" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for ((replicate = 1; replicate <= replicates; replicate++)); do
        printf -v replicate_tag '%02d' "$replicate"
        instance_id="table5-${scale}-r${replicate_tag}"
        result_path="$result_root/$solver/${instance_id}.toml"
        raw_log="$result_root/$solver/logs/${instance_id}.log"
        printf 'LASSO_CASE_DISPATCH solver=%s id=%s utc=%s\n' \
            "$solver" "$instance_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

        timeout --signal=INT --kill-after=60 "$case_timeout" \
            env \
                JULIA_PKG_OFFLINE=true \
                JULIA_PKG_PRECOMPILE_AUTO=0 \
                JULIA_CONDAPKG_BACKEND=Null \
                JULIA_PYTHONCALL_EXE=/usr/bin/python3.12 \
                PDCS_SKIP_GPU_PRECOMPILE=1 \
            "$julia_bin" --startup-file=no --threads="$workers" \
                --project="$solver_env" \
                "$benchmark_dir/run_gpu_solver.jl" \
                --solver "$solver" \
                --config "$config" \
                --instance-id "$instance_id" \
                --result "$result_path" \
                --time-limit "$time_limit" \
                --tolerance "$tolerance" \
                --workers "$workers" \
                --verbose "$verbose" \
                >"$raw_log" 2>&1
        return_code=$?
        if ((return_code == 0)); then
            printf 'LASSO_CASE_PASS solver=%s id=%s result=%s\n' \
                "$solver" "$instance_id" "$result_path"
        else
            printf 'LASSO_CASE_FAIL solver=%s id=%s exit_code=%s log=%s\n' \
                "$solver" "$instance_id" "$return_code" "$raw_log" >&2
            if [[ ! -f "$result_path" ]]; then
                write_process_failure \
                    "$result_path" "$instance_id" "$scale" "$replicate" \
                    "$return_code"
            fi
            scale_failed=1
            overall_failed=1
        fi
    done
    printf 'LASSO_SCALE_FINISH solver=%s scale=%s failed=%s utc=%s\n' \
        "$solver" "$scale" "$scale_failed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if ((scale_failed != 0)) && [[ "$solver" != "cupdcs" ]]; then
        stopped_after="$scale"
        printf 'LASSO_EARLY_STOP solver=%s failed_scale=%s larger_scales_skipped=true\n' \
            "$solver" "$scale"
        break
    fi
done

summary="$result_root/$solver/campaign_summary.txt"
{
    printf 'solver=%s\n' "$solver"
    printf 'gpu=%s\n' "$gpu_names"
    printf 'max_scales=%s\n' "$max_scales"
    printf 'replicates=%s\n' "$replicates"
    printf 'time_limit_seconds=%s\n' "$time_limit"
    printf 'tolerance=%s\n' "$tolerance"
    printf 'overall_failed=%s\n' "$overall_failed"
    printf 'stopped_after_scale=%s\n' "$stopped_after"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$summary"
printf 'LASSO_CAMPAIGN_FINISH solver=%s failed=%s summary=%s\n' \
    "$solver" "$overall_failed" "$summary"
exit "$overall_failed"
