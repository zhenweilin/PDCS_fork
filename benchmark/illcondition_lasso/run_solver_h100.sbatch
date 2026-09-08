#!/usr/bin/env bash
#SBATCH --job-name=illlasso-gpu
#SBATCH --account=p53143
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=benchmark/illcondition_lasso/results/current/slurm_logs/job-%j.out
#SBATCH --error=benchmark/illcondition_lasso/results/current/slurm_logs/job-%j.out

set -uo pipefail

repo_root="${PDCS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
solver="${ILL_LASSO_SOLVER:?ILL_LASSO_SOLVER is required}"
manifest="${ILL_LASSO_MANIFEST:?ILL_LASSO_MANIFEST is required}"
result_root="${ILL_LASSO_RESULT_ROOT:?ILL_LASSO_RESULT_ROOT is required}"
tolerance="${ILL_LASSO_TOLERANCE:-1e-6}"
time_limit="${ILL_LASSO_TIME_LIMIT:-3600}"
[[ -n "$repo_root" ]] || { printf 'PDCS_REPO_ROOT is required.\n' >&2; exit 2; }
repo_root="$(cd -- "$repo_root" && pwd)"
benchmark_dir="$repo_root/benchmark/illcondition_lasso"
large_lasso_dir="$repo_root/benchmark/large_scale_lasso"
julia_bin="$large_lasso_dir/.tools/julia-1.10.4/bin/julia"
artifact_dir="${PDCS_CUDA_PROJECTION_ARTIFACT_DIR:-$large_lasso_dir/results/current/artifacts/cupdcs-sm90}"
workers="${SLURM_CPUS_PER_TASK:-16}"

case "$solver" in
    cupdcs|cuclarabel) julia_project="$large_lasso_dir/.gpu_solver_env" ;;
    cuscs) julia_project="$large_lasso_dir/.gpu_scs_env" ;;
    *) printf 'Unsupported solver: %s\n' "$solver" >&2; exit 2 ;;
esac
[[ "$tolerance" == "1e-6" ]] || {
    printf 'Formal tolerance must be 1e-6, got %s.\n' "$tolerance" >&2
    exit 2
}
[[ "$time_limit" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ILL_LASSO_TIME_LIMIT must be a positive integer.\n' >&2
    exit 2
}
[[ -x "$julia_bin" && -f "$julia_project/Manifest.toml" ]] || {
    printf 'Missing required large-scale-lasso Julia environment.\n' >&2
    exit 2
}
[[ -f "$manifest" ]] || { printf 'Missing manifest: %s\n' "$manifest" >&2; exit 2; }

module purge
if [[ "$solver" != "cuscs" ]]; then
    module load cuda/12.6.2-gcc-12.4.0
fi
[[ "$($julia_bin --startup-file=no --version)" == "julia version 1.10.4" ]] || {
    printf 'Formal campaign requires the shared Julia 1.10.4 binary.\n' >&2
    exit 2
}
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || exit 2
gpu_count="$(printf '%s\n' "$gpu_names" | sed '/^[[:space:]]*$/d' | wc -l)"
[[ "$gpu_count" == "1" && "$gpu_names" == *H100* ]] || {
    printf 'Expected exactly one Slurm-visible H100, found: %s\n' "$gpu_names" >&2
    exit 2
}

if [[ "$solver" == "cupdcs" ]]; then
    for artifact in \
        libfew_block_proj.so \
        moderate_block_proj.ptx \
        sufficient_block_proj.ptx \
        massive_block_proj.ptx \
        utils.ptx
    do
        [[ -s "$artifact_dir/$artifact" ]] || {
            printf 'Missing H100 cuPDCS artifact: %s\n' "$artifact_dir/$artifact" >&2
            exit 2
        }
    done
    export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir"
    export PDCS_GRIDWISE_MODE="${PDCS_GRIDWISE_MODE:-native}"
    export PDCS_GRIDWISE_STRICT="${PDCS_GRIDWISE_STRICT:-1}"
    export PDCS_GRIDWISE_SELFTEST="${PDCS_GRIDWISE_SELFTEST:-1}"
    export PDCS_CUBLAS_REPRODUCIBLE="${PDCS_CUBLAS_REPRODUCIBLE:-1}"
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3.12
export PDCS_SKIP_GPU_PRECOMPILE=1

solver_dir="$result_root/$solver"
mkdir -p "$solver_dir/logs" "$result_root/gpu_status" "$result_root/slurm_logs"
gpu_log="$result_root/gpu_status/${solver}-job-${SLURM_JOB_ID}.csv"
nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -l 300 >"$gpu_log" 2>&1 &
gpu_monitor_pid=$!
cleanup_gpu_monitor() {
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
}
trap cleanup_gpu_monitor EXIT

mapfile -t instance_ids < <(
    "$julia_bin" --startup-file=no --project="$julia_project" \
        -e 'using TOML; for entry in TOML.parsefile(ARGS[1])["instances"]; println(entry["id"]); end' \
        "$manifest"
)
[[ "${#instance_ids[@]}" == "6" ]] || {
    printf 'Expected six instances, found %s.\n' "${#instance_ids[@]}" >&2
    exit 2
}

printf 'ILL_LASSO_CAMPAIGN_START solver=%s job=%s gpu=%s julia=1.10.4 project=%s manifest=%s utc=%s\n' \
    "$solver" "$SLURM_JOB_ID" "$gpu_names" "$julia_project" "$manifest" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
overall_exit=0
completed_cases=0
case_timeout=$((time_limit + 1800))
for instance_id in "${instance_ids[@]}"; do
    result_path="$solver_dir/${instance_id}.toml"
    if [[ -f "$result_path" ]] && "$julia_bin" --startup-file=no \
        --project="$julia_project" -e \
        'using TOML; value=TOML.parsefile(ARGS[1]); manifest=TOML.parsefile(ARGS[4]); entries=filter(e->e["id"]==ARGS[3],manifest["instances"]); good=length(entries)==1 && get(value,"run_status","runtime_error")!="runtime_error" && get(value,"solver","")==ARGS[2] && get(value,"instance_id","")==ARGS[3] && get(value,"matrix_sha256","")==only(entries)["matrix_sha256"] && get(value,"tolerance",NaN)==1e-6 && get(value,"time_limit_seconds",NaN)==3600.0; exit(good ? 0 : 1)' \
        "$result_path" "$solver" "$instance_id" "$manifest"
    then
        printf 'ILL_LASSO_CASE_SKIP solver=%s instance=%s result=%s\n' \
            "$solver" "$instance_id" "$result_path"
        ((completed_cases += 1))
        continue
    fi
    driver_log="$solver_dir/logs/${instance_id}_driver_slurm_${SLURM_JOB_ID}.log"
    solver_log="$solver_dir/logs/${instance_id}_solver_slurm_${SLURM_JOB_ID}.log"
    printf 'ILL_LASSO_CASE_START solver=%s instance=%s utc=%s\n' \
        "$solver" "$instance_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    timeout --signal=INT --kill-after=300 "$case_timeout" \
        "$julia_bin" --startup-file=no --threads="$workers" \
            --project="$julia_project" "$benchmark_dir/solve_case.jl" \
            --solver "$solver" --manifest "$manifest" \
            --instance-id "$instance_id" --output "$result_path" \
            --solver-log "$solver_log" --pdcs-root "$repo_root" \
            --tolerance "$tolerance" --time-limit "$time_limit" \
            --verbose 2 >"$driver_log" 2>&1
    return_code=$?
    if ((return_code == 0)); then
        ((completed_cases += 1))
    else
        overall_exit=1
    fi
    printf 'ILL_LASSO_CASE_FINISH solver=%s instance=%s exit_code=%s utc=%s\n' \
        "$solver" "$instance_id" "$return_code" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

summary_tmp="$solver_dir/campaign_summary.toml.tmp.${SLURM_JOB_ID}"
summary="$solver_dir/campaign_summary.toml"
{
    printf 'schema_version = 1\n'
    printf 'solver = "%s"\n' "$solver"
    printf 'julia_version = "1.10.4"\n'
    printf 'julia_executable = "%s"\n' "$julia_bin"
    printf 'julia_project = "%s"\n' "$julia_project"
    printf 'manifest = "%s"\n' "$manifest"
    printf 'gpu = "%s"\n' "$gpu_names"
    printf 'case_count = 6\n'
    printf 'completed_cases = %s\n' "$completed_cases"
    printf 'time_limit_seconds_per_case = %s\n' "$time_limit"
    printf 'tolerance = %s\n' "$tolerance"
    printf 'slurm_job_id = "%s"\n' "$SLURM_JOB_ID"
    printf 'exit_code = %s\n' "$overall_exit"
    printf 'finished_utc = "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$summary_tmp"
mv "$summary_tmp" "$summary"
printf 'ILL_LASSO_CAMPAIGN_FINISH solver=%s job=%s completed=%s exit_code=%s summary=%s utc=%s\n' \
    "$solver" "$SLURM_JOB_ID" "$completed_cases" "$overall_exit" "$summary" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$overall_exit"
