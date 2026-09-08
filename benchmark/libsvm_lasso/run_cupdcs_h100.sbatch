#!/usr/bin/env bash
#SBATCH --job-name=libsvm-cupdcs-h100
#SBATCH --account=p53143
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=benchmark/libsvm_lasso/results/current/cupdcs/slurm_logs/job-%j.out
#SBATCH --error=benchmark/libsvm_lasso/results/current/cupdcs/slurm_logs/job-%j.out

set -uo pipefail

repo_root="${PDCS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$repo_root" ]] || {
    printf 'PDCS_REPO_ROOT or SLURM_SUBMIT_DIR is required.\n' >&2
    exit 2
}
repo_root="$(cd -- "$repo_root" && pwd)"
benchmark_dir="$repo_root/benchmark/libsvm_lasso"
large_lasso_dir="$repo_root/benchmark/large_scale_lasso"
julia_bin="$large_lasso_dir/.tools/julia-1.10.4/bin/julia"
julia_project="$large_lasso_dir/.gpu_solver_env"
raw_dir="${LIBSVM_RAW_DIR:-$benchmark_dir/raw}"
result_root="${LIBSVM_RESULT_ROOT:-$benchmark_dir/results/current/cupdcs}"
artifact_dir="${PDCS_CUDA_PROJECTION_ARTIFACT_DIR:-$large_lasso_dir/results/current/artifacts/cupdcs-sm90}"
time_limit="${LIBSVM_TIME_LIMIT:-3600}"
tolerance="${LIBSVM_TOLERANCE:-1e-6}"
workers="${SLURM_CPUS_PER_TASK:-16}"

[[ "$time_limit" =~ ^[1-9][0-9]*$ ]] || {
    printf 'LIBSVM_TIME_LIMIT must be a positive integer number of seconds.\n' >&2
    exit 2
}
[[ "$tolerance" == "1e-6" ]] || {
    printf 'This formal campaign requires LIBSVM_TOLERANCE=1e-6, got %s.\n' \
        "$tolerance" >&2
    exit 2
}
[[ -x "$julia_bin" ]] || {
    printf 'Missing large-scale-lasso Julia executable: %s\n' "$julia_bin" >&2
    exit 2
}
[[ -f "$julia_project/Manifest.toml" ]] || {
    printf 'Missing large-scale-lasso GPU environment: %s\n' "$julia_project" >&2
    exit 2
}
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

module purge
module load cuda/12.6.2-gcc-12.4.0

julia_version="$($julia_bin --startup-file=no --version)" || exit 2
[[ "$julia_version" == "julia version 1.10.4" ]] || {
    printf 'Expected the large-scale-lasso Julia 1.10.4 binary, got: %s\n' \
        "$julia_version" >&2
    exit 2
}
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || exit 2
gpu_count="$(printf '%s\n' "$gpu_names" | sed '/^[[:space:]]*$/d' | wc -l)"
[[ "$gpu_count" == "1" && "$gpu_names" == *H100* ]] || {
    printf 'Expected exactly one Slurm-visible H100, found: %s\n' "$gpu_names" >&2
    exit 2
}

LIBSVM_RAW_DIR="$raw_dir" "$benchmark_dir/download_default_datasets.sh" verify || exit 2
mkdir -p "$result_root/logs" "$result_root/gpu_status" "$result_root/slurm_logs"

export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir"
export PDCS_GRIDWISE_MODE="${PDCS_GRIDWISE_MODE:-native}"
export PDCS_GRIDWISE_STRICT="${PDCS_GRIDWISE_STRICT:-1}"
export PDCS_GRIDWISE_SELFTEST="${PDCS_GRIDWISE_SELFTEST:-1}"
export PDCS_CUBLAS_REPRODUCIBLE="${PDCS_CUBLAS_REPRODUCIBLE:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3.12
export PDCS_SKIP_GPU_PRECOMPILE=1

gpu_log="$result_root/gpu_status/job-${SLURM_JOB_ID}.csv"
nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
    -l 300 >"$gpu_log" 2>&1 &
gpu_monitor_pid=$!
cleanup_gpu_monitor() {
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
}
trap cleanup_gpu_monitor EXIT

printf 'LIBSVM_CUPDCS_CAMPAIGN_START job=%s host=%s gpu=%s julia=%s project=%s utc=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$gpu_names" "$julia_version" \
    "$julia_project" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'LIBSVM_CUPDCS_SETTINGS datasets=3 alphas=9 per_solve_time_limit=%s tolerance=%s modeling=bulk index_type=int32\n' \
    "$time_limit" "$tolerance"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader

datasets=(news20 E2006-log1p rcv1-train)
overall_exit=0
completed_datasets=0
driver_timeout=$((9 * time_limit + 3600))

for dataset in "${datasets[@]}"; do
    driver_log="$result_root/logs/${dataset}_driver_slurm_${SLURM_JOB_ID}.log"
    printf 'LIBSVM_DATASET_RUN_START dataset=%s log=%s utc=%s\n' \
        "$dataset" "$driver_log" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    timeout --signal=INT --kill-after=300 "$driver_timeout" \
        "$julia_bin" \
            --startup-file=no \
            --threads="$workers" \
            --project="$julia_project" \
            "$benchmark_dir/run_penalty_sweep.jl" \
            --dataset "$dataset" \
            --mode pdcs-gpu \
            --modeling bulk \
            --workers "$workers" \
            --pdcs-root "$repo_root" \
            --raw-dir "$raw_dir" \
            --output-dir "$result_root" \
            --compact-zero-columns true \
            --index-type int32 \
            --time-limit "$time_limit" \
            --rel-tol "$tolerance" \
            --abs-tol "$tolerance" \
            --verbose 2 \
            --resume true \
            --run-tag "slurm-${SLURM_JOB_ID}" \
            >"$driver_log" 2>&1
    return_code=$?
    if ((return_code == 0)); then
        ((completed_datasets += 1))
    else
        overall_exit=1
    fi
    printf 'LIBSVM_DATASET_RUN_FINISH dataset=%s exit_code=%s utc=%s\n' \
        "$dataset" "$return_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

summary_tmp="$result_root/campaign_summary.toml.tmp.${SLURM_JOB_ID}"
summary="$result_root/campaign_summary.toml"
{
    printf 'schema_version = 1\n'
    printf 'solver = "cupdcs"\n'
    printf 'gpu = "%s"\n' "$gpu_names"
    printf 'julia_version = "1.10.4"\n'
    printf 'julia_executable = "%s"\n' "$julia_bin"
    printf 'julia_project = "%s"\n' "$julia_project"
    printf 'modeling = "bulk"\n'
    printf 'index_type = "int32"\n'
    printf 'dataset_count = 3\n'
    printf 'alpha_count_per_dataset = 9\n'
    printf 'completed_dataset_processes = %s\n' "$completed_datasets"
    printf 'time_limit_seconds_per_solve = %s\n' "$time_limit"
    printf 'relative_tolerance = %s\n' "$tolerance"
    printf 'absolute_tolerance = %s\n' "$tolerance"
    printf 'slurm_job_id = "%s"\n' "$SLURM_JOB_ID"
    printf 'exit_code = %s\n' "$overall_exit"
    printf 'finished_utc = "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$summary_tmp"
mv "$summary_tmp" "$summary"

nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader
printf 'LIBSVM_CUPDCS_CAMPAIGN_FINISH job=%s completed=%s exit_code=%s summary=%s utc=%s\n' \
    "$SLURM_JOB_ID" "$completed_datasets" "$overall_exit" "$summary" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$overall_exit"
