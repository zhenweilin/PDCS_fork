#!/usr/bin/env bash
#SBATCH --job-name=cupdcs-represent
#SBATCH --account=p53143
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=benchmark/represent_data/results/current/slurm_logs/cupdcs-%j.out
#SBATCH --error=benchmark/represent_data/results/current/slurm_logs/cupdcs-%j.out

set -uo pipefail

repo_root="${PDCS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_root" ]]; then
    local_benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd -- "$local_benchmark_dir/../.." && pwd)"
fi
repo_root="$(cd -- "$repo_root" && pwd)"
benchmark_dir="$repo_root/benchmark/represent_data"
input_root="${CBF_INPUT_ROOT:-$repo_root/represent_data}"
result_root="${CBF_RESULT_ROOT:-$benchmark_dir/results/current}"
case_list="${CBF_CASE_LIST:-$result_root/cases.txt}"
expected_cases="${CBF_EXPECTED_CASES:-62}"
time_limit="${CBF_TIME_LIMIT:-3600}"
tolerance="${CBF_TOLERANCE:-1e-6}"
rerun="${CBF_RERUN:-false}"
julia_bin="${CBF_JULIA_BIN:-$repo_root/benchmark/large_scale_lasso/.tools/julia-1.10.4/bin/julia}"
julia_project="${CBF_CUPDCS_PROJECT:-$repo_root/benchmark/large_scale_fisher_market/.envs/cupdcs}"
julia_depot="${CBF_JULIA_DEPOT:-$repo_root/benchmark/large_scale_fisher_market/.julia-depot}"
artifact_dir="${PDCS_CUDA_PROJECTION_ARTIFACT_DIR:-$result_root/artifacts/cupdcs-sm90}"

[[ "$time_limit" == "3600" || "$time_limit" == "3600.0" ]] || {
    printf 'Each CBF time limit must be 3600 seconds; got %s\n' "$time_limit" >&2
    exit 2
}
[[ "$tolerance" == "1e-6" || "$tolerance" == "1.0e-6" ]] || {
    printf 'Tolerance must be exactly 1e-6; got %s\n' "$tolerance" >&2
    exit 2
}
[[ -x "$julia_bin" ]] || { printf 'Missing Julia: %s\n' "$julia_bin" >&2; exit 2; }
[[ -f "$benchmark_dir/run_cupdcs_batch.jl" ]] || {
    printf 'Missing cuPDCS batch driver: %s\n' "$benchmark_dir/run_cupdcs_batch.jl" >&2
    exit 2
}
[[ -f "$julia_project/Project.toml" ]] || {
    printf 'Missing cuPDCS Julia project: %s\n' "$julia_project" >&2
    exit 2
}
[[ -s "$case_list" ]] || { printf 'Missing case list: %s\n' "$case_list" >&2; exit 2; }

module purge
module load cuda/12.6.2-gcc-12.4.0

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || exit 2
gpu_count="$(printf '%s\n' "$gpu_names" | sed '/^[[:space:]]*$/d' | wc -l)"
[[ "$gpu_count" == "1" && "$gpu_names" == *H100* ]] || {
    printf 'Expected exactly one Slurm-visible H100; found: %s\n' "$gpu_names" >&2
    exit 2
}

mkdir -p "$result_root/slurm_logs" "$artifact_dir"
make -C "$repo_root/src/pdcs_gpu/cuda" rebuild-gpu \
    CUDA_HOME="$CUDA_HOME" ARCH=sm_90 OUTPUT_DIR="$artifact_dir"
for artifact in \
    libfew_block_proj.so \
    moderate_block_proj.ptx \
    sufficient_block_proj.ptx \
    massive_block_proj.ptx \
    utils.ptx
do
    [[ -s "$artifact_dir/$artifact" ]] || {
        printf 'Missing production CUDA artifact: %s\n' "$artifact_dir/$artifact" >&2
        exit 2
    }
done

export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir"
export PDCS_GRIDWISE_MODE=native
export PDCS_GRIDWISE_STRICT=1
export PDCS_GRIDWISE_SELFTEST=1
export PDCS_CUBLAS_REPRODUCIBLE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PDCS_SKIP_GPU_PRECOMPILE=1
export JULIA_DEPOT_PATH="$julia_depot"
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3.12

printf 'CUPDCS_CBF_JOB_START job=%s cases=%s gpu=%s tolerance=%s per_case_time_limit=%s utc=%s\n' \
    "$SLURM_JOB_ID" "$expected_cases" "$gpu_names" "$tolerance" "$time_limit" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$julia_bin" --version
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader

"$julia_bin" \
    --startup-file=no \
    --threads="${SLURM_CPUS_PER_TASK:-16}" \
    --project="$julia_project" \
    "$benchmark_dir/run_cupdcs_batch.jl" \
    --input-root "$input_root" \
    --case-list "$case_list" \
    --result-root "$result_root" \
    --expected-cases "$expected_cases" \
    --time-limit "$time_limit" \
    --tolerance "$tolerance" \
    --print-frequency 10000 \
    --verbose 2 \
    --required-julia 1.10.4 \
    --rerun "$rerun"
return_code=$?

nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader || true
printf 'CUPDCS_CBF_JOB_FINISH job=%s exit=%s utc=%s\n' \
    "$SLURM_JOB_ID" "$return_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$return_code"
