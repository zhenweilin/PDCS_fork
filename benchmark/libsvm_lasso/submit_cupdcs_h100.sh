#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
large_lasso_dir="$repo_root/benchmark/large_scale_lasso"
julia_bin="$large_lasso_dir/.tools/julia-1.10.4/bin/julia"
julia_project="$large_lasso_dir/.gpu_solver_env"
artifact_dir="$large_lasso_dir/results/current/artifacts/cupdcs-sm90"
result_root="$script_dir/results/current/cupdcs"

[[ -x "$julia_bin" && -f "$julia_project/Manifest.toml" ]] || {
    printf 'The verified large-scale-lasso Julia/GPU environment is missing.\n' >&2
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

"$script_dir/download_default_datasets.sh" verify
mkdir -p "$result_root/slurm_logs"

existing_job_ids="$(squeue -h -u "$USER" -n libsvm-cupdcs-h100 -o '%A' | sort -u)"
if [[ -n "$existing_job_ids" ]]; then
    printf 'An active LIBSVM cuPDCS H100 job already exists: %s\n' \
        "$(printf '%s' "$existing_job_ids" | paste -sd, -)" >&2
    exit 3
fi

cd "$repo_root"
job_id="$(sbatch \
    --parsable \
    --chdir="$repo_root" \
    --output="$result_root/slurm_logs/job-%j.out" \
    --error="$result_root/slurm_logs/job-%j.out" \
    --export="ALL,PDCS_REPO_ROOT=$repo_root,LIBSVM_RAW_DIR=$script_dir/raw,LIBSVM_RESULT_ROOT=$result_root,LIBSVM_TIME_LIMIT=3600,LIBSVM_TOLERANCE=1e-6,PDCS_CUDA_PROJECTION_ARTIFACT_DIR=$artifact_dir" \
    "$script_dir/run_cupdcs_h100.sbatch")"

submission_tmp="$result_root/submitted_job_id.txt.tmp.$$"
printf '%s\n' "$job_id" >"$submission_tmp"
mv "$submission_tmp" "$result_root/submitted_job_id.txt"
printf 'LIBSVM_CUPDCS_SUBMITTED job_id=%s result_root=%s\n' \
    "$job_id" "$result_root"
