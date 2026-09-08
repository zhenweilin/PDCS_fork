#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
large_lasso_dir="$repo_root/benchmark/large_scale_lasso"
julia_bin="$large_lasso_dir/.tools/julia-1.10.4/bin/julia"
julia_project="$large_lasso_dir/.gpu_solver_env"
scs_project="$large_lasso_dir/.gpu_scs_env"
artifact_dir="$large_lasso_dir/results/current/artifacts/cupdcs-sm90"
config="$script_dir/illcondition_lasso.toml"
instance_dir="$script_dir/instances"
manifest="$instance_dir/generated_instances.toml"
result_root="$script_dir/results/current"

[[ -x "$julia_bin" && -f "$julia_project/Manifest.toml" && \
   -f "$scs_project/Manifest.toml" ]] || {
    printf 'The validated large-scale-lasso Julia/GPU environments are missing.\n' >&2
    exit 2
}

solvers=(cuclarabel cuscs cupdcs)
for solver in "${solvers[@]}"; do
    job_name="illlasso-$solver"
    existing_job_ids="$(squeue -h -u "$USER" -n "$job_name" -o '%A' | sort -u)"
    if [[ -n "$existing_job_ids" ]]; then
        printf 'An active %s job already exists: %s\n' "$solver" \
            "$(printf '%s' "$existing_job_ids" | paste -sd, -)" >&2
        exit 3
    fi
done

if [[ -f "$manifest" ]] && env JULIA_PKG_OFFLINE=true \
    "$julia_bin" --startup-file=no --project="$julia_project" \
    "$script_dir/verify_instances.jl" "$manifest"
then
    printf 'ILL_LASSO_INSTANCE_CACHE_REUSED manifest=%s\n' "$manifest"
else
    env JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
        JULIA_CONDAPKG_BACKEND=Null PDCS_SKIP_GPU_PRECOMPILE=1 \
        "$julia_bin" --startup-file=no --threads=16 --project="$julia_project" \
        "$script_dir/generate_instances.jl" --config "$config" \
        --output-dir "$instance_dir"
    env JULIA_PKG_OFFLINE=true "$julia_bin" --startup-file=no \
        --project="$julia_project" "$script_dir/verify_instances.jl" "$manifest"
fi

mkdir -p "$result_root/slurm_logs"
job_ids=()

cd "$repo_root"
for solver in "${solvers[@]}"; do
    job_name="illlasso-$solver"
    job_id="$(sbatch --parsable --job-name="$job_name" \
        --chdir="$repo_root" \
        --output="$result_root/slurm_logs/${solver}-%j.out" \
        --error="$result_root/slurm_logs/${solver}-%j.out" \
        --export="ALL,PDCS_REPO_ROOT=$repo_root,ILL_LASSO_SOLVER=$solver,ILL_LASSO_MANIFEST=$manifest,ILL_LASSO_RESULT_ROOT=$result_root,ILL_LASSO_TOLERANCE=1e-6,ILL_LASSO_TIME_LIMIT=3600,PDCS_CUDA_PROJECTION_ARTIFACT_DIR=$artifact_dir" \
        "$script_dir/run_solver_h100.sbatch")"
    job_ids+=("$job_id")
    printf 'ILL_LASSO_SUBMITTED solver=%s job_id=%s\n' "$solver" "$job_id"
done

submission_tmp="$result_root/submitted_jobs.toml.tmp.$$"
{
    printf 'schema_version = 1\n'
    printf 'manifest = "%s"\n' "$manifest"
    printf 'julia_version = "1.10.4"\n'
    printf 'tolerance = 1.0e-6\n'
    printf 'time_limit_seconds_per_case = 3600\n'
    for index in "${!solvers[@]}"; do
        printf '\n[[jobs]]\nsolver = "%s"\njob_id = "%s"\n' \
            "${solvers[$index]}" "${job_ids[$index]}"
    done
} >"$submission_tmp"
mv "$submission_tmp" "$result_root/submitted_jobs.toml"
printf 'ILL_LASSO_ALL_SUBMITTED jobs=%s result_root=%s\n' \
    "$(IFS=,; printf '%s' "${job_ids[*]}")" "$result_root"
