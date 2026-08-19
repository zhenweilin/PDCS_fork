#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
artifact_dir=${1:-"$repo_root/benchmark/R3.5/artifacts/final_newton2_v3_sm90"}
result_dir=${2:-"$repo_root/benchmark/R3.5/projection_benchmark/results/represent_hierarchy_ab"}
samples=${SAMPLES:-5}

# Keep all four mappings of a solver-shaped input on one card.  The groups are
# balanced by the measured cost of the slowest mapping, not by case count.
regexes=(
  '^represent_joint_FC_12'
  '^represent_(qssp180|db_plane_strain_prism)'
  '^represent_(cx02_100|integrated)'
  '^represent_(ravem|gams01|batch|batchs101006m|varun)'
)

mkdir -p "$result_dir"
declare -a pids
for device in 0 1 2 3; do
    env JULIA_DEPOT_PATH="$repo_root/.julia-depot" \
        JULIA_PKG_OFFLINE=true PDCS_SKIP_GPU_PRECOMPILE=1 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir" \
        julia --project="$repo_root" \
        "$repo_root/benchmark/R3.5/projection_benchmark/run_projection_stress.jl" \
        --device="$device" --shard-index=0 --shard-count=1 \
        --tier=full --variant=represent_hierarchy_ab \
        --heterogeneous=true --profile=true --samples="$samples" \
        --case-regex="${regexes[$device]}" \
        --output="$result_dir/gpu${device}.csv" \
        >"$result_dir/gpu${device}.log" 2>&1 &
    pids[$device]=$!
done

status=0
for device in 0 1 2 3; do
    if ! wait "${pids[$device]}"; then
        echo "GPU $device failed; inspect $result_dir/gpu${device}.log" >&2
        status=1
    fi
done
exit "$status"
