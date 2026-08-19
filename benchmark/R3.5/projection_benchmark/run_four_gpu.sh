#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
variant=${1:-final_newton2_v3}
tier=${2:-full}
artifact_dir=${3:-"$repo_root/benchmark/R3.5/artifacts/final_newton2_v3_sm90"}
result_dir=${4:-"$repo_root/benchmark/R3.5/projection_benchmark/results/${variant}_${tier}"}
samples=${SAMPLES:-11}
profile=${PROFILE:-true}

mkdir -p "$result_dir"
declare -a pids
for device in 0 1 2 3; do
    log="$result_dir/shard${device}.log"
    output="$result_dir/shard${device}.csv"
    env JULIA_DEPOT_PATH="$repo_root/.julia-depot" \
        JULIA_PKG_OFFLINE=true PDCS_SKIP_GPU_PRECOMPILE=1 \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir" \
        julia --project="$repo_root" \
        "$repo_root/benchmark/R3.5/projection_benchmark/run_projection_stress.jl" \
        --device="$device" --shard-index="$device" --shard-count=4 \
        --tier="$tier" --variant="$variant" --heterogeneous=true \
        --profile="$profile" --samples="$samples" --output="$output" \
        >"$log" 2>&1 &
    pids[$device]=$!
done

status=0
for device in 0 1 2 3; do
    if ! wait "${pids[$device]}"; then
        echo "GPU shard $device failed; inspect $result_dir/shard${device}.log" >&2
        status=1
    fi
done
exit "$status"
