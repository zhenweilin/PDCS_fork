#!/usr/bin/env bash

set -euo pipefail

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$benchmark_dir/../.." && pwd)"
julia="$repo_root/benchmark/large_scale_lasso/.tools/julia-1.10.4/bin/julia"

[[ -x "$julia" ]] || {
    printf 'Missing the shared Lasso/Fisher Julia 1.10.4 at %s\n' \
        "$julia" >&2
    exit 2
}

module purge
module load cuda/12.6.2-gcc-12.4.0

export JULIA_DEPOT_PATH="$benchmark_dir/.julia-depot"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=/usr/bin/python3.12
export PDCS_SKIP_GPU_PRECOMPILE=1

"$julia" --startup-file=no "$benchmark_dir/prepare_gpu_environments.jl"
