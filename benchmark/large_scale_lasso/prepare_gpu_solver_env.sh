#!/usr/bin/env bash

set -euo pipefail

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tools_dir="$benchmark_dir/.tools"
julia_version="1.10.4"
julia_dir="$tools_dir/julia-$julia_version"
archive="julia-$julia_version-linux-x86_64.tar.gz"
archive_path="$tools_dir/$archive"
download_url="https://julialang-s3.julialang.org/bin/linux/x64/1.10/$archive"
checksums_url="https://julialang-s3.julialang.org/bin/checksums/julia-$julia_version.sha256"

mkdir -p "$tools_dir"
if [[ ! -x "$julia_dir/bin/julia" ]]; then
    if [[ ! -f "$archive_path" ]]; then
        curl --fail --location --retry 3 "$download_url" --output "$archive_path"
    fi
    expected="$tools_dir/julia-$julia_version.sha256"
    curl --fail --location --retry 3 "$checksums_url" --output "$expected"
    (
        cd "$tools_dir"
        grep "  $archive\$" "$expected" | sha256sum --check -
        tar -xzf "$archive"
    )
fi

module purge
module load cuda/12.6.2-gcc-12.4.0
"$julia_dir/bin/julia" --startup-file=no \
    "$benchmark_dir/prepare_gpu_solver_env.jl"
