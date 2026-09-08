#!/usr/bin/env bash

set -euo pipefail

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$benchmark_dir/../.." && pwd)"
input_root="$repo_root/represent_data"
result_root="${CBF_RESULT_ROOT:-$benchmark_dir/results/current}"
case_list="$result_root/cases.txt"
ledger="$result_root/submitted_jobs.tsv"
expected_cases=62
julia_bin="${CBF_JULIA_BIN:-$repo_root/benchmark/large_scale_lasso/.tools/julia-1.10.4/bin/julia}"
dependency="${CBF_DEPENDENCY:-}"

if [[ -n "$dependency" && ! "$dependency" =~ ^after(any|ok):[0-9]+$ ]]; then
    printf 'Invalid CBF_DEPENDENCY: %s\n' "$dependency" >&2
    exit 2
fi
dependency_option=()
[[ -z "$dependency" ]] || dependency_option=(--dependency="$dependency")

[[ -x "$julia_bin" ]] || {
    printf 'Missing Julia 1.10.4 executable: %s\n' "$julia_bin" >&2
    exit 2
}
"$julia_bin" --startup-file=no -e '
function contains_parse_error(value)
    value isa Expr || return false
    value.head in (:error, :incomplete) && return true
    return any(contains_parse_error, value.args)
end
for path in ARGS
    parsed = Meta.parseall(read(path, String); filename = path)
    contains_parse_error(parsed) && error("Julia parse error in $path")
end
' "$benchmark_dir/run_cupdcs_batch.jl"

mkdir -p "$result_root/slurm_logs"
find "$input_root" -maxdepth 1 -type f \
    \( -name '*.cbf' -o -name '*.cbf.gz' -o -name '*.cbf.bz2' \) \
    ! -name 'isil01.cbf.gz' -printf '%f\n' | LC_ALL=C sort > "$case_list"
actual_cases="$(wc -l < "$case_list")"
[[ "$actual_cases" == "$expected_cases" ]] || {
    printf 'Expected %s represent_data cases after excluding isil01; found %s\n' \
        "$expected_cases" "$actual_cases" >&2
    exit 2
}
if [[ -s "$ledger" && "${CBF_ALLOW_RESUBMIT:-0}" != "1" ]]; then
    printf 'Submission ledger already exists: %s\n' "$ledger" >&2
    printf 'Set CBF_ALLOW_RESUBMIT=1 only to resume intentionally.\n' >&2
    exit 2
fi

commit="$(git -C "$repo_root" rev-parse HEAD)"
job_id="$(
    cd "$repo_root"
    sbatch --parsable \
        "${dependency_option[@]}" \
        --output="$result_root/slurm_logs/cupdcs-%j.out" \
        --error="$result_root/slurm_logs/cupdcs-%j.out" \
        --export="ALL,PDCS_REPO_ROOT=$repo_root,CBF_INPUT_ROOT=$input_root,CBF_RESULT_ROOT=$result_root,CBF_CASE_LIST=$case_list,CBF_EXPECTED_CASES=$expected_cases,CBF_TIME_LIMIT=3600,CBF_TOLERANCE=1e-6" \
        "$benchmark_dir/cupdcs_h100.sbatch"
)"

if [[ ! -s "$ledger" ]]; then
    printf 'submitted_utc\tjob_id\tdataset\tsolver\tcases\ttolerance\tper_case_time_limit\tcommit\n' > "$ledger"
fi
printf '%s\t%s\trepresent_data\tcupdcs\t%s\t1e-6\t3600\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_id" "$expected_cases" "$commit" >> "$ledger"
printf 'SUBMITTED dataset=represent_data solver=cupdcs job=%s cases=%s ledger=%s\n' \
    "$job_id" "$expected_cases" "$ledger"
