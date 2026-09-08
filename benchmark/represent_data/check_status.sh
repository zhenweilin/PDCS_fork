#!/usr/bin/env bash

set -euo pipefail

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
result_root="${CBF_RESULT_ROOT:-$benchmark_dir/results/current}"
ledger="$result_root/submitted_jobs.tsv"
[[ -s "$ledger" ]] || { printf 'No submission ledger: %s\n' "$ledger"; exit 0; }
job_ids="$(awk -F '\t' 'NR > 1 {print $2}' "$ledger" | paste -sd, -)"
printf 'dataset=represent_data expected=62 results=%s\n' \
    "$(find "$result_root/cupdcs/cases" -type f -name result.toml 2>/dev/null | wc -l)"
squeue -j "$job_ids" -o '%.18i %.32j %.9T %.10M %.10l %R' || true
sacct -X -j "$job_ids" --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End,NodeList%14 || true
if [[ -f "$result_root/campaign_summary.toml" ]]; then
    sed -n '1,120p' "$result_root/campaign_summary.toml"
fi

