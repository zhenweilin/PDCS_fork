#!/usr/bin/env bash

set -uo pipefail

benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
result_root="${FISHER_RESULT_ROOT:-$benchmark_dir/results/current}"
ledger="$result_root/submitted_jobs.tsv"
monitor_dir="$result_root/monitor"
interval="${FISHER_MONITOR_INTERVAL:-1800}"
status_log="$monitor_dir/fisher_status.log"
lock_file="$monitor_dir/fisher_status.lock"
pid_file="$monitor_dir/fisher_status.pid"
snapshot_file=""

[[ "$interval" =~ ^[1-9][0-9]*$ ]] || {
    printf 'FISHER_MONITOR_ERROR invalid_interval=%s\n' "$interval" >&2
    exit 2
}
[[ -s "$ledger" ]] || {
    printf 'FISHER_MONITOR_ERROR missing_ledger=%s\n' "$ledger" >&2
    exit 2
}

mkdir -p "$monitor_dir"
exec 9>"$lock_file"
flock -n 9 || {
    printf 'FISHER_MONITOR_ALREADY_RUNNING pid=%s\n' \
        "$(<"$pid_file" 2>/dev/null || printf unknown)"
    exit 0
}
printf '%s\n' "$$" > "$pid_file"
trap 'rm -f "$pid_file"; [[ -z "$snapshot_file" ]] || rm -f "$snapshot_file"' EXIT

latest_submission="$(
    awk -F '\t' '$3 == "preflight" { value = $1 } END { print value }' \
        "$ledger"
)"
mapfile -t job_ids < <(
    awk -F '\t' -v timestamp="$latest_submission" \
        '$1 == timestamp { print $2 }' "$ledger"
)
[[ "${#job_ids[@]}" == "19" ]] || {
    printf 'FISHER_MONITOR_ERROR submission=%s expected_jobs=19 found=%s\n' \
        "$latest_submission" "${#job_ids[@]}" >&2
    exit 2
}
job_csv="$(IFS=,; printf '%s' "${job_ids[*]}")"

while true; do
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    if queue_lines="$(
        squeue -h -j "$job_csv" \
            -o '%i|%j|%T|%M|%S|%R' 2>&1
    )"; then
        queue_ok=true
        active_count="$(
            printf '%s\n' "$queue_lines" |
                awk 'NF { count += 1 } END { print count + 0 }'
        )"
    else
        queue_ok=false
        active_count=unknown
    fi
    if accounting_lines="$(
        sacct -X -j "$job_csv" \
            --format=JobIDRaw,JobName%38,State,ExitCode,Elapsed,Start,End,NodeList \
            -Pn 2>&1
    )"; then
        accounting_ok=true
    else
        accounting_ok=false
    fi
    result_count="$(find "$result_root" -name result.toml -type f | wc -l)"
    snapshot_file="$(mktemp "$monitor_dir/.fisher_status.XXXXXX")"
    {
        printf 'FISHER_MONITOR_POLL time=%s submission=%s active=%s results=%s queue_ok=%s accounting_ok=%s\n' \
            "$timestamp" "$latest_submission" "$active_count" \
            "$result_count" "$queue_ok" "$accounting_ok"
        printf 'JOBID|NAME|STATE|ELAPSED|START|REASON_OR_NODE\n'
        printf '%s\n' "$queue_lines"
        printf 'ACCOUNTING\n'
        printf '%s\n' "$accounting_lines"
        printf 'FISHER_MONITOR_END time=%s\n' "$timestamp"
    } > "$snapshot_file"
    chmod 0664 "$snapshot_file"
    mv -f "$snapshot_file" "$status_log"
    snapshot_file=""

    if [[ "$queue_ok" != true ]]; then
        sleep "$interval"
        continue
    fi
    (( active_count > 0 )) || break
    sleep "$interval"
done
