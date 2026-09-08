#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
raw_dir="${LIBSVM_RAW_DIR:-$script_dir/raw}"
mode="${1:-download}"

case "$mode" in
    download|verify) ;;
    *)
        printf 'Usage: %s [download|verify]\n' "$0" >&2
        exit 2
        ;;
esac

dataset_ids=(news20 E2006-log1p rcv1-train)
dataset_files=(
    news20.binary.bz2
    log1p.E2006.train.bz2
    rcv1_train.binary.bz2
)
dataset_urls=(
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.train.bz2
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
)
dataset_bytes=(26779006 245706096 13730096)

mkdir -p "$raw_dir"

for index in "${!dataset_ids[@]}"; do
    dataset_id="${dataset_ids[$index]}"
    file_name="${dataset_files[$index]}"
    url="${dataset_urls[$index]}"
    expected_bytes="${dataset_bytes[$index]}"
    final_path="$raw_dir/$file_name"
    partial_path="$final_path.part"

    if [[ -f "$final_path" ]]; then
        actual_bytes="$(stat -c '%s' "$final_path")"
        if [[ "$actual_bytes" == "$expected_bytes" ]]; then
            printf 'LIBSVM_DATASET_OK id=%s bytes=%s path=%s\n' \
                "$dataset_id" "$actual_bytes" "$final_path"
            continue
        fi
        printf 'Invalid completed file for %s: expected %s bytes, found %s at %s\n' \
            "$dataset_id" "$expected_bytes" "$actual_bytes" "$final_path" >&2
        exit 1
    fi

    if [[ "$mode" == "verify" ]]; then
        printf 'Missing validated dataset %s at %s\n' "$dataset_id" "$final_path" >&2
        exit 1
    fi

    if [[ -f "$partial_path" ]]; then
        partial_bytes="$(stat -c '%s' "$partial_path")"
        if ((partial_bytes > expected_bytes)); then
            invalid_path="$partial_path.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
            mv "$partial_path" "$invalid_path"
            printf 'Moved oversized partial download to %s\n' "$invalid_path" >&2
        fi
    fi

    printf 'LIBSVM_DOWNLOAD_START id=%s url=%s path=%s\n' \
        "$dataset_id" "$url" "$partial_path"
    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 5 \
        --continue-at - \
        --output "$partial_path" \
        "$url"

    actual_bytes="$(stat -c '%s' "$partial_path")"
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
        printf 'Incomplete download for %s: expected %s bytes, found %s at %s\n' \
            "$dataset_id" "$expected_bytes" "$actual_bytes" "$partial_path" >&2
        exit 1
    fi
    mv "$partial_path" "$final_path"
    printf 'LIBSVM_DOWNLOAD_FINISH id=%s bytes=%s path=%s\n' \
        "$dataset_id" "$actual_bytes" "$final_path"
done

printf 'LIBSVM_DEFAULT_DATASETS_READY count=%s raw_dir=%s\n' \
    "${#dataset_ids[@]}" "$raw_dir"
