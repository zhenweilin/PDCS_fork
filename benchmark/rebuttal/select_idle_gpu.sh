#!/usr/bin/env bash
# Print one lightly loaded GPU index.  This is a scheduling hint, not a lock:
# the caller should still respect any local cluster scheduler allocation.
set -euo pipefail

MAX_UTILIZATION="${1:-10}"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 1; }

query="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1)" || {
  printf '%s\n' "$query" >&2
  exit 1
}

best=""
best_free=-1
while IFS=, read -r index free util; do
  index="${index//[[:space:]]/}"
  free="${free//[[:space:]]/}"
  util="${util//[[:space:]]/}"
  [[ "$index" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || continue
  (( util <= MAX_UTILIZATION && free > best_free )) || continue
  best="$index"; best_free="$free"
done <<< "$query"

[[ -n "$best" ]] || { echo "no GPU has utilization <= ${MAX_UTILIZATION}%" >&2; exit 1; }
printf '%s\n' "$best"
