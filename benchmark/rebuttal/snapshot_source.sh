#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT=""
while (($#)); do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --help|-h) printf '%s\n' 'Usage: snapshot_source.sh --output PATH'; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUTPUT" ]] || { printf '%s\n' '--output is required' >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { printf 'Refusing to overwrite %s\n' "$OUTPUT" >&2; exit 1; }
mkdir -p "$OUTPUT/untracked_sources"

git -C "$REPO_ROOT" rev-parse HEAD > "$OUTPUT/git_commit.txt"
git -C "$REPO_ROOT" status --short > "$OUTPUT/git_status.txt"
git -C "$REPO_ROOT" diff --binary > "$OUTPUT/git_diff_binary.patch"
git -C "$REPO_ROOT" diff --cached --binary > "$OUTPUT/git_diff_cached_binary.patch"

while IFS= read -r -d '' path; do
  case "$path" in
    benchmark/results/*|.julia-depot/*|.CondaPkg/*|.git/*) continue ;;
  esac
  destination="$OUTPUT/untracked_sources/$path"
  mkdir -p "$(dirname "$destination")"
  cp -p -- "$REPO_ROOT/$path" "$destination"
done < <(git -C "$REPO_ROOT" ls-files --others --exclude-standard -z)

(
  cd "$REPO_ROOT"
  find benchmark src test rebuttal_plan -type f \
    ! -path 'benchmark/results/*' -print0 |
    sort -z |
    xargs -0 sha256sum
) > "$OUTPUT/source_hashes.sha256"

(
  cd "$REPO_ROOT"
  find src/pdcs_gpu/cuda -maxdepth 1 -type f \
    \( -name '*.ptx' -o -name '*.so' \) -print0 |
    sort -z |
    xargs -0 -r sha256sum
) > "$OUTPUT/binary_hashes.sha256"
