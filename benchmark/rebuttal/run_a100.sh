#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PDCS_GPU_ARCH=sm_80
exec "$SCRIPT_DIR/reproduce_all.sh" --arch sm_80 "$@"
