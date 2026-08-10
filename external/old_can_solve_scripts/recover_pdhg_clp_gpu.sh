#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="${ROOT_DIR}/external/pdhg_clp"
REMOTE_URL="https://gitee.com/zhenweilin/pdhg_clp.git"
CREDENTIAL_FILE="${ROOT_DIR}/external/gitee.txt"
COMMIT="3432afc69f96891168c13c926d08cdaacd5b0e41"
WORKTREE_DIR="${SCRIPT_DIR}/worktrees/pdhg_clp_gpu_3432afc"
NVCC="/usr/local/cuda-12.6/bin/nvcc"
ARCH="sm_90"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO_DIR="$2"; shift 2 ;;
        --remote-url) REMOTE_URL="$2"; shift 2 ;;
        --credential-file) CREDENTIAL_FILE="$2"; shift 2 ;;
        --commit) COMMIT="$2"; shift 2 ;;
        --worktree) WORKTREE_DIR="$2"; shift 2 ;;
        --nvcc) NVCC="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --help|-h)
            printf '%s\n' \
                "Usage: $0 [options]" \
                "  --repo PATH" \
                "  --remote-url HTTPS_URL" \
                "  --credential-file PATH" \
                "  --commit HASH" \
                "  --worktree PATH" \
                "  --nvcc PATH" \
                "  --arch sm_XX"
            exit 0
            ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

REPO_DIR="$(readlink -m "$REPO_DIR")"
CREDENTIAL_FILE="$(readlink -f "$CREDENTIAL_FILE")"
WORKTREE_PARENT="$(dirname "$WORKTREE_DIR")"
mkdir -p "$WORKTREE_PARENT"
WORKTREE_DIR="$(readlink -m "$WORKTREE_DIR")"

[[ -f "$CREDENTIAL_FILE" ]] || {
    printf 'Missing credential file: %s\n' "$CREDENTIAL_FILE" >&2
    exit 2
}
[[ -x "$NVCC" ]] || {
    printf 'Missing nvcc: %s\n' "$NVCC" >&2
    exit 2
}

# Avoid exposing the credential in the command line, Git configuration, remote
# URL, or logs. The helper emits only the field requested by Git.
chmod 600 "$CREDENTIAL_FILE"
if [[ -d "$REPO_DIR/.git" ]]; then
    GIT_TERMINAL_PROMPT=0 \
    GIT_ASKPASS="${SCRIPT_DIR}/gitee_askpass.sh" \
    PDCS_GITEE_CREDENTIAL_FILE="$CREDENTIAL_FILE" \
        git -C "$REPO_DIR" fetch --all --tags --prune
else
    if [[ -d "$REPO_DIR" ]] &&
       [[ -n "$(find "$REPO_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'Repository target exists and is not empty: %s\n' "$REPO_DIR" >&2
        exit 2
    fi
    mkdir -p "$(dirname "$REPO_DIR")"
    GIT_TERMINAL_PROMPT=0 \
    GIT_ASKPASS="${SCRIPT_DIR}/gitee_askpass.sh" \
    PDCS_GITEE_CREDENTIAL_FILE="$CREDENTIAL_FILE" \
        git clone "$REMOTE_URL" "$REPO_DIR"
fi

git -C "$REPO_DIR" cat-file -e "${COMMIT}^{commit}"
if [[ ! -e "$WORKTREE_DIR/.git" ]]; then
    git -C "$REPO_DIR" worktree add --detach "$WORKTREE_DIR" "$COMMIT"
else
    actual_commit="$(git -C "$WORKTREE_DIR" rev-parse HEAD)"
    expected_commit="$(git -C "$REPO_DIR" rev-parse "$COMMIT")"
    [[ "$actual_commit" == "$expected_commit" ]] || {
        printf 'Existing worktree is at %s, expected %s\n' \
            "$actual_commit" "$expected_commit" >&2
        exit 2
    }
fi

patch_file="${SCRIPT_DIR}/patches/pdhg_clp_3432afc_compile_total_threads.patch"
if [[ "$(git -C "$WORKTREE_DIR" rev-parse HEAD)" == \
      "3432afc69f96891168c13c926d08cdaacd5b0e41" ]]; then
    if git -C "$WORKTREE_DIR" apply --check "$patch_file" 2>/dev/null; then
        git -C "$WORKTREE_DIR" apply "$patch_file"
    elif ! git -C "$WORKTREE_DIR" apply --reverse --check "$patch_file" \
        2>/dev/null; then
        printf 'Compile patch is neither applicable nor already applied.\n' >&2
        exit 2
    fi
fi

cuda_dir="${WORKTREE_DIR}/code/src/rpdhg_clp_gpu/cuda"
make -C "$cuda_dir" \
    NVCC="$NVCC" \
    ARCH="$ARCH" \
    CFLAGS="-Xcompiler -fPIC -arch=${ARCH}"

git -C "$WORKTREE_DIR" rev-parse HEAD > "$WORKTREE_DIR/RECOVERED_COMMIT.txt"
sha256sum \
    "$cuda_dir/libfew_block_proj.so" \
    "$cuda_dir/massive_block_proj.ptx" \
    "$cuda_dir/moderate_block_proj.ptx" \
    "$cuda_dir/sufficient_block_proj.ptx" \
    "$cuda_dir/utils.ptx" \
    > "$WORKTREE_DIR/RECOVERED_KERNEL_SHA256.txt"

printf 'PDHG_CLP_GPU_RECOVERED worktree=%s commit=%s arch=%s\n' \
    "$WORKTREE_DIR" "$(git -C "$WORKTREE_DIR" rev-parse HEAD)" "$ARCH"
