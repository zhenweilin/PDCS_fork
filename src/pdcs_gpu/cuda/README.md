# Portable CUDA projection artifacts

This directory contains the CUDA sources used by PDCS.  The grid-wise
projection is slightly different from the other projection kernels: its
implementation is a shared library (`libfew_block_proj.so`) that calls
cuBLAS, while Julia/CUDA.jl may load a different CUDA runtime.  A cuBLAS
handle is an opaque object and must never be created by one cuBLAS library
instance and consumed by another one.

The current implementation is designed to run on both of the environments
used for the PDCS experiments:

| Environment | GPU | Julia/CUDA.jl runtime | Native build |
|---|---|---|---|
| Altman | H100 80 GB | CUDA 12.4 / CUDA.jl 5.8.x | CUDA 12.4, `sm_90` |
| Northwestern | H100 80 GB | CUDA artifact may be 13.x | CUDA 12.6, `sm_90` |

The versions do not have to be identical.  The native library now creates,
configures, queries, uses, and destroys its own cuBLAS handle.  Julia passes
only the pointer returned by that same library.  This removes the invalid
cross-runtime handle assumption.

## What is protected

The production grid-wise path has four independent safeguards:

1. `PDCS_GPU.__init__` resolves the artifact directory before loading the
   library and checks that the current ABI symbols are exported.
2. The native library owns the cuBLAS handle and explicitly uses the handle's
   stream for the reduction, scalar branch, and scaling operation.
3. If the caller passes `temp === vec`, Julia substitutes a persistent,
   non-aliasing workspace.  This prevents `cublasDnrm2` from overwriting the
   SOC head before the projection branch reads it.
4. The first grid-wise call performs a small SOC projection self-test (both
   independent and aliasing workspaces).  If it fails, PDCS automatically uses
   the semantically equivalent block-wise implementation instead of silently
   returning an invalid projection.

The self-test runs after package loading on the actual GPU.  CUDA contexts and
native handles are deliberately not serialized in Julia's precompile cache.

## Build production artifacts

Build into a separate directory; do not commit generated `.ptx` or `.so`
files.  The same command works on both machines when `CUDA_HOME` points to the
desired toolkit:

```bash
cd /home/zhenwei/PDCS_fork
artifact_dir=$(mktemp -d /tmp/pdcs-cuda-artifacts.XXXXXX)
make -C src/pdcs_gpu/cuda rebuild-gpu \
    CUDA_HOME=/usr/local/cuda-12.4 ARCH=sm_90 OUTPUT_DIR="$artifact_dir"
```

On Northwestern use its loaded CUDA module, for example:

```bash
module load cuda/12.6.2-gcc-12.4.0
artifact_dir=$(mktemp -d /tmp/pdcs-cuda-artifacts.XXXXXX)
make -C src/pdcs_gpu/cuda rebuild-gpu \
    CUDA_HOME="$CUDA_HOME" ARCH=sm_90 OUTPUT_DIR="$artifact_dir"
```

The output directory must contain:

```text
libfew_block_proj.so
moderate_block_proj.ptx
sufficient_block_proj.ptx
massive_block_proj.ptx
utils.ptx
```

Check the native ABI before running a solver:

```bash
nm -D "$artifact_dir/libfew_block_proj.so" | grep -E \
  'few_block_proj|create_cublas|configure_cublas|configuration_inner|destroy_cublas'
ldd "$artifact_dir/libfew_block_proj.so" | grep -E 'cublas|cudart|cuda'
```

The four handle helper symbols are required.  An old `.so` that only exports
`few_block_proj` is rejected in native mode and automatically falls back in
`auto` mode.

## Runtime configuration

Set the artifact directory before starting Julia:

```bash
export PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PDCS_SKIP_GPU_PRECOMPILE=1       # useful on shared clusters
```

`PDCS_GRIDWISE_MODE` controls the policy:

| Value | Behavior |
|---|---|
| `auto` (default) | Run the ABI/self-test; fall back to block-wise on failure. |
| `native` | Require the native path; any missing symbol or failed self-test is an error. |
| `block` | Do not load/use the grid-wise library; always use block-wise projection. |

Whenever a grid-wise call actually takes the block-wise fallback, PDCS emits a
warning containing the selected mode, runtime state, and failure reason.  The
warning is emitted once per Julia process so a long solve is not flooded with
identical messages.

The following optional variables are useful for diagnosis:

```bash
export PDCS_GRIDWISE_SELFTEST=1       # default; set to 0 only for controlled timing
export PDCS_GRIDWISE_STRICT=1         # fail instead of falling back in auto mode
export PDCS_CUBLAS_REPRODUCIBLE=1     # default; uses prescribed cuBLAS math mode
```

Use one process per GPU.  For example, the command below binds the process to
physical GPU 2 even when the host has four GPUs:

```bash
CUDA_VISIBLE_DEVICES=2 \
PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir" \
PDCS_SKIP_GPU_PRECOMPILE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
julia --startup-file=no --project=. test/test_gridwise_lazy_handle_gpu.jl
```

`CUDA_VISIBLE_DEVICES=2` makes that physical card appear as CUDA device 0 to
the process.  Do not launch a second PDCS process on the same card.

## Package precompile behavior

PDCS has two separate checks because Julia precompilation and runtime loading
have different constraints:

* During precompile, PDCS checks artifact files and the exported native ABI
  symbols, but does not create a CUDA context or cuBLAS handle.  If the
  selected directory is incomplete or stale, GPU precompile is skipped with
  an actionable warning; CPU functionality remains loadable.
* GPU contexts, cuBLAS handles, and CUDA modules are never saved in the cache.
  After loading on the target machine, the first grid-wise projection runs
  the native handle/configuration/SOC alias self-test described above.

To force a clean package precompile without using a GPU:

```bash
PDCS_SKIP_GPU_PRECOMPILE=1 julia --startup-file=no --project=. -e \
  'using PDCS; println("PDCS CPU package loaded")'
```

To inspect the selected artifact and runtime state from Julia:

```julia
using CUDA
using PDCS
using PDCS: PDCS_GPU

println(PDCS_GPU.gridWise_runtime_status())
PDCS_GPU.check_gridWise_runtime!()
println(PDCS_GPU.gridWise_runtime_status())
```

The status reports the artifact directory, missing files, native/fallback
state, failure reason, and effective cuBLAS reproducibility configuration.

## Regression test

The lightweight regression checks lazy handle creation, native symbol use,
finite output, handle reuse, and runtime status:

```bash
CUDA_VISIBLE_DEVICES=2 \
PDCS_SKIP_GPU_PRECOMPILE=1 \
PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$artifact_dir" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
julia --startup-file=no --project=. test/test_gridwise_lazy_handle_gpu.jl
```

For a large alias regression, use an ordinary SOC of dimension 10,002 and
compare `temp` independent of `vec` with `temp === vec`.  The two outputs must
agree to the requested tolerance over repeated calls.  The current H100
validation produced:

```text
gridWise lazy cuBLAS handle: 7/7 passed
dimension=10002: independent/alias max error = 0
native configuration: reproducible=true, atomics_mode=0, math_mode=18
```

The same test can be run with `PDCS_GRIDWISE_MODE=block` to verify the safe
fallback path.  Fallback is intended for compatibility/correctness, not for
publication timing; rebuild a matching native artifact to restore peak
grid-wise performance.

## Comparing another machine

Record all of the following, not only `nvidia-smi`:

```bash
nvidia-smi
nvcc --version
julia --version
julia --project=. --startup-file=no -e \
  'using CUDA, Libdl; println(pkgversion(CUDA)); println(CUDA.runtime_version()); println(Libdl.dlpath(CUDA.CUBLAS.libcublas))'
ldd "$artifact_dir/libfew_block_proj.so" | grep -E 'cublas|cudart|cuda'
git rev-parse HEAD
sha256sum "$artifact_dir/libfew_block_proj.so" "$artifact_dir"/*.ptx
```

Also record loaded modules, `LD_LIBRARY_PATH`,
`PDCS_CUDA_PROJECTION_ARTIFACT_DIR`, `CUBLAS_WORKSPACE_CONFIG`, and the exact
Julia project/manifest.  A CUDA toolkit used by `nvcc` is not necessarily the
CUDA runtime selected by CUDA.jl; that distinction is precisely why native
handle ownership is required.
