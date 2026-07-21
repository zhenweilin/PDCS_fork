# Reproducing the GPU SOC projection benchmark

This guide reproduces the random second-order-cone (SOC) projection benchmark
with either CUDA 12.6 or CUDA 13.2. Run every command from the repository root.

## Version model

Three CUDA versions may appear on the same machine, and they are independent:

1. `nvidia-smi` reports the NVIDIA driver and the newest CUDA API supported by
   that driver. It does not select the compiler used by this project.
2. CUDA.jl selects a CUDA runtime for Julia GPU operations.
3. `CUDA_HOME` below selects the `nvcc` toolkit used to compile
   `src/pdcs_gpu/cuda/libfew_block_proj.so`.

The native library creates and destroys its cuBLAS handle through its own
linked cuBLAS library. Therefore, a `libfew_block_proj.so` built with CUDA 12.6
can safely run in a Julia process where CUDA.jl uses CUDA 13.x. A sufficiently
recent NVIDIA 13.x driver is backward compatible with CUDA-12.6 applications.

Only one build of `libfew_block_proj.so` is active at a time. Rebuilding it
with another toolkit replaces the current file. Always start a new Julia
process after rebuilding the library.

## Prerequisites

- NVIDIA GPU and working driver (`nvidia-smi` must succeed)
- CUDA 12.6 at `/usr/local/cuda-12.6`, CUDA 13.2 at
  `/usr/local/cuda-13.2`, or adjusted paths in the commands below
- GNU Make and a host C++ compiler supported by the chosen CUDA toolkit
- Repository-local Julia at `./.julia-bin/julia`

The examples target an NVIDIA H100 (`sm_90`). For another GPU, set `ARCH` to
its compute architecture.

Check the environment:

```bash
nvidia-smi
readlink -f /usr/local/cuda
/usr/local/cuda-12.6/bin/nvcc --version
/usr/local/cuda-13.2/bin/nvcc --version
./.julia-bin/julia --version
```

## Install the Julia environment

The following command keeps all Julia packages inside the repository:

```bash
JULIA_DEPOT_PATH="$PWD/.julia-depot" \
  ./.julia-bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Verify CUDA.jl separately from the native library:

```bash
JULIA_DEPOT_PATH="$PWD/.julia-depot" \
  ./.julia-bin/julia --project=. -e '
    using CUDA
    println("CUDA functional: ", CUDA.functional())
    CUDA.versioninfo()
  '
```

`CUDA functional: true` is required before running the benchmark.

## Workflow A: build and test with CUDA 12.6

### A1. Confirm the selected compiler

```bash
make -C src/pdcs_gpu/cuda print-config \
  CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90
```

The last line should identify CUDA 12.6.

### A2. Rebuild the native library

```bash
make -C src/pdcs_gpu/cuda rebuild-few \
  CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90
```

### A3. Verify architecture and linkage

```bash
/usr/local/cuda-12.6/bin/cuobjdump --list-elf \
  src/pdcs_gpu/cuda/libfew_block_proj.so
ldd src/pdcs_gpu/cuda/libfew_block_proj.so | grep -E 'cublas|not found'
```

Expected for this H100 build:

```text
libfew_block_proj.*.sm_90.cubin
libcublas.so.12 => /usr/local/cuda-12.6/...
```

There must be no `not found` entries.

### A4. Run and save the CUDA-12.6 results

```bash
mkdir -p benchmark/results
JULIA_DEPOT_PATH="$PWD/.julia-depot" \
PDCS_SKIP_GPU_PRECOMPILE=1 \
  ./.julia-bin/julia --project=. benchmark/random_soc_projection.jl \
  > benchmark/results/h100_soc_projection_cuda12_6.csv
```

## Workflow B: build and test with CUDA 13.2

### B1. Confirm the selected compiler

```bash
make -C src/pdcs_gpu/cuda print-config \
  CUDA_HOME=/usr/local/cuda-13.2 ARCH=sm_90
```

The last line should identify CUDA 13.2.

### B2. Rebuild the native library

```bash
make -C src/pdcs_gpu/cuda rebuild-few \
  CUDA_HOME=/usr/local/cuda-13.2 ARCH=sm_90
```

### B3. Verify architecture and linkage

```bash
/usr/local/cuda-13.2/bin/cuobjdump --list-elf \
  src/pdcs_gpu/cuda/libfew_block_proj.so
ldd src/pdcs_gpu/cuda/libfew_block_proj.so | grep -E 'cublas|not found'
```

`cuobjdump` should report `sm_90`, `ldd` should resolve cuBLAS from the intended
CUDA installation, and there must be no `not found` entries.

### B4. Run and save the CUDA-13.2 results

Start a new Julia process by running this command after the rebuild:

```bash
mkdir -p benchmark/results
JULIA_DEPOT_PATH="$PWD/.julia-depot" \
PDCS_SKIP_GPU_PRECOMPILE=1 \
  ./.julia-bin/julia --project=. benchmark/random_soc_projection.jl \
  > benchmark/results/h100_soc_projection_cuda13_2.csv
```

The two distinct output names prevent one toolkit's results from overwriting
the other toolkit's results.

## Benchmark definition

The default seeded sweep uses:

- random seed: `2026`
- cone counts: `1, 2, 3, 4, 32, 1,024, 65,536`
- cone dimensions: `4, 16, 64, 256, 2,048`
- timed samples per case: `10`
- maximum elements per case: `8,000,000`

Cases above the element limit are skipped. Counts one and two run only the
`few` strategy because the other native kernels assume at least three block
metadata entries. For larger counts, the harness compares every applicable
strategy. The solver-selected strategy has a trailing `*` in the CSV.

Every strategy is checked against the closed-form Euclidean SOC projection
before timing. CSV columns are:

```text
gpu,cone_count,cone_dimension,total_elements,strategy,median_ms,max_error,status
```

Only `PASS` rows are valid timing results. A `FAIL` row records `NaN` for its
timing because an incorrect kernel must not be compared by performance.

Override the default sweep with environment variables. For example, test the
actual `few` selection range:

```bash
JULIA_DEPOT_PATH="$PWD/.julia-depot" \
PDCS_SKIP_GPU_PRECOMPILE=1 \
PDCS_SOC_COUNTS=1,2,3 \
PDCS_SOC_DIMS=4,16,64,256,2048 \
PDCS_SOC_SAMPLES=10 \
PDCS_SOC_SEED=2026 \
PDCS_SOC_MAX_ELEMENTS=8000000 \
  ./.julia-bin/julia --project=. benchmark/random_soc_projection.jl
```

## Recorded H100 result

The checked-in workspace currently contains:

```text
benchmark/results/h100_soc_projection.csv
```

It was generated on an NVIDIA H100 with the native `few` library built by the
CUDA 12.6 toolkit while `/usr/local/cuda` pointed to CUDA 13.2. It contains 94
data rows, and all 94 rows have `status=PASS`.

Validate any generated CSV with:

```bash
wc -l benchmark/results/h100_soc_projection.csv
awk -F, 'NR > 1 {count[$8]++} END {for (key in count) print key, count[key]}' \
  benchmark/results/h100_soc_projection.csv
```

The current file has 95 lines including the header and reports `PASS 94`.

## Troubleshooting

### CUDA.jl reports `CUDA runtime not found`

This can happen if CUDA.jl was first precompiled in a sandbox, on a login node,
or without an NVIDIA driver. With GPU access enabled, rebuild the cached CUDA
runtime selection:

```bash
JULIA_DEPOT_PATH="$PWD/.julia-depot" \
  ./.julia-bin/julia --project=. -e '
    pkg = Base.PkgId(
        Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"),
        "CUDA_Runtime_jll",
    )
    Base.compilecache(pkg)
  '
```

Start a new Julia process and rerun `CUDA.versioninfo()`.

### `ldd` reports a missing CUDA library

Expose the library directory for the selected toolkit in the shell that starts
Julia. For CUDA 12.6:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
```

For CUDA 13.2:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
```

Rerun `ldd` before starting Julia.

### The wrong CUDA toolkit is used

Do not rely on the `/usr/local/cuda` symlink. Pass `CUDA_HOME` explicitly to
both `print-config` and `rebuild-few`, then confirm `nvcc --version` and `ldd`.

### Results change after rebuilding

Do not reuse an already-running Julia session after replacing
`libfew_block_proj.so`; the old shared object remains loaded in that process.
Exit Julia completely and start a new process for each build.
