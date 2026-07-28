# Grid-wise cuBLAS and GPU/CPU Lasso Debug Record

## 1. Status and final conclusion

**Status: fixed and verified on 2026-07-28.**

The GPU/CPU discrepancy on the shared tiny ill-conditioned Lasso instance was
not caused by an early restart, sparse matrix multiplication, preprocessing,
or an intentional change of projection strategy. It was caused by two
independent grid-wise SOC implementation defects:

1. The diagonal SOC implementation in `few_block_proj.cu` reused persistent
   scaling entries as root-search bounds and had the wrong bisection update for
   the negative-head (`t < 0`) branch.
2. More decisively for the Lasso solve, `few_con_proj!` passed `y.y` as both
   the SOC input/output vector and the cuBLAS temporary buffer. For an unscaled
   SOC, `cublasDnrm2` writes the tail norm to `temp[0]`. Because `temp === y.y`,
   it overwrote the cone head before the projection branch was selected.

Both bugs are repaired. The automatic strategy selector is unchanged, and the
small Lasso cone layouts still select `:gridWise`.

The corrected GPU solver reaches `solved_verified` at iteration 500 on the
cached tiny Lasso instance. The direct production-kernel regression passes all
28 SOC comparisons against `blockWise`.

## 2. Machine used for the final verification

```text
Repository:       /home/zhenwei/PDCS_fork
GPU:              NVIDIA H100 80GB HBM3 (GPU 7)
Driver:           595.71.05
Julia:            1.12.6, installed locally under .julia-bin/
Julia CUDA:       13.2.0
CUDA.jl cuBLAS:   /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so
Grid-wise .so:    libcublas.so.13 => /usr/local/cuda/lib64/libcublas.so.13
GPU architecture: sm_90
```

The final shared library and CUDA.jl therefore use the same cuBLAS major
version and the same local CUDA installation.

Verification commands:

```bash
cd /home/zhenwei/PDCS_fork

./.julia-bin/julia --version
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader

CUDA_VISIBLE_DEVICES=7 \
JULIA_DEPOT_PATH=/home/zhenwei/PDCS_fork/.julia-depot \
PDCS_SKIP_GPU_PRECOMPILE=1 \
./.julia-bin/julia -O1 --project=. -e '
  using CUDA
  println("CUDA.jl runtime: ", CUDA.runtime_version())
  println("libcublas: ", CUDA.CUBLAS.libcublas)
  println("device: ", CUDA.name(CUDA.device()))
'

ldd src/pdcs_gpu/cuda/libfew_block_proj.so | \
  grep -E 'cublas|cudart|libstdc'
```

## 3. Original failure modes

### 3.1 `DivideError` in grid-wise SOC projection

On another H100 machine, pure CUDA-kernel projection types succeeded, while
the SOC types that called cuBLAS failed:

| Projection code | Meaning | Native path | Result |
| ---: | --- | --- | --- |
| 0 | dual free | no-op | passed |
| 3 | nonnegative | CUDA kernel | passed |
| 5 | SOC, constant scale | `cublasDnrm2`/`cublasDscal` | `DivideError` |
| 17 | box | CUDA kernel | passed |
| 20 | SOC | `cublasDnrm2`/`cublasDscal` | `DivideError` |

The Julia stack ended at `few_block_proj`, while the underlying native failure
occurred when the C/CUDA shared library entered cuBLAS.

That machine mixed a system CUDA 12.5 build of
`libfew_block_proj.so` with a Julia CUDA 12.9 runtime. A non-null handle is not
enough: the handle creator and all users must use a compatible cuBLAS ABI and
CUDA context.

### 3.2 GPU Lasso failed to converge with grid-wise enabled

After the handle was made valid and the library could run, the tiny Lasso
still exhibited an incorrect primal residual:

```text
iter 500:  relative primal residual = 1.715e-01
iter 2000: relative primal residual = 1.715e-01
iter 5000: relative primal residual = 1.715e-01
```

The objective and dual residual looked nearly converged, but the primal
residual remained fixed. This was a correctness error, not merely slower
convergence.

Raw failing log:

```text
benchmark/lasso_results/
  cupdcs_gpu_gridwise_cublas_fixed_20260728.raw.log
```

### 3.3 Block-wise diagnostic succeeded

A diagnostic run that forced all relevant projections to use `blockWise`
reached `solved_verified`:

```text
iteration:                    500
relative primal residual:     3.1198e-04
relative dual residual:       9.7559e-06
relative gap:                 6.5441e-05
```

Raw diagnostic log:

```text
benchmark/lasso_results/
  cupdcs_gpu_blockwise_diagnostic2_20260728.raw.log
```

This diagnostic was used only to isolate the grid-wise implementation. No
production selector bypass was retained.

## 4. Reproduction instance and preprocessing parity

All CPU and GPU checks use the same serialized instance:

```text
benchmark/results/rebuttal/ill_conditioned_lasso/
  cupdcs_tiny_20260728/instances/seed_2026_K_1.jls
```

The instance has:

```text
variables:                  2201
constraints:                402
nonzeros in G:              22362
equality constraints:       200
SOC constraints:            1
SOC block dimension:        202
```

The CPU and GPU preprocessing values agree to floating-point roundoff:

| Quantity | CPU | GPU |
| --- | ---: | ---: |
| `norm(G, 1)` before scaling | 5583.739379175164 | 5583.739379175161 |
| maximum `Dr_product` | 1.4469334990767284 | 1.4469334990767284 |
| maximum `Dl_product` | 38.63409423734253 | 38.63409423734254 |
| `norm(G, Inf)` after scaling | 0.7071067811865476 | 0.7071067811865475 |

This ruled out random instance generation, serialization, host/device transfer,
and Ruiz--Pock--Chambolle scaling.

## 5. Why restart was not the root cause

The following logs use the same instance and print the first iterations:

```text
benchmark/lasso_results/cupdcs_cpu_tiny_first100_20260728.raw.log
benchmark/lasso_results/cupdcs_gpu_tiny_first100_20260728.raw.log
```

The GPU trace had already diverged at iterations 10, 20, and 100. Its first
restart did not occur until much later. At iteration 10, for example, the CPU
reported an L2 primal residual of `2.186e-02`, while the GPU reported an
infinity-norm primal residual of `1.206e-01`. These two norms are not directly
comparable as a ratio, but the iterates and objectives were already different
before any restart.

Therefore:

```text
incorrect projection
    -> incorrect convergence/restart quantities
    -> later restart behavior amplifies the discrepancy
```

Restart was a downstream symptom, not the initial cause.

Diagnostic warning: lowering `print_freq` changes when this solver checks
termination. It is useful for tracing early iterations, but should not be used
as the final accuracy benchmark configuration.

## 6. Non-projection components that were ruled out

The GPU update kernels were compared independently with the corresponding CPU
formulas in Float64:

```text
UTILITY_KERNEL_ERRORS
  primal                 2.22e-16
  extrapolation          0
  dual                   4.44e-16
  reflection_x           4.44e-16
  reflection_y           8.88e-16
  mean_x                 2.22e-16
  mean_y                 3.33e-16
```

The cuSPARSE matrix-vector products also agreed:

```text
CUSPARSE_MV_ERRORS
  normal operation       1.07e-14
  transpose operation    2.66e-15
```

Thus the primal/dual update PTX and sparse matrix multiplication were not the
source of the early difference.

## 7. Grid-wise call path

For the small Lasso cone layouts, the normal call path is:

```text
select_projection_strategy(...)
    -> :gridWise
    -> gridWise_*_proj!
    -> few_block_proj(...) in Julia
    -> @ccall few_block_proj(...) in libfew_block_proj.so
    -> SOC CUDA/cuBLAS implementation
```

Relevant projection codes are:

| Code | Projection |
| ---: | --- |
| 2 | zero cone |
| 5, 7, 20, 21 | unscaled or constant-scale SOC |
| 6, 22 | diagonally rescaled SOC |

The direct Lasso dual-layout regression uses the actual `[zero, SOC]` block
structure and the production projection codes.

## 8. cuBLAS handle lifecycle fix

### 8.1 Old behavior and risk

Older revisions either left the module-level handle undefined or created it
inside `libfew_block_proj.so`. The latter approach can become invalid when
CUDA.jl and the shared library resolve different cuBLAS runtimes.

Creating and destroying a cuBLAS handle for every projection would also be
unacceptably expensive because projection is called repeatedly throughout the
optimization.

### 8.2 Final ownership model

The final implementation in `src/pdcs_gpu/gpu_kernel.jl`:

1. creates the handle through `CUDA.CUBLAS.cublasCreate()` on the first
   grid-wise projection;
2. stores it in `_gridWise_cublas_handle`;
3. protects lazy initialization with a `SpinLock`;
4. reuses the exact same handle for all later projection calls;
5. destroys it through `CUDA.CUBLAS.cublasDestroy_v2`;
6. registers one Julia `atexit` callback;
7. provides `release_gridWise_cublas_handle!()` for explicit teardown after
   the final solve.

The native shared library no longer exports or calls
`create_cublas_handle_inner` or `destroy_cublas_handle_inner`.

`PDCS_SKIP_GPU_PRECOMPILE=1` skips only the optional GPU warm-up solve.
Julia still invokes the module's `__init__()` normally, so the native
`few_block_proj` symbol and the cleanup callback are initialized. A manual
call to `PDCS_GPU.__init__()` is not required.

### 8.3 Stream ordering

CUDA.jl may enqueue work on a task-local stream, while the CUDA C++ kernels and
Thrust operations in `libfew_block_proj.so` use the legacy default stream.
The cuBLAS handle remains on that default stream so the native kernels and
cuBLAS reductions are ordered with each other.

`few_block_proj` calls `CUDA.synchronize()` before entering the shared library
and again after the native projection. This prevents a race between pending
CUDA.jl work and default-stream native work.

### 8.4 ABI requirement on a reproduction machine

The shared library must be rebuilt against the CUDA toolkit compatible with
the runtime reported by CUDA.jl. Always compare:

```bash
julia --project=. -e 'using CUDA; println(CUDA.CUBLAS.libcublas)'
ldd src/pdcs_gpu/cuda/libfew_block_proj.so | grep cublas
```

Do not copy a CUDA 12.x-built `.so` into an environment where CUDA.jl uses
CUDA 13.x. Rebuild it on the target machine.

## 9. Diagonal SOC root-finding defects

The old `soc_proj_diagonal` implementation contained four correctness risks.

### 9.1 Root bounds aliased persistent scaling data

The old code used:

```cpp
double *xiRight_gpu = D_scaled_gpu;
double *xiLeft_gpu = D_scaled_squared_gpu;
```

The bisection then overwrote the first entries of `D_scaled` and
`D_scaled_squared`, although these arrays are persistent solver scaling data.
Later projections therefore received corrupted diagonals.

The repaired implementation uses two independent device scalars, `xi_left`
and `xi_right`, and never stores root bounds in the scaling arrays.

### 9.2 Wrong negative-head bisection update

For `t < 0`, the scalar oracle is increasing on `(1/2, +inf)`. Therefore:

```text
f(midpoint) < 0  -> root is to the right -> left = midpoint
f(midpoint) >= 0 -> root is to the left  -> right = midpoint
```

The old implementation updated the opposite bound. The corrected
`binary_search_case1` follows the monotonicity above and receives its
`xi_left`, `xi_right` arguments in the correct order.

### 9.3 Stale device flags

The return flag is now reset before each feasibility/oracle evaluation.
Otherwise a true value from an earlier check could terminate a later search
without evaluating the current midpoint.

### 9.4 Unbounded root-search loops and pointer mode

The bisection and right-bound expansion loops are capped at 256 iterations.
The feasible early-return branch also restores the cuBLAS pointer mode to
`CUBLAS_POINTER_MODE_DEVICE`.

The large-vector work remains a cuBLAS implementation:

```text
cublasDnrm2  -> vector norm/reduction
cublasDscal  -> vector scaling
small CUDA kernels -> scalar oracle and solution recovery
```

This fix did not replace the grid-wise algorithm with a serial projection
kernel.

Implementation note: the diagonal SOC native path still allocates a few small
device scalars/flags per native call. These are not cuBLAS handles and do not
change handle reuse. They are a possible future performance optimization, not
a remaining correctness issue.

## 10. Decisive Lasso bug: `temp` aliased the projected SOC

The Lasso nonconvergence remained after the diagonal root-finding fixes. The
decisive bug was in `few_con_proj!`.

The old call passed the same array for every vector argument:

```julia
few_block_proj(
    y.y, y.y, y.y, y.y, y.y, y.y, y.y,
    ...
)
```

For an unscaled SOC, the native routine performs:

```cpp
cublasDnrm2(handle, len, sol + 1, 1, temp);
```

With `temp === sol === y.y`, the operation is effectively:

```text
temp[0] = norm(sol[1:end])
sol[0]  = temp[0]
```

Thus the original cone head `t = sol[0]` was destroyed before
`soc_proj_scale_kernel` tested whether the point was:

- already in the SOC;
- in the polar cone; or
- in the interpolation region.

This projection is used when constructing convergence, restart, and KKT
quantities. The resulting residuals could therefore remain incorrect even
when the main primal/dual iterates appeared numerically close to a solution.

### 10.1 Final workspace fix

`few_con_proj!` now obtains a separate `CuArray` workspace:

```julia
workspace = few_con_workspace(y)
few_block_proj(..., workspace, ...)
```

The cache is a `WeakKeyDict{dualVector,CuArray}`:

- one workspace is allocated for each live `dualVector`;
- it is reused across all checks during that solve;
- there is no allocation on every convergence check;
- the workspace can be reclaimed after its owning solver vector is no longer
  live;
- repeated independent solves do not retain every old GPU workspace forever.

This fixed the Lasso without changing the projection strategy.

## 11. Files changed for this repair

| File | Purpose |
| --- | --- |
| `src/pdcs_gpu/gpu_kernel.jl` | Julia-owned lazy handle, reuse, teardown, and native-call synchronization |
| `src/pdcs_gpu/PDCS_GPU.jl` | registers Julia-side handle cleanup in `__init__()` |
| `src/pdcs_gpu/cuda/few_block_proj.cu` | cuBLAS diagonal-SOC root-finding corrections and removal of native handle ownership |
| `src/pdcs_gpu/def_rpdhg_gen.jl` | independent weak-key cached workspace for `few_con_proj!` |
| `test/test_gridwise_lazy_handle_gpu.jl` | handle creation/reuse/release regression |
| `test/test_gridwise_diagonal_soc_gpu.jl` | production grid-wise SOC correctness regression |
| `src/pdcs_gpu/cuda/libfew_block_proj.so` | rebuilt production shared library |

`src/pdcs_gpu/projection_strategy.jl` was not changed as part of this repair.

## 12. Direct GPU regression coverage

The permanent regression test is:

```text
test/test_gridwise_diagonal_soc_gpu.jl
```

It compares the production `gridWise` shared-library path against the
production `blockWise` PTX path.

Coverage:

| Test set | Cases | Result |
| --- | ---: | --- |
| diagonal SOC, negative/zero/positive head | 6 | 6 passed |
| exact Lasso `[zero, SOC]` layout | 2 | 2 passed |
| heterogeneous diagonal SOC, sigma 1/3/6 | 15 | 15 passed |
| unscaled SOC, heads -3/-0.3/0/0.3/3 | 5 | 5 passed |
| **Total** | **28** | **28 passed** |

Representative maximum absolute differences:

```text
diagonal SOC, head -0.3:     5.551115123125783e-17
diagonal SOC, head  0.0:     1.1102230246251565e-16
diagonal SOC, head  8.0:     1.6653345369377348e-16
heterogeneous cases:         <= 8.881784197001252e-16
unscaled SOC cases:          <= 8.881784197001252e-16
```

Complete test output:

```text
benchmark/lasso_results/
  gridwise_diagonal_soc_regression_20260728.raw.log
```

The handle lifecycle test additionally verifies:

1. importing/loading does not eagerly create the handle;
2. the first grid-wise call creates it;
3. the next grid-wise call reuses the identical handle object;
4. explicit release clears the cached handle.

Its final result was:

```text
gridWise lazy Julia-owned cuBLAS handle: 8/8 passed
```

Complete output:

```text
benchmark/lasso_results/
  gridwise_lazy_handle_regression_20260728.raw.log
```

## 13. Final Lasso verification

The fixed grid-wise GPU run:

```text
iteration:                         500
exit status:                       optimal
benchmark verification status:    solved_verified
L-infinity relative primal res:    3.1277e-04
L-infinity relative dual res:      7.4528e-06
relative gap:                      2.2775e-05
primal objective:                  4.4645e-02
dual objective:                    4.4669e-02
solver-reported time:              7.0434 s
projection time:                   0.9205 s
```

The outer wall time was `50.845 s`, which includes Julia startup,
precompilation/JIT, model loading, and post-solve verification. It must not be
reported as the solver kernel time.

Raw fixed GPU log:

```text
benchmark/lasso_results/
  cupdcs_gpu_gridwise_con_workspace_final_20260728.raw.log
```

For comparison, the CPU solver on the same cached instance and tolerance:

| Quantity | CPU | fixed GPU grid-wise |
| --- | ---: | ---: |
| iterations | 500 | 500 |
| relative primal residual, infinity norm | 3.1237e-04 | 3.1277e-04 |
| relative dual residual, infinity norm | 2.3914e-07 | 7.4528e-06 |
| relative gap | 1.6118e-06 | 2.2775e-05 |
| verification | `solved_verified` | `solved_verified` |

The CPU and GPU reductions are not expected to be bitwise identical, but both
now satisfy the requested `1e-3` verification tolerance and the formerly
stalled GPU primal residual is gone.

## 14. Exact rebuild and regression commands

### 14.1 Select the CUDA toolkit

First inspect the runtime CUDA.jl will use:

```bash
cd /home/zhenwei/PDCS_fork

JULIA_DEPOT_PATH=/home/zhenwei/PDCS_fork/.julia-depot \
./.julia-bin/julia --project=. -e '
  using CUDA
  println(CUDA.runtime_version())
  println(CUDA.CUBLAS.libcublas)
'
```

For CUDA 13.2 on the current machine:

```bash
make -C src/pdcs_gpu/cuda rebuild-few \
  CUDA_HOME=/usr/local/cuda \
  ARCH=sm_90
```

For a CUDA 12.6 machine:

```bash
make -C src/pdcs_gpu/cuda rebuild-few \
  CUDA_HOME=/usr/local/cuda-12.6 \
  ARCH=sm_90
```

Use `ARCH=sm_80` for A100. After building, verify the dependency:

```bash
ldd src/pdcs_gpu/cuda/libfew_block_proj.so | grep cublas
```

`rebuild-few` rebuilds the production grid-wise shared library only. It does
not enable the root-search profiling counters.

### 14.2 Run the SOC regression

```bash
cd /home/zhenwei/PDCS_fork

CUDA_VISIBLE_DEVICES=7 \
JULIA_DEPOT_PATH=/home/zhenwei/PDCS_fork/.julia-depot \
PDCS_SKIP_GPU_PRECOMPILE=1 \
timeout 180s \
./.julia-bin/julia -O1 --project=. \
  test/test_gridwise_diagonal_soc_gpu.jl
```

Expected final summaries:

```text
gridWise diagonal SOC agrees with blockWise                  6/6
gridWise Lasso dual layout agrees with blockWise             2/2
gridWise diagonal SOC under heterogeneous scaling           15/15
gridWise unscaled SOC agrees with blockWise                   5/5
```

### 14.3 Run the handle lifecycle test

```bash
CUDA_VISIBLE_DEVICES=7 \
JULIA_DEPOT_PATH=/home/zhenwei/PDCS_fork/.julia-depot \
PDCS_SKIP_GPU_PRECOMPILE=1 \
timeout 180s \
./.julia-bin/julia -O1 --project=. \
  test/test_gridwise_lazy_handle_gpu.jl
```

### 14.4 Run the fixed tiny Lasso

```bash
mkdir -p benchmark/lasso_results

CUDA_VISIBLE_DEVICES=7 \
JULIA_DEPOT_PATH=/home/zhenwei/PDCS_fork/.julia-depot \
PDCS_SKIP_GPU_PRECOMPILE=1 \
timeout 180s \
./.julia-bin/julia -O1 --project=. \
  benchmark/ill_conditioned_lasso.jl \
  --mode solve \
  --cache benchmark/results/rebuttal/ill_conditioned_lasso/cupdcs_tiny_20260728/instances/seed_2026_K_1.jls \
  --solver pdcs_gpu \
  --tol 1e-3 \
  --time-limit 120 \
  --verbose-level 2 \
  --print-freq 500 \
  --output-dir benchmark/results/rebuttal/ill_conditioned_lasso/cupdcs_gpu_fixed \
  > benchmark/lasso_results/cupdcs_gpu_fixed.raw.log 2>&1
```

Use `-O1` on Julia installations that spend a long time in LLVM GPU-kernel
compilation. This changes Julia compilation behavior, not the projection
formula or benchmark tolerance.

## 15. Expected failure signatures when reproducing elsewhere

### Immediate `DivideError` on SOC types only

Check:

```bash
julia --project=. -e 'using CUDA; println(CUDA.CUBLAS.libcublas)'
ldd src/pdcs_gpu/cuda/libfew_block_proj.so | grep cublas
```

Rebuild the shared library with the matching toolkit. Do not manually call
`PDCS_GPU.__init__()` as a workaround.

### `few_block_proj not initialized`

Confirm that `using PDCS` loads the same source tree and shared library:

```bash
julia --project=. -e '
  using PDCS
  println(PDCS.PDCS_GPU.few_block_proj_ptr[])
'
```

The pointer must be non-null. `PDCS_SKIP_GPU_PRECOMPILE=1` is safe and does not
disable `__init__()`.

### SOC direct tests pass but Lasso primal residual stays near `1.7e-1`

Confirm that `few_con_proj!` passes an independent workspace, not `y.y`, as
the `temp` argument. This was the decisive Lasso-specific alias bug.

### Test process appears idle before producing output

The first run may spend tens of seconds precompiling PDCS and CUDA code. Check
the Julia process before concluding that the GPU kernel is hung. Retain `-O1`
and allow precompilation to complete once.

## 16. Remaining scope and non-issues

- The fix covers the grid-wise SOC paths used by the Lasso case and their
  diagonal/unscaled regression cases.
- Grid-wise RSOC branches still print that their cuBLAS implementation is
  under development; the strategy selector already excludes the small-layout
  RSOC codes from `gridWise`.
- Profiling PTX files are separate from production PTX. They are not involved
  in this Lasso repair.
- The production strategy heuristic remains unchanged.
- The `use_accelerated` compatibility parameter is unrelated to this bug.
- No benchmark result or earlier raw log was deleted during debugging.

## 17. Final acceptance checklist

- [x] Julia owns the cuBLAS handle.
- [x] The handle is lazily created once and reused across iterations.
- [x] The handle is released by Julia at exit or explicitly after the solve.
- [x] The shared library does not create/destroy the handle.
- [x] CUDA.jl and `libfew_block_proj.so` use compatible cuBLAS libraries.
- [x] Grid-wise diagonal SOC no longer overwrites scaling arrays.
- [x] Negative-head bisection updates the correct bound.
- [x] `few_con_proj!` no longer aliases the SOC input and cuBLAS workspace.
- [x] Workspace memory is reused and weakly retained.
- [x] Direct grid-wise SOC regression passes 28/28.
- [x] Tiny Lasso uses the normal grid-wise strategy.
- [x] Tiny Lasso reaches `solved_verified`.
- [x] CPU/GPU primal residuals agree at the requested `1e-3` level.
- [x] All raw diagnostic and final logs are retained.
