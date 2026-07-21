# GPU cone-projection strategy

PDCS chooses a projection kernel once, while constructing each primal or dual
block vector. Let `B` be the total number of blocks and let `d_max` be the
largest structured-cone dimension (the first two entries in the internal block
layout are scalar/simple cones). The empirical selection algorithm is:

```text
if B <= 3 and the blocks contain no rotated SOC:
    strategy = few
else if B <= 1,000 or d_max >= 2,000:
    strategy = moderate
else if B <= 60,000 or d_max >= 150:
    strategy = sufficient
else:
    strategy = massive
```

The strategies expose progressively different levels of the GPU hierarchy.
`few` launches work cone by cone; `moderate` assigns one CUDA block to each
cone; `sufficient` assigns a warp to each cone; and `massive` assigns one CUDA
thread to each cone. The two size conditions deliberately use **or**: a large
cone needs cooperative threads even when the problem contains many cones.

These thresholds are performance choices rather than correctness conditions.
Run `julia --project=. benchmark/random_soc_projection.jl` to sweep random SOC
counts and dimensions on a target GPU. The script verifies every kernel against
the closed-form Euclidean SOC projection before reporting median timings as CSV.

## Selecting the CUDA toolkit

The native `few` projection library can be compiled with a toolkit that differs
from the system default. For example, on a machine where `/usr/local/cuda`
points to CUDA 13.2, build a CUDA 12.6 library for an H100 with:

```bash
make -C src/pdcs_gpu/cuda print-config \
  CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90
make -C src/pdcs_gpu/cuda rebuild-few \
  CUDA_HOME=/usr/local/cuda-12.6 ARCH=sm_90
```

`CUDA_HOME` chooses `nvcc`; `ARCH` chooses the generated GPU machine code. The
NVIDIA driver is backward compatible, so a sufficiently recent CUDA 13.x driver
can execute a library built with the CUDA 12.6 toolkit. The library creates its
cuBLAS handle through its own linked cuBLAS runtime, allowing it to coexist with
CUDA.jl even when CUDA.jl selects a different toolkit version.
