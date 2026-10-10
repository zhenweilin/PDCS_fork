# GPU cone-projection strategy

PDCS uses a deterministic, structure-based heuristic with two stages: select a
baseline GPU mapping from the cone types, number of cones, and largest cone
dimension; then, for block-wise or warp-wise execution, optionally batch small
cones using finer mappings. Different cone families in one problem can
therefore use different levels of the GPU hierarchy.

The baseline is selected during solver setup, separately for the primal,
slack, and dual block layouts. The batching plan is constructed on first use
and cached for each projection layout. Both decisions depend on fixed layout
metadata and are reused during iterations; the solver does not time competing
kernels or search for a new mapping during the solve.

| Mapping | Work assignment | Legacy kernel name |
|---|---|---|
| Grid-wise | Multiple CUDA blocks cooperate on a cone; cones are processed in sequence | `few` |
| Block-wise | One CUDA block cooperates on each cone | `moderate` |
| Warp-wise | One warp cooperates on each cone | `sufficient` |
| Thread-wise | One CUDA thread processes each cone | `massive` |

These descriptions apply to the structured cones; free, zero, nonnegative,
and box blocks have their own simple projection operations. A warp contains
32 threads, and the current block/warp/thread kernels use 256 threads per
CUDA block.

## Algorithm 1: baseline mapping

The source of this rule is
[`select_projection_strategy`](../src/pdcs_gpu/projection_strategy.jl).
Let `B` be the number of blocks in the internal layout. Its first two entries
are reserved for simple blocks and are excluded from the structured-cone
statistics. Thus `N = B - 2` is the structured-cone count. For a pure SOC
layout, let `d` be the **maximum** SOC dimension, including the cone's leading
coordinate. SOC dimensions need not be equal.

The default automatic rule is:

```text
SelectBaseline(block_sizes, projection_types):
    B = number of blocks
    if B < 3:
        return Grid

    structured = blocks 3 through B
    if structured contains any cone other than an unrotated SOC:
        return Block

    N = B - 2
    d = maximum dimension among structured cones
    if N == 1:
        if d <= 3:       return Thread
        if d <= 64:      return Warp
        if d >= 32768:   return Grid
        return Block

    if d <= 4:          return Thread
    if d <= 64:
        if N >= 1024*d: return Thread
        return Warp
    if d >= 1024:       return Block
    if N <= 998:        return Block
    return Warp
```

The `B < 3` branch preserves a compatibility path for calls without the
solver's full padded layout. The non-SOC branch includes rotated SOCs (RSOCs),
exponential cones, dual exponential cones, and mixtures of cone families.
Ordinary, scalar-rescaled, and diagonally rescaled SOC projection codes are
all classified as SOCs for this baseline decision.

For multiple SOCs with maximum dimension between 65 and 1023, the count boundary is
exactly `N <= 998`, equivalently `B <= 1000` including the two simple blocks.
For a single dimension-4 SOC, the rule selects Warp; for multiple SOCs with
maximum dimension 4, it selects Thread. These boundary details are part of the
implemented rule.

The rationale is to expose parallelism within a cone when its dimension is
large, and across cones when there are many short projections. A single very
large SOC can use a whole grid. For many small SOCs, assigning one thread per
cone can avoid the cost of reserving a warp for each short reduction. The
thresholds are empirical performance choices, not correctness requirements or
a guarantee of the fastest mapping on every GPU or instance.

## Algorithm 2: batching within block-wise or warp-wise execution

The default implementation also enables a cached batching plan in
[`_get_heterogeneous_projection_plan`](../src/pdcs_gpu/gpu_kernel.jl).
This refinement applies when Algorithm 1 selects Block or Warp. Grid and
Thread keep their baseline execution. The planner examines all blocks,
including the leading simple blocks, and excludes free blocks because their
projection is the identity.

It partitions the remaining blocks into the following disjoint groups:

| Group | Contents | Eligibility for a finer mapping | Mapping if enabled |
|---|---|---|---|
| `S` | Primal/dual exponential cones and simple blocks of dimension at most 32 | At least 512 primal exponential projections in `S`, **or** at least 768 blocks in `S` | One thread per cone or simple block |
| `T` | SOCs of dimension at most 4 | At least 256 SOCs in `T` | One thread per SOC |
| `W` | SOCs of dimension 5 through 32 | At least 64 SOCs in `W` | One warp per SOC |
| `L` | Simple blocks of dimension greater than 32 | Kept as simple projections | One CUDA block per simple block |
| `R` | All remaining cones, including RSOCs and larger SOCs | Keep the baseline mapping | Block or Warp |

Here, a "primal exponential projection" denotes the exponential-cone
projection type, which may occur in either a primal or a dual solver vector.
The SOC groups include diagonally rescaled SOCs with the default compaction
setting. Groups that fail their eligibility threshold retain the baseline
mapping.

Batching is enabled only if at least one group is eligible and a static work
estimate decreases by at least 20%. With `s`, `t`, `w`, `l`, and `r` denoting
the group sizes, the exact gate is:

```text
PackThreads(k, eligible) = ceil(k / 256) if eligible, otherwise k
PackWarps(k, eligible)   = ceil(32*k / 256) if eligible, otherwise k

original_work = s + t + w + l + r
packed_work   = PackThreads(s, S_is_eligible)
              + PackThreads(t, T_is_eligible)
              + PackWarps(w, W_is_eligible) + l + r

if any group is eligible and 5*packed_work <= 4*original_work:
    use the grouped mappings and cache their cone-index lists
else:
    retain the baseline mapping
```

This work estimate is a dispatch heuristic, not a measured runtime reduction
or an exact launch-count model for every baseline. In the block-wise path,
eligible `S` projections are packed one per thread into the same kernel launch
as the remaining block-wise work; eligible small SOC groups use indexed
thread/warp kernels. In the warp-wise path, the grouped projections are
dispatched to indexed simple, thread, and warp kernels. Cone-index lists keep
the original vector layout intact.

For example, a compact Lasso formulation has one SOC of dimension `m + 2`:
`m = 10000` selects Block, while `m = 70000` selects Grid. A Fisher market
layout with many exponential cones selects Block as its baseline, then packs
the exponential projections one per thread when the batching gate passes.
Thus the baseline label alone does not describe every cone's final mapping.

## Reproducibility and validation

The thresholds were selected empirically using the projection benchmarks in
[`benchmark/R3.5/projection_benchmark`](../benchmark/R3.5/projection_benchmark/README.md),
which compare mappings on the same inputs and GPU and include numerical
correctness checks. They are fixed defaults; no per-instance online tuning is
performed. The baseline is assigned by `setFunctionPointerPrimal!` and
`setFunctionPointerDual!` in
[`def_rpdhg_gen.jl`](../src/pdcs_gpu/def_rpdhg_gen.jl), and the cached batching
rules and dispatch reside in [`gpu_kernel.jl`](../src/pdcs_gpu/gpu_kernel.jl).

For controlled comparisons, `PDCS_PROJECTION_STRATEGY_OVERRIDE` accepts
`grid`, `block`, `warp`, or `thread`. It overrides Algorithm 1; batching remains
enabled for Block and Warp unless `PDCS_ENABLE_HETEROGENEOUS_PROJECTION=0` is
also set. Normal automatic selection leaves the strategy override unset.
Grid-wise native-library checks and explicitly configured compatibility
fallbacks are described in the [CUDA runtime documentation](../src/pdcs_gpu/cuda/README.md).

The baseline thresholds can be checked without a GPU:

```bash
julia --startup-file=no test/test_projection_strategy.jl
```

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
