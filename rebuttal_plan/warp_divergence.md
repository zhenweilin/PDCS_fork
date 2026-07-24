# R1-3: warp divergence and GPU utilization in root-finding projections

> **Execution status (2026-07-22):** Controlled timing experiments were implemented and run on GPU 7. See `warp_divergence_results.md` and `benchmark/rescaled_soc_warp_profile.jl`. Nsight counters are blocked on this host by `ERR_NVGPUCTRPERM`; the profiling procedure below is ready for an unrestricted machine. Existing projection results were not removed.

## Objective

Measure whether data-dependent root finding in diagonally rescaled SOC
projections causes harmful warp divergence, and determine whether the GPU is
still used efficiently. The experiment must separate four effects that can
otherwise be confused:

1. different cones taking different top-level projection branches;
2. different root-search iteration counts across cones;
3. insufficient parallel work or occupancy;
4. a memory-bandwidth bottleneck rather than a compute bottleneck.

The experiment will report both direct divergence metrics and utilization. A
single `nvidia-smi` utilization percentage is not sufficient for the
conclusion.

## Relevant implementation

The rescaled SOC projection has projection code `22`. In
`src/pdcs_gpu/cuda/sufficient_block_proj.cu`, `soc_proj_diagonal` contains:

- early returns for points already in the cone or its polar cone;
- separate positive-`t`, negative-`t`, and near-zero branches;
- optional warm-started Newton steps;
- data-dependent interval expansion and bisection loops;
- stopping tests controlled by `abs_tol` and `rel_tol`.

The GPU mapping matters when interpreting divergence:

- `massive`/thread-wise assigns one cone to each CUDA thread. Thirty-two
  different cones share a hardware warp, so different branches or iteration
  counts cause true intra-warp divergence.
- `sufficient`/warp-wise assigns one complete hardware warp to each cone. The
  scalar root-search decision is shared by the lanes cooperating on that cone;
  cone-to-cone iteration differences occur between warps rather than within a
  warp. Its remaining lane inefficiency mainly comes from reductions and a
  cone dimension that is not a multiple of 32.
- `moderate`/block-wise assigns one block to each cone. It is a useful
  utilization reference but is not the primary divergence target.

This distinction must be stated in the manuscript. “Different cones require
different iterations” does not automatically imply intra-warp divergence for
the warp-wise implementation.

## Experimental hypotheses

H1. Similar scaling and nearby parametric inputs produce similar branch paths
and bisection counts. Thread-wise branch uniformity should therefore remain
high, and warm starts should further reduce iteration variance.

H2. Independently heterogeneous scaling increases per-warp iteration spread
and reduces thread-wise branch efficiency.

H3. For the same multiset of cones, grouping cones of similar difficulty into
the same warp should outperform deliberately interleaving easy and difficult
cones. This paired ordering experiment isolates divergence without changing
the mathematical workload.

H4. Warp-wise projection should be less sensitive to cone-to-cone root-search
variation than thread-wise projection, because each warp owns only one cone.

H5. High device utilization can coexist with some divergence. The effect is
material only if lower branch uniformity is accompanied by lower issue
efficiency/throughput and increased runtime.

## Benchmark harness to implement

Add a dedicated GPU-only harness, for example
`benchmark/rescaled_soc_warp_profile.jl`. It must call projection type `22`
directly, not time an entire PDCS solve. This isolates the projection kernels
from sparse matrix multiplication and termination checks.

The harness must:

- generate Float64 rescaled-SOC blocks directly on the GPU;
- keep cone count, dimension, inputs, scaling, tolerances, and seeds explicit;
- run forced `massive`, `sufficient`, and `moderate` strategies, while marking
  which strategy the solver heuristic would select;
- perform unprofiled warm-up launches before measurement;
- execute a configurable number of identical projection launches so short
  kernels are long enough for device-utilization sampling;
- retain `t_warm_start` between parametric projections in the warm-start cases;
- write raw timing and correctness CSVs directly to a run directory;
- print NVTX ranges named by case and strategy so Nsight Systems/Compute can
  select only the projection region;
- provide `--profile-one` mode that performs one preselected launch and exits,
  making Nsight Compute replay predictable;
- provide `--duration 10` mode that loops the same projection workload for at
  least ten seconds for coarse utilization sampling.

All inputs must be reproducible from a recorded base seed. For a given case and
replicate, every strategy receives exactly the same vector, scaling, cone
ordering, and warm-start state.

## Optional diagnostic instrumentation

Create a separate profiling build guarded by a compile-time macro such as
`PDCS_PROFILE_ROOT_SEARCH`. It must not be enabled for timing or Nsight
utilization runs.

For each cone, record:

- top-level path: already feasible, polar/zero, positive-`t` root search,
  negative-`t` root search, or near-zero path;
- number of interval-expansion iterations;
- number of attempted and accepted Newton steps;
- bisection iterations;
- total oracle evaluations;
- whether `MAX_ITER` was reached;
- final residual and warm-start hit/miss.

Use one output slot per cone and avoid global atomics in the root-search loop.
Copy counters to the CPU after projection. Diagnostic counter results describe
the workload; they must not be mixed with timings collected from the
uninstrumented binary.

From these counters compute, for every group of 32 consecutive cones:

- minimum, median, maximum, standard deviation, and max-minus-min iteration
  count;
- fraction of lanes taking the modal top-level branch;
- branch-path entropy;
- fraction of lanes with the same iteration count;
- a model-based active-lane efficiency
  `sum(iterations) / (32 * maximum(iterations))` for thread-wise execution.

The model-based value is supporting evidence, not a substitute for hardware
metrics.

## Controlled input regimes

Use ordinary rescaled SOCs, not rotated or exponential cones. Construct inputs
so root search is actually exercised; do not rely solely on unconstrained
Gaussian vectors, which may overwhelmingly select one early-return path.

For each cone, let the diagonal scale be log-spaced around one. Define these
regimes:

### A. Uniform root-search baseline

- All cones use the same diagonal scaling vector.
- Inputs are constructed just outside the rescaled cone boundary.
- Add only a tiny per-cone perturbation (`1e-6` relative magnitude).
- All cones use the positive-`t` root-search branch.

This is the minimum-divergence baseline.

### B. Parametric-similar workload

- Begin from one base scaling and one base vector.
- Perturb each cone by 1% relative magnitude.
- Use log-scale standard deviation `0.05` across cones.
- Evaluate both cold starts (`t_warm_start = 0`) and warm starts obtained from
  the preceding nearby parameter instance.

This models the reviewer’s similar-scaling parametric setting.

### C. Heterogeneous root-only workload

- Force every cone into root search, avoiding early exits.
- Draw per-cone log scaling independently with standard deviations
  `0.5`, `1.0`, and `2.0`.
- Vary boundary distance so different cones need different interval and
  bisection counts.

This produces a controlled divergence curve without changing top-level branch
type.

### D. Mixed-branch stress test

Build an equal multiset containing cones in four classes:

- already feasible;
- polar/zero projection;
- positive-`t` root search;
- negative-`t` root search.

Run the identical multiset in three orderings:

1. grouped: each hardware warp contains one class;
2. randomly shuffled;
3. adversarially interleaved: each warp contains all four classes.

Runtime and hardware differences between these orderings measure branch
divergence while holding total mathematical work fixed.

### E. Iteration-mix ordering test

Calibrate root-search inputs into four bisection-count quartiles using the
instrumented build. Keep all cones on the same top-level branch. Compare
quartile-grouped and quartile-interleaved layouts using the uninstrumented
binary. This isolates loop-exit divergence from top-level branch divergence.

## Problem sizes

Use cone counts that are multiples of 32. The primary saturated configuration
is:

```text
cone_count = 1,048,576
cone_dimension = 10
```

This is the important thread-wise case: it supplies many warps while keeping
each cone small enough that one thread is plausible.

Add dimension sensitivity at fixed cone count `262,144`:

```text
cone_dimensions = 10, 32, 128, 256
```

Dimension 256 ensures the warp-wise method is included where cooperative lane
work is naturally useful. Before the full run, confirm that every case fits in
GPU memory and that the projection kernel lasts long enough to profile. If a
kernel is shorter than 1 ms, increase the number of repeated launches rather
than changing the mathematical case.

## Replication and timing

- Use base seed `2026`.
- Use ten independent seeds for unprofiled runtime statistics.
- Report median, mean, and sample standard deviation.
- Use three representative seeds for Nsight Compute because metric replay is
  expensive: seeds `2026`, `2027`, and `2028`.
- Warm up every kernel and allocate all buffers before timing.
- Time synchronized projection only.
- Run the profiler with no other GPU processes where possible and record GPU
  clocks, power mode, and compute mode.
- Repeat the entire profiling sequence if any primary metric varies by more
  than 5% across the three seeds.

## Correctness checks

Before profiling, validate every generated regime:

- compare sampled cones against the CPU rescaled-SOC projection;
- check finite outputs and absence of `MAX_ITER` failures;
- report maximum infinity-norm error;
- check primal feasibility and the projection KKT residual;
- verify grouped, shuffled, and interleaved layouts are permutations of the
  same cone multiset;
- verify profiling instrumentation does not change projected values.

Do not include an incorrect or nonconverged case in performance summaries.

## Nsight Compute collection

Nsight Compute 2024.3 is available with CUDA 12.6 on the current machine. On a
new machine, locate it with:

```bash
command -v ncu || find "$CUDA_HOME" -type f -name ncu
ncu --version
ncu --list-sets
ncu --query-metrics | grep -E 'branch|warps_active|warps_eligible|throughput'
```

Collect one warmed target kernel per process. Example structure:

```bash
CUDA_VISIBLE_DEVICES=7 ncu \
  --target-processes all \
  --kernel-name regex:sufficient_block_proj \
  --launch-skip 1 --launch-count 1 \
  --section SpeedOfLight \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section SourceCounters \
  --export results/ncu/similar_warm_warp \
  ./.julia-bin/julia --project=. benchmark/rescaled_soc_warp_profile.jl \
    --profile-one --case similar --warm-start --strategy warpWise \
    --cone-count 1048576 --cone-dimension 10 --seed 2026
```

Repeat with native kernel name `massive_block_proj` and strategy `threadWise`.
Add native kernel `moderate_block_proj` with strategy `blockWise` for the
block-wise reference. Use section names rather
than a hard-coded metric list for the primary collection because PerfWorks
metric availability differs by GPU architecture and Nsight version.

Export a CSV after collection:

```bash
ncu --import results/ncu/similar_warm_warp.ncu-rep \
  --page raw --csv > results/ncu/similar_warm_warp.csv
```

Record at least these metrics when supported:

| Quantity | Nsight metric/section |
|:---|:---|
| Branch uniformity | `smsp__sass_average_branch_targets_threads_uniform.pct` |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Eligible warps/cycle | `smsp__warps_eligible.sum.per_cycle_active` or architecture equivalent |
| SM throughput | SpeedOfLight SM throughput percentage |
| Memory throughput | SpeedOfLight memory/DRAM throughput percentage |
| Issued warps/cycle | SchedulerStats |
| Branch-resolving stalls | SourceCounters/WarpStateStats |
| Barrier stalls | SourceCounters/WarpStateStats |
| Average active threads | SourceCounters source-level thread execution columns |

The branch-uniformity metric is the Nsight Compute replacement for legacy
`branch_efficiency`. Resolve exact names from `ncu --query-metrics` and save
that query output with the results.

Nsight Compute replays kernels to collect metrics. Do not use its elapsed time
as the publication runtime; use the separate unprofiled timing run.

If Nsight reports `ERR_NVGPUCTRPERM`, hardware performance counters are
restricted by the system administrator. Do not run with broad privileges by
default. Ask the administrator to grant profiling access according to the
NVIDIA deployment policy, and record that configuration in `environment.txt`.

## Sustained GPU-utilization collection

For the reviewer’s 80–90% utilization question, execute each primary case for
at least ten seconds and collect two views.

First, use Nsight Systems GPU metrics when permissions permit:

```bash
nsys profile --gpu-metrics-device=help
export NSYS_GPU_METRICS_DEVICE=7  # physical device reported by the help command

CUDA_VISIBLE_DEVICES=7 nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-device="$NSYS_GPU_METRICS_DEVICE" \
  --output results/nsys/similar_warm_thread \
  ./.julia-bin/julia --project=. benchmark/rescaled_soc_warp_profile.jl \
    --duration 10 --case similar --warm-start --strategy threadWise \
    --cone-count 1048576 --cone-dimension 10 --seed 2026
```

Second, record coarse device-busy samples during the same ten-second loop:

```bash
nvidia-smi dmon -i 7 -s pucvmet -d 1 \
  > results/nvidia_smi/similar_warm_thread.txt &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES=7 ./.julia-bin/julia --project=. \
  benchmark/rescaled_soc_warp_profile.jl \
  --duration 10 --case similar --warm-start --strategy threadWise \
  --cone-count 1048576 --cone-dimension 10 --seed 2026

kill "$MONITOR_PID"
```

The process-management wrapper should implement cleanup with a shell `trap` so
the monitor is stopped even if Julia fails. On a system where physical GPU
indexing differs from CUDA-visible indexing, record both the GPU UUID from
Julia and the UUID reported by `nvidia-smi`.

## Primary reported metrics

For every regime, strategy, and dimension report:

- unprofiled median projection time and variability;
- branch uniformity percentage;
- achieved occupancy percentage;
- SM SOL throughput percentage;
- memory SOL throughput percentage;
- eligible and issued warps per cycle;
- branch-resolving and barrier stall shares;
- `nvidia-smi` mean, median, and 10th/90th-percentile GPU busy percentage;
- root-search iteration median, 90th percentile, maximum, and per-warp spread;
- model-based active-lane efficiency;
- warm-start hit rate and `MAX_ITER` count.

Define effective hardware utilization as

```text
max(SM SOL throughput, memory SOL throughput)
```

because a projection can use the GPU efficiently while being memory-bound.
Report SM and memory values separately as well. Do not equate achieved
occupancy with utilization: occupancy measures resident active warps, while
SOL throughput measures use of compute or memory resources.

## Analysis and statistical comparisons

Use paired comparisons because layouts and strategies receive the same cone
multiset and seed.

For each heterogeneous case report:

- runtime ratio relative to the uniform baseline;
- branch-uniformity change;
- effective-utilization change;
- eligible-warps/cycle change;
- correlation between per-warp iteration spread and runtime;
- correlation between branch uniformity and runtime;
- grouped-versus-interleaved runtime ratio for the identical multiset;
- cold-versus-warm runtime and iteration-count ratios.

Summarize three seeds for hardware metrics with mean and range. Summarize ten
unprofiled timing seeds with mean, sample standard deviation, and median. Do
not calculate significance tests from Nsight replay passes as if replay passes
were independent observations.

## Interpretation rules and acceptance criteria

The experiment supports “efficient utilization under similar scaling” when
all of the following hold for the primary parametric-similar case:

- branch uniformity is at least 90%, or the source view shows no material
  divergence in the root-search lines;
- effective hardware utilization is at least 80%;
- no scheduler starvation is visible from eligible/issued warp metrics;
- no correctness or convergence failure occurs;
- warm-started runtime is not worse than cold-started runtime outside measured
  variability.

The reviewer’s suggested 80–90% threshold should apply to effective SOL
throughput or sustained GPU busy, not necessarily to SM throughput alone. A
memory-bound kernel may have low SM SOL and high memory SOL while still using
the GPU effectively.

Evidence of harmful divergence requires all three:

1. lower branch uniformity or active-lane efficiency;
2. lower issue efficiency/effective utilization;
3. higher unprofiled runtime for the paired identical-work layout.

If utilization is below 80%, report it rather than adjusting the experiment to
hide it. Determine whether the limiting factor is divergence, insufficient
parallelism, memory bandwidth, or occupancy. If the grouped/interleaved test
shows a meaningful difference, discuss grouping cones with similar scaling or
expected iteration counts as a possible future optimization; do not change
the production ordering solely for the rebuttal experiment.

## Output artifacts

Write results under `benchmark/results/warp_divergence/<run-id>/`:

```text
environment.txt
case_manifest.csv
timings_raw.csv
timings_summary.csv
root_iterations_raw.csv
root_iterations_summary.csv
correctness.csv
ncu/*.ncu-rep
ncu/*.csv
nsys/*.nsys-rep
nvidia_smi/*.txt
profile_metrics.csv
```

Generate:

1. a table of timing, branch uniformity, occupancy, SM SOL, memory SOL, and
   effective utilization for the uniform, similar, and heterogeneous cases;
2. a plot of branch uniformity and runtime versus scaling heterogeneity;
3. a paired grouped-versus-interleaved plot for branch and iteration mixes;
4. an iteration-count distribution or per-warp spread plot;
5. a concise reviewer response and manuscript paragraph populated only from
   measured values.

## Proposed reviewer response template

> We thank the reviewer for highlighting possible warp divergence in the
> root-finding projections. We added a controlled study of diagonally rescaled
> SOC projection using NVIDIA Nsight Compute and Nsight Systems. We distinguish
> the thread-wise kernel, where 32 different cones share a warp, from the
> warp-wise kernel, where all lanes cooperate on one cone and cone-to-cone
> iteration variation occurs between warps. We profile similar-scaling,
> independently heterogeneous, and adversarially interleaved workloads, and
> report branch uniformity, achieved occupancy, SM and memory throughput,
> eligible warps, GPU busy time, root-search iteration distributions, and
> unprofiled runtime. [Insert measured similar-scaling utilization and branch
> uniformity.] [Insert the paired grouped/interleaved result.] These results
> show [measured conclusion], while the deliberately heterogeneous case
> quantifies the degradation when cones within a warp follow different paths.

## References for metric interpretation

- NVIDIA Nsight Compute 2024.3 CLI metric mapping:
  https://docs.nvidia.com/nsight-compute/2024.3/NsightComputeCli/index.html
- NVIDIA Nsight Compute Profiling Guide:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

NVIDIA defines SpeedOfLight sections as achieved compute/memory throughput
relative to theoretical maximum, SourceCounters as the source of branch
efficiency and warp-stall data, and achieved occupancy as active warps relative
to the hardware maximum. These definitions should be used verbatim when
interpreting the collected results.
