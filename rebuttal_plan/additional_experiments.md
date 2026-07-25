# Comprehensive Rebuttal Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task by task. Track
> execution with the checkboxes in this document.

**Goal:** Produce a reproducible, internally consistent experimental package
that answers Reviewer 1's questions R1-1, R1-2, and R1-3, validates every
projection result, and supports only claims justified by the measured data.

**Architecture:** The work is divided into four layers: (1) timing and data
integrity, (2) controlled projection microbenchmarks, (3) hardware profiling,
and (4) application-level validation. Uninstrumented kernels provide
publication timings; separately compiled diagnostic kernels provide branch and
root-iteration data; a single summarization pipeline generates all tables and
figures from immutable raw files.

**Tech stack:** Julia 1.10.4, CUDA.jl, CUDA Toolkit 12.5 or the toolkit recorded
on the target machine, NVIDIA H100 80 GB (`sm_90`), NVIDIA A100 80 GB
(`sm_80`), CUDA C++, cuBLAS, NVIDIA Nsight Compute, NVIDIA Nsight Systems,
`nvidia-smi`, CSV, Zstandard compression, and LaTeX.

## Global constraints

- Use `Float64` for every vector, scaling array, timing run, and CPU reference.
- Use base seed `2026`; timing repetitions use seeds `2026:2035`.
- Give every strategy exactly the same input, scaling, cone order, tolerance,
  and warm-start state for a paired comparison.
- Perform all allocations, input generation, compilation, and warm-up launches
  outside the measured projection interval.
- Measure uninstrumented projection kernels with synchronized CUDA events.
- Never use profiler-replay elapsed time as a publication runtime.
- Compile diagnostic counters into separate PTX files guarded by
  `PDCS_PROFILE_ROOT_SEARCH`; never enable this macro for timing runs.
- Treat thread-wise ordering comparisons as the primary direct test of
  intra-warp divergence. One complete warp processes one cone in the warp-wise
  implementation, so cone-to-cone ordering effects there are not automatically
  intra-warp divergence.
- Reject results with nonfinite output, a maximum-iteration failure, a CPU/GPU
  infinity-norm discrepancy above `5e-8`, or a failed feasibility/KKT check.
- Record timeouts rather than silently dropping configurations. The
  per-projection timeout is 15 seconds for strategy-map experiments.
- Keep the H100 and A100 results separate. Compare qualitative regimes and
  normalized ratios across architectures; do not pool their absolute times.
- Preserve every raw file used to generate a manuscript number.
- Do not update the manuscript or reviewer response until all mandatory
  acceptance gates in this plan pass.

---

# 1. Questions, hypotheses, and claim boundaries

## 1.1 R1-2: cone dimension and projection strategy

**Question.** At a fixed number of ordinary second-order cones, how does the
full dimension of each cone affect the fastest GPU projection strategy?

**H1.** Thread-wise projection becomes progressively less competitive as cone
dimension grows because one thread traverses an entire cone serially.

**H2.** Warp-wise projection is competitive for small cones, block-wise
projection is preferred over an intermediate region, and grid-wise projection
becomes attractive only when there are sufficiently few and sufficiently large
cones to amortize repeated cuBLAS calls.

**H3.** No single dimension threshold applies to every cone count; the
selection rule must depend jointly on cone count and cone dimension.

**Permitted claim.** The experiment may identify empirical strategy regions on
the tested GPUs. It may not claim an architecture-independent universal
threshold.

## 1.2 R1-3: divergence and utilization

**Question.** Do data-dependent projection branches and root-finding iteration
counts cause harmful warp divergence, especially in similar-scaling parametric
instances, and are the GPU resources used effectively?

**H4.** In a nearby parametric sequence, cones within a thread-wise warp follow
similar projection paths and have small root-iteration spread; runtime should
remain close to the uniform baseline.

**H5.** Warm-starting the scalar root from the preceding, nonidentical
parametric instance reduces root iterations and projection time.

**H6.** Interleaving cones with different top-level projection paths within a
thread-wise warp is slower than grouping the same cone multiset by path.

**H7.** Even when every cone follows the same top-level root-search branch,
interleaving cones from different root-iteration quartiles is slower than
grouping them.

**H8.** High GPU-busy percentage can coexist with divergence. A claim of
effective utilization must therefore use runtime together with SM/memory
throughput, eligible warps, and branch/iteration evidence.

**Permitted claim.** If the gates pass, the paper may state that divergence is
not material for the tested similar-scaling parametric regimes but is
measurable under controlled heterogeneous stress. It may not state that all
root-finding projections are divergence-free.

# 2. Planned repository structure

Create the following focused files under the PDCS repository root:

```text
benchmark/rebuttal/
  common.jl
  generate_soc_cases.jl
  generate_exp_cases.jl
  validate_projection.jl
  timing.jl
  audit_existing_results.jl
  soc_strategy_map.jl
  soc_root_profile.jl
  exp_root_profile.jl
  application_trace.jl
  summarize_strategy_map.jl
  summarize_root_profiles.jl
  summarize_application_traces.jl
  run_h100.sh
  run_a100.sh
  profile_ncu.sh
  profile_nsys.sh
  reproduce_all.sh
  README.md
test/
  test_rebuttal_case_generation.jl
  test_rebuttal_permutations.jl
  test_rebuttal_timing.jl
  test_rebuttal_correctness.jl
src/pdcs_gpu/cuda/
  root_profile.h
  massive_block_proj_profile.cu
  moderate_block_proj_profile.cu
  sufficient_block_proj_profile.cu
benchmark/results/rebuttal/2026-07-comprehensive/
  audit/
  h100-sm90/
  a100-sm80/
  cross-hardware/
  manuscript/
```

Responsibilities:

- `common.jl` defines the command-line schema, seeds, strategy names, run
  manifests, CSV schemas, and environment capture.
- `generate_soc_cases.jl` creates ordinary and diagonally rescaled SOC cases.
- `generate_exp_cases.jl` creates diagonally rescaled exponential-cone cases.
- `validate_projection.jl` runs CPU comparisons, feasibility tests, residual
  tests, permutation checks, and maximum-iteration checks.
- `timing.jl` is the only timing implementation used by all benchmarks.
- `audit_existing_results.jl` checks units, sizes, bandwidth lower bounds, and
  consistency among existing Markdown summaries.
- `soc_strategy_map.jl` executes the R1-2 count-dimension matrix.
- `soc_root_profile.jl` executes controlled SOC branch and iteration studies.
- `exp_root_profile.jl` executes the corresponding exponential-cone studies.
- `application_trace.jl` records production-solver projection traces.
- The three `summarize_*.jl` files are the only scripts allowed to create
  manuscript tables or figures.
- The `_profile.cu` files preserve production numerical behavior while writing
  per-cone diagnostic records.

# 3. Phase 0: audit and freeze the measurement pipeline

This phase precedes all new experiments.

## 3.1 Audit existing summaries

- [ ] Compare the numerical entries in:
  - `doc/pdcs_overleaf/rebuttal_plan/gpu2_results.md`;
  - `doc/pdcs_overleaf/rebuttal_plan/warp_divergence_results.md`;
  - `doc/pdcs_overleaf/rebuttal_plan/cone_dimension_results.md`;
  - `doc/pdcs_overleaf/rebuttal_plan/cone_count_results.md`;
  - `doc/pdcs_overleaf/rebuttal_plan/exp_projection_results.md`.
- [ ] Write
  `benchmark/results/rebuttal/2026-07-comprehensive/audit/summary_diff.csv`
  with columns:
  `experiment,case,strategy,metric,source_a,value_a,source_b,value_b,status`.
- [ ] Mark an entry `consistent` only if both values agree after applying the
  displayed rounding. Otherwise mark it `rerun_required`.
- [ ] Do not select one conflicting summary by preference; regenerate every
  `rerun_required` value from raw output.

## 3.2 Validate workload size and timing units

- [ ] Record the actual cone count, each cone dimension, total scalar
  dimension, input bytes, auxiliary bytes, and output bytes before every run.
- [ ] For each measured time, calculate the input-only bandwidth lower bound
  \[
  B_{\min}=\frac{8\,(\text{total scalar dimension})}{\text{seconds}}.
  \]
- [ ] Flag a result if `B_min` exceeds the recorded device's theoretical HBM
  bandwidth. Since a projection requires more than one memory pass, any such
  result is invalid without further investigation.
- [ ] Specifically rerun the fixed-total-dimension grid-wise cases. The
  previously reported `1.2e9` `Float64` scalars in `0.0003` seconds imply an
  input-only rate of 32 TB/s and fail this sanity check.

## 3.3 Establish the single timing implementation

`timing.jl` must implement the following protocol:

1. allocate all arrays;
2. generate and validate one immutable host-side case;
3. transfer the case to preallocated device buffers;
4. launch at least five unmeasured warm-ups;
5. restore the device input outside the timed region;
6. record a CUDA start event on the kernel stream;
7. launch the projection;
8. record and synchronize a CUDA stop event on the same stream;
9. check `CUDA.last_error()` and synchronize before accepting the observation;
10. repeat with a fresh paired seed.

For kernels whose single-launch time is below 1 ms, time a batch of repeated
launches long enough to exceed 100 ms and divide by the number of launches.
Restore the input outside each measured projection unless the experiment is an
explicit warm-start sequence.

## 3.4 Phase 0 acceptance gates

Phase 0 passes only when:

- every conflicting summary value is assigned a reproducible raw source or
  marked for replacement;
- a synthetic delayed kernel test demonstrates that the timing wrapper includes
  device execution rather than host launch time;
- every timing row contains a positive workload size and unit;
- no accepted result exceeds the input-only bandwidth sanity bound;
- two repeated summarization runs produce byte-identical CSV tables.

# 4. Phase 1: comprehensive R1-2 count-dimension strategy map

## 4.1 Mathematical workload

For each pair `(m,d)`, project a vector onto

\[
(\mathcal K_{\mathrm{soc}}^d)^m,
\]

where `d` is the full SOC dimension, including the leading coordinate. Generate
all coordinates independently as

\[
x_i\sim\mathcal N(0,2^2).
\]

For trial `r` at grid cell `q`, use seed

\[
2026+10000q+r,\qquad r=0,\ldots,9.
\]

Generate the vector once per trial and give an identical copy to every
strategy.

## 4.2 Primary grid

Use:

```text
cone_counts     = 3, 10, 100, 1000, 10000, 100000, 1000000
cone_dimensions = 10, 32, 100, 500, 2000, 10000, 50000
strategies       = gridWise, blockWise, warpWise, threadWise
trials           = 10
```

Before allocating a cell, compute its full memory requirement, including all
strategy-specific buffers. Run the cell only if it uses at most 70% of the
recorded free GPU memory. Mark larger cells `memory_excluded`; do not reduce
their mathematical size silently.

Apply a synchronized 15-second timeout to each projection. Mark a timeout
`timeout`; do not replace it with 15 seconds when determining the fastest
completed strategy.

## 4.3 Strategy-map outputs

For every valid cell and strategy, report:

- mean, median, and sample standard deviation;
- 95% bootstrap confidence interval for the median using 10,000 paired
  resamples;
- runtime ratio to the fastest strategy on the same paired trials;
- effective input-plus-output bandwidth;
- timeout and correctness status.

Generate:

1. a complete numeric table;
2. a heat map of the strategy with the smallest paired median;
3. a second heat map marking every strategy within 5% of the smallest median;
4. count-wise crossover plots at fixed dimensions;
5. dimension-wise crossover plots at fixed counts.

## 4.4 Heuristic training and validation

Use the primary grid to define a deterministic selection table indexed by
logarithmic count and dimension bins. Validate it on the off-grid matrix:

```text
validation_counts     = 30, 300, 3000, 30000, 300000
validation_dimensions = 20, 64, 256, 1000, 5000, 25000
```

For each validation cell, compare:

- the strategy chosen by the heuristic;
- the empirically fastest valid strategy;
- the slowdown of the chosen strategy relative to the empirical oracle.

The heuristic passes if:

- at least 90% of validation cells are within 10% of the oracle;
- every validation cell is within 25% of the oracle, excluding timed-out oracle
  strategies;
- every heuristic decision is reproducible from cone type, count, dimension,
  and device architecture without inspecting input values.

## 4.5 R1-2 claim gate

The revised manuscript may describe a crossover only when:

- the nominal winner is at least 5% faster than the alternative, or the paired
  confidence interval excludes a runtime ratio of 1;
- all compared strategies passed correctness checks;
- the same qualitative regime is observed on at least two adjacent grid cells.

Otherwise describe the strategies as practically tied.

# 5. Phase 2: root-search diagnostic instrumentation

## 5.1 Diagnostic record

Define one fixed-width record per cone with:

```text
branch_code
interval_expansion_iterations
newton_attempts
newton_accepts
bisection_iterations
oracle_evaluations
warm_start_attempted
warm_start_accepted
max_iter_reached
final_residual
```

Use branch codes:

```text
0 = feasible early return
1 = polar/zero early return
2 = positive-t root search
3 = negative-t root search
4 = near-zero path
5 = numerical failure
```

Thread-wise kernels write one record per thread/cone. Warp-wise kernels write
one record per cone from lane 0 after all cooperative operations complete.
Block-wise kernels write one record per cone from thread 0.

## 5.2 Build isolation

- [ ] Add `PDCS_PROFILE_ROOT_SEARCH` guards around counter updates.
- [ ] Build diagnostic outputs as
  `massive_block_proj_profile.ptx`,
  `sufficient_block_proj_profile.ptx`, and
  `moderate_block_proj_profile.ptx`.
- [ ] Keep the production PTX targets and exported function signatures
  unchanged.
- [ ] Confirm that the production binaries contain no diagnostic buffer or
  counter operations.

## 5.3 Instrumentation correctness

For 1,024 cones from every controlled case:

- compare production GPU output with diagnostic GPU output;
- compare both with the CPU reference;
- require maximum infinity-norm differences no greater than `5e-8`;
- require identical top-level branch classifications between CPU and GPU;
- require zero `max_iter_reached` values;
- verify that disabling the macro restores the production PTX path.

The instrumentation phase passes only if counters change as expected on
hand-constructed cases requiring zero, one, and multiple bisection iterations.

# 6. Phase 3: controlled SOC divergence experiments

## 6.1 Common SOC configuration

Primary saturated configuration:

```text
cone_count     = 1048576
cone_dimension = 10
projection     = diagonally rescaled SOC, type 22
strategies     = threadWise, warpWise, blockWise
timing_seeds   = 2026:2035
profile_seeds  = 2026, 2027, 2028
abs_tol        = 1e-12
rel_tol        = 1e-12
```

Set the first diagonal coordinate to one. Generate positive remaining diagonal
coordinates in log space, normalize each cone's diagonal to geometric mean
one, and clamp entries to `[1e-3,1e3]`.

## 6.2 Experiment SOC-A: parametric-similarity curve

Construct a base root-search case with

\[
t=0.2\lVert D u\rVert_2,
\]

so the point is outside the feasible cone but remains on the positive-\(t\)
root-search path.

Construct a sequence of nonidentical instances using perturbation magnitudes:

```text
delta = 0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1
```

For each step, independently perturb the normalized tail direction and the
log-diagonal entries. Renormalize the diagonal but do not multiply the input
and diagonal by one common scalar.

For each nonzero `delta`, run:

- cold start with the root state reset to zero;
- warm start using the preceding sequence member's root;
- all three strategies on identical sequences.

Report:

- runtime and warm/cold ratio;
- branch-class fractions;
- root-iteration median, 90th percentile, 99th percentile, and maximum;
- per-thread-wise-warp iteration spread;
- model-based active-lane efficiency;
- branch-target uniformity;
- SM and memory Speed-of-Light throughput;
- achieved occupancy and eligible warps per cycle;
- sustained GPU-busy distribution.

## 6.3 Experiment SOC-B: root-only difficulty curve

Generate a candidate pool of `4 * 1048576` positive-\(t\) root-search cones by
crossing:

```text
boundary_ratio = 0.05, 0.20, 0.50, 0.80
log_scale_std  = 0.00, 0.25, 0.50, 1.00, 2.00
```

Use `t = boundary_ratio * norm(D*u)`. Run the diagnostic kernel once, discard
early-return and failed cones, sort the remaining cones by total root
iterations, and select equal-sized samples from the four iteration-count
quartiles.

Create three permutations of the identical selected multiset:

1. `iteration_grouped`: every thread-wise warp contains one quartile;
2. `iteration_random`: Fisher-Yates permutation with seed 2026;
3. `iteration_interleaved`: each group of 32 consecutive cones contains eight
   cones from each quartile.

Run uninstrumented timings on all three permutations. This is the primary test
of loop-exit divergence because every cone follows the same positive-\(t\)
top-level branch.

## 6.4 Experiment SOC-C: mixed top-level branches

Create equal numbers of four classes:

- feasible: `t = 1.2 * norm(D*u)`;
- polar/zero: `t = -1.2 * norm(u ./ D)`;
- positive root: `t = 0.2 * norm(D*u)`;
- negative root: `t = -0.2 * norm(u ./ D)`.

Validate each class using diagnostic branch codes before timing. Construct:

1. `branch_grouped`;
2. `branch_random` using seed 2026;
3. `branch_interleaved`, with eight lanes from each class in every thread-wise
   warp.

Verify that the input, diagonal, warm-start, and metadata arrays are permuted
by the same bijection and that inverse permutation recovers the original
multiset exactly.

Use the thread-wise grouped/random/interleaved comparison as the direct
intra-warp divergence result. Report warp-wise ordering effects separately as
cross-warp scheduling or workload-balance effects unless source-level evidence
establishes another mechanism.

## 6.5 Experiment SOC-D: dimension sensitivity

Use:

```text
cone_count      = 262144
cone_dimensions = 10, 32, 128, 256
cases           = parametric delta=1e-2,
                  iteration_grouped,
                  iteration_interleaved,
                  branch_grouped,
                  branch_interleaved
```

Run all three strategies and all ten timing seeds. Report runtime both
absolutely and per scalar coordinate.

## 6.6 SOC divergence acceptance gates

The statement “no material divergence in the similar-scaling regime” is
supported only if, for `delta <= 1e-2`:

- thread-wise paired median runtime is no more than 5% slower than the uniform
  baseline;
- median modeled active-lane efficiency is at least 90%;
- effective hardware throughput
  `max(SM SOL, memory SOL)` is at least 80%, or a documented latency/occupancy
  analysis explains why the reviewer’s threshold is not the relevant limit;
- no correctness or convergence failure occurs;
- the conclusion holds for all three profile seeds.

Evidence of harmful divergence requires:

1. a larger per-warp branch or iteration spread;
2. a lower active-lane or issue-efficiency measure; and
3. a higher paired uninstrumented runtime for the identical cone multiset.

# 7. Phase 4: diagonally rescaled exponential-cone experiments

## 7.1 Common configuration

```text
cone_count     = 1048576
cone_dimension = 3
projection     = primal diagonal exponential cone, type 27
strategies     = threadWise, warpWise, blockWise
timing_seeds   = 2026:2035
profile_seeds  = 2026, 2027, 2028
```

Use independent streams for input vectors and diagonal entries. Never change
`sigma_x` and `sigma_D` together in a sensitivity experiment.

## 7.2 Experiment EXP-A: independent heterogeneity sweeps

Input sweep:

```text
sigma_x = 0.1, 0.5, 1, 2, 5, 10
sigma_D = 1
```

Diagonal sweep:

```text
sigma_x = 1
sigma_D = 0.1, 0.5, 1, 2, 5, 10
```

Generate `D_i = clamp(abs(sigma_D * Z_i),1e-3,1e3)` from an independent
Gaussian stream. Report branch and iteration distributions so a slowdown is
not attributed to divergence merely because two different workloads have
different total root-search work.

## 7.3 Experiment EXP-B: branch and iteration ordering

Repeat the grouped/random/interleaved design from SOC-B and SOC-C using
exponential-cone branch classes and measured iteration quartiles. Use the same
multiset for every ordering and treat the thread-wise comparison as primary.

## 7.4 Experiment EXP-C: nearby parametric sequence

Run the same `delta` sequence as SOC-A, with independent perturbations to the
input and log-diagonal entries. Compare cold and warm starts on nonidentical
successive inputs.

## 7.5 Exponential-cone acceptance gates

- Report exponential-cone results separately from SOC results.
- Do not generalize the SOC scale-invariance result to exponential cones.
- If the heterogeneous exponential workload remains substantially slower,
  separate the fraction explained by greater total iteration count from the
  fraction explained by within-warp iteration spread.
- Require the same correctness, three-seed profile, and identical-multiset
  ordering checks used for SOCs.

# 8. Phase 5: application-level validation

Synthetic cases isolate mechanisms; production traces establish relevance.

## 8.1 Selected application families and sizes

Use three sizes from each manuscript family:

### Fisher market equilibrium

```text
(m,n) = (100,5000), (100000,1000), (250000,1000)
```

These instances exercise many three-dimensional exponential cones.

### Lasso

```text
(m,n) = (10000,100000), (70000,700000), (400000,7000000)
```

These instances provide the contrasting regime of one high-dimensional SOC.

### Multi-period portfolio optimization

```text
T = 48, 360, 2160
assets = 844
```

These instances exercise many SOCs and mixed cone structures.

Use the generation rules and seeds already stated in the manuscript. Preserve
the existing solver tolerances and record both `1e-3` and `1e-6` termination
targets.

## 8.2 Parametric sequences

For each base instance, generate ten nearby instances:

- Fisher: perturb nonzero utility values multiplicatively by independent 1%
  log-normal noise;
- Lasso: vary `lambda` through
  `0.95,0.96,...,1.04` times the base value while keeping `A` and `b` fixed;
- MPO: use the manuscript's period-to-period prediction perturbations and
  preserve chronological order.

Run cold and warm projection-root states for the same sequence.

## 8.3 Production trace fields

For every solver iteration, record:

```text
application
instance_size
sequence_index
solver_iteration
cone_type
cone_count
cone_dimension_min
cone_dimension_median
cone_dimension_max
selected_strategy
projection_time_ms
solver_iteration_time_ms
branch_class_counts
root_iteration_p50
root_iteration_p90
root_iteration_max
mean_per_warp_iteration_spread
mean_active_lane_efficiency
```

Counter collection may run on a separate diagnostic solve. Production-solve
timings must use uninstrumented kernels.

## 8.4 Application validation outputs

Generate:

- projection time as a fraction of total solver time;
- histograms of real branch classes and root iterations;
- real per-warp iteration-spread distributions;
- comparison of real traces with the controlled uniform, similar, and
  heterogeneous regimes;
- warm/cold solver and projection ratios;
- the fraction of production iterations for which the strategy heuristic is
  within 10% of the microbenchmark oracle.

The intended-use claim passes only if the real parametric traces fall within
the iteration-spread and utilization range described as “similar” in the
controlled experiment.

# 9. Phase 6: cross-architecture validation

## 9.1 Hardware

Run the full primary experiment on:

1. NVIDIA H100 80 GB, `sm_90`;
2. NVIDIA A100 80 GB, `sm_80`.

Record:

- GPU model and UUID;
- compute capability;
- driver, toolkit, and CUDA runtime versions;
- Julia and CUDA.jl versions;
- memory capacity and free memory;
- persistence mode, power limit, compute mode, and MIG state;
- application clocks when readable;
- all other GPU processes at run start and finish.

Compile native PTX or binaries for each architecture. Do not run an `sm_90`
binary as the A100 measurement.

## 9.2 Cross-hardware experiment subset

Run on both GPUs:

- the full R1-2 strategy-map primary grid;
- SOC-A at `delta = 0,1e-2,1e-1`;
- SOC-B grouped/random/interleaved;
- SOC-C grouped/random/interleaved;
- SOC-D at all four dimensions;
- EXP-A at `sigma_x = 0.1,1,10` with `sigma_D = 1`, and at
  `sigma_D = 0.1,1,10` with `sigma_x = 1`;
- EXP-B grouped/random/interleaved;
- the middle-sized Fisher, Lasso, and MPO application traces.

## 9.3 Portability interpretation

Report:

- absolute time by GPU;
- time normalized to the fastest strategy on the same GPU;
- whether strategy-region boundaries move;
- whether the signs of grouped/interleaved and warm/cold effects agree;
- whether the reviewer-facing conclusion changes.

Do not attribute H100/A100 differences to one architectural feature without
supporting counters or source-level analysis.

# 10. Profiling protocol

## 10.1 Nsight Compute

Profile one warmed kernel per process with:

```text
SpeedOfLight
Occupancy
SchedulerStats
WarpStateStats
SourceCounters
```

Collect at least:

- branch-target uniformity;
- achieved occupancy;
- SM and memory Speed-of-Light throughput;
- eligible and issued warps per cycle;
- branch-resolving, barrier, memory-dependency, and not-selected stall shares;
- average active threads per executed instruction when supported.

Use seeds 2026, 2027, and 2028. Summarize their mean and range; do not treat
profiler replay passes as independent observations.

## 10.2 Nsight Systems and sustained utilization

For every primary sustained case:

- loop the same validated workload for at least 30 seconds;
- collect Nsight Systems CUDA/NVTX traces;
- collect GPU metrics at the highest permitted sampling rate;
- collect `nvidia-smi` utilization as a coarse secondary view;
- report mean, median, 10th percentile, 90th percentile, and peak GPU-busy
  percentage;
- report completed launches and work units per second.

GPU-busy percentage is not called “efficiency” by itself. Interpret it together
with throughput, eligible warps, runtime, and iteration spread.

## 10.3 Profiling failure policy

If hardware counters return `ERR_NVGPUCTRPERM`, keep the timing run, record the
permission failure verbatim, and rerun the hardware-profile subset on an
authorized machine. Do not substitute `nvidia-smi` for the missing Nsight
metrics in a claim about branch or issue efficiency.

# 11. Statistical analysis

Use paired observations wherever strategies or orderings share a seed and cone
multiset.

For each timing comparison, report:

- mean, median, sample standard deviation;
- paired median ratio;
- 95% bootstrap confidence interval using 10,000 paired resamples;
- all ten raw observations.

Use practical-effect thresholds:

- less than 5%: no material difference;
- 5% to 15%: small but measurable;
- 15% to 30%: moderate;
- above 30%: large.

These labels describe magnitude, not statistical significance.

For divergence analysis, report correlations between:

- per-warp iteration spread and runtime;
- modeled active-lane efficiency and runtime;
- branch uniformity and runtime;
- eligible warps per cycle and runtime.

Do not infer causality from correlation alone. The identical-multiset ordering
experiments provide the causal control.

# 12. Correctness protocol

For every case before timing:

1. compare at least 1,024 sampled cones with the CPU projection;
2. require maximum infinity-norm error no greater than `5e-8`;
3. require finite output;
4. check cone feasibility to the benchmark tolerance;
5. check projection KKT/Moreau residuals;
6. require zero maximum-iteration failures;
7. verify that grouped, random, and interleaved layouts are exact permutations;
8. verify that inverse permutation produces identical projected values;
9. verify that diagnostic instrumentation does not change outputs.

Write one row per run to `correctness.csv`; never summarize failed timing rows
as performance results.

# 13. Output contract

Each hardware directory must contain:

```text
environment.txt
manifest.csv
correctness.csv
timings_raw.csv
timings_summary.csv
root_iterations_raw.csv.zst
root_iterations_summary.csv
strategy_map.csv
application_trace.csv.zst
ncu/
nsys/
nvidia_smi/
figures/
logs/
```

The manuscript directory must contain:

```text
r1_2_dimension_table.tex
r1_2_strategy_map.pdf
r1_3_similar_profile_table.tex
r1_3_iteration_ordering_table.tex
r1_3_branch_ordering_table.tex
r1_3_dimension_sensitivity.pdf
r1_3_exponential_table.tex
r1_3_application_trace_table.tex
reviewer_response_r1_2.tex
reviewer_response_r1_3.tex
manuscript_section4_revision.tex
```

Every generated table must include a comment containing:

- summarization script path;
- raw input manifest path;
- git commit hash;
- hardware identifier.

# 14. Execution tasks and review gates

## Task 1: Measurement audit

**Files:**

- Create: `benchmark/rebuttal/audit_existing_results.jl`
- Create: `benchmark/rebuttal/timing.jl`
- Create: `test/test_rebuttal_timing.jl`

- [ ] Implement workload-byte and bandwidth sanity checks.
- [ ] Implement synchronized CUDA-event timing.
- [ ] Add a delayed-kernel timing test that fails without synchronization.
- [ ] Regenerate the fixed-total-dimension grid-wise cases.
- [ ] Review Phase 0 acceptance gates before proceeding.

## Task 2: Shared case generation and validation

**Files:**

- Create: `benchmark/rebuttal/common.jl`
- Create: `benchmark/rebuttal/generate_soc_cases.jl`
- Create: `benchmark/rebuttal/generate_exp_cases.jl`
- Create: `benchmark/rebuttal/validate_projection.jl`
- Create: `test/test_rebuttal_case_generation.jl`
- Create: `test/test_rebuttal_permutations.jl`
- Create: `test/test_rebuttal_correctness.jl`

- [ ] Implement deterministic generators and manifests.
- [ ] Implement exact paired-case copying across strategies.
- [ ] Implement grouped/random/interleaved permutations and inverse checks.
- [ ] Implement CPU, feasibility, KKT, and finite-value validation.
- [ ] Review generator metadata and correctness outputs.

## Task 3: R1-2 strategy map

**Files:**

- Create: `benchmark/rebuttal/soc_strategy_map.jl`
- Create: `benchmark/rebuttal/summarize_strategy_map.jl`

- [ ] Run the H100 primary grid.
- [ ] Fit the deterministic lookup heuristic.
- [ ] Run the H100 off-grid validation matrix.
- [ ] Run the A100 primary and validation grids.
- [ ] Generate the numeric tables and heat maps.
- [ ] Review all crossover claims against the 5% and confidence-interval gate.

## Task 4: Diagnostic CUDA kernels

**Files:**

- Create: `src/pdcs_gpu/cuda/root_profile.h`
- Create: `src/pdcs_gpu/cuda/massive_block_proj_profile.cu`
- Create: `src/pdcs_gpu/cuda/moderate_block_proj_profile.cu`
- Create: `src/pdcs_gpu/cuda/sufficient_block_proj_profile.cu`
- Modify: `src/pdcs_gpu/cuda/Makefile`

- [ ] Implement the fixed diagnostic record.
- [ ] Instrument SOC and exponential-cone root paths.
- [ ] Build separate `sm_90` and `sm_80` profile targets.
- [ ] Compare diagnostic, production, and CPU outputs.
- [ ] Review hand-constructed iteration-counter tests.

## Task 5: Controlled SOC study

**Files:**

- Create: `benchmark/rebuttal/soc_root_profile.jl`
- Create: `benchmark/rebuttal/summarize_root_profiles.jl`

- [ ] Run SOC-A similarity and warm-start curves.
- [ ] Run SOC-B root-only iteration ordering.
- [ ] Run SOC-C mixed-branch ordering.
- [ ] Run SOC-D dimension sensitivity.
- [ ] Run three-seed Nsight Compute profiles.
- [ ] Run 30-second sustained-utilization profiles.
- [ ] Review every SOC acceptance gate.

## Task 6: Controlled exponential-cone study

**Files:**

- Create: `benchmark/rebuttal/exp_root_profile.jl`

- [ ] Run independent input and diagonal heterogeneity sweeps.
- [ ] Run branch and iteration ordering studies.
- [ ] Run nearby cold/warm parametric sequences.
- [ ] Run three-seed hardware profiles and sustained-utilization runs.
- [ ] Separate additional work from within-warp imbalance in the analysis.

## Task 7: Application traces

**Files:**

- Create: `benchmark/rebuttal/application_trace.jl`
- Create: `benchmark/rebuttal/summarize_application_traces.jl`

- [ ] Trace all nine specified application sizes.
- [ ] Run the ten-member parametric sequence for each of the nine specified
  application sizes.
- [ ] Compare production traces with controlled regimes.
- [ ] Quantify projection share of solver time.
- [ ] Review whether the intended-use claim is supported.

## Task 8: Cross-hardware replication

**Files:**

- Create: `benchmark/rebuttal/run_h100.sh`
- Create: `benchmark/rebuttal/run_a100.sh`
- Create: `benchmark/rebuttal/profile_ncu.sh`
- Create: `benchmark/rebuttal/profile_nsys.sh`

- [ ] Record both hardware environments.
- [ ] Run the defined H100/A100 common subset.
- [ ] Produce normalized cross-hardware comparisons.
- [ ] Identify which heuristic boundaries are architecture-specific.

## Task 9: Reproduction and manuscript artifacts

**Files:**

- Create: `benchmark/rebuttal/reproduce_all.sh`
- Create: `benchmark/rebuttal/README.md`
- Modify after all gates pass:
  `doc/pdcs_overleaf/response_to_reviewers_PDCS.tex`
- Modify after all gates pass:
  `doc/pdcs_overleaf/PDCS_arxiv.tex`

- [ ] Reproduce every summary from an empty results directory.
- [ ] Confirm byte-identical manuscript CSV and LaTeX tables.
- [ ] Generate the R1-2 and R1-3 response text from accepted results.
- [ ] Add only qualified claims permitted by Sections 1, 4.5, 6.6, and 7.5.
- [ ] Compile both PDFs and visually inspect the new tables and figures.

# 15. Final completion criteria

The comprehensive experiment package is complete only when:

1. all timing sanity checks pass;
2. all conflicting legacy values are replaced or reconciled;
3. every manuscript number traces to raw data and a deterministic script;
4. the R1-2 count-dimension map and off-grid heuristic validation pass;
5. per-cone root iterations and per-warp spreads are measured directly;
6. root-only and mixed-branch ordering experiments use identical multisets;
7. warm starts use nearby but nonidentical instances;
8. SOC and exponential-cone conclusions are reported separately;
9. real application traces are consistent with the claimed intended-use
   regime;
10. the principal qualitative conclusions are checked on H100 and A100;
11. all correctness and convergence checks pass;
12. the revised reviewer response acknowledges both benign similar-scaling
    behavior and measurable heterogeneous stress costs.

# 16. Execution order

Use the following dependency order:

```text
Phase 0 audit
  -> shared generators and validators
  -> R1-2 strategy map
  -> diagnostic kernels
  -> controlled SOC study
  -> controlled exponential-cone study
  -> application traces
  -> A100 replication
  -> manuscript tables and reviewer response
```

The submission-critical checkpoint occurs after the controlled SOC and
exponential-cone studies pass on the H100. Application traces and A100
replication complete the approved comprehensive scope and must finish before
making cross-architecture or real-workload generalizations.
