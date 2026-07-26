# Additional R1-3 Experiments: Exhaustive Completion Plan

> **For agentic workers:** Execute this plan task by task and retain every raw
> artifact required by the acceptance gates. Use the checkboxes for progress
> tracking. Do not reuse superseded timing or profiler summaries.

**Goal:** Complete the evidence needed to answer Reviewer R1-3 rigorously,
explain the unresolved behavior in the current SOC divergence report, and test
whether the conclusions extend to parametric-similar SOCs, other SOC
dimensions, diagonally rescaled exponential cones, and application-level
projection workloads.

**Primary decision:** Every experiment with statistical inference uses exactly
10 independent workload seeds:

```text
2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035
```

Ten repeated kernel launches are technical replicates, not additional
statistical observations. The workload seed is always the inferential unit.
No experiment in this plan requires seeds 2036--2045.

**Architecture:** Preserve the completed paired SOC experiment as the baseline.
Add targeted experiments in five layers: measurement repair, realistic
parametric validation, causal-control and profiler diagnosis, cone-family and
dimension generalization, and application-level validation. All timing uses
uninstrumented production kernels; diagnostic counters, Nsight Compute replay,
and sustained-utilization collection remain separate.

**Tech stack:** Julia 1.10.4, CUDA.jl, CUDA C++, NVIDIA H100 80 GB HBM3
(`sm_90`), CUDA Toolkit 12.4 for the current run, NVIDIA driver 555.42.02,
Nsight Compute 2024.2, Nsight Systems or DCGM where available,
`nvidia-smi dmon`, CSV, SHA-256 manifests, and `Float64` arithmetic.

---

## 1. Why Additional Experiments Are Still Needed

The completed run
`20260726T181035Z_soc_divergence_full` already establishes several important
facts:

- The same-branch negative-root manipulation increases modeled within-warp
  work spread from 4 to 46 oracle evaluations and reduces modeled active-lane
  efficiency from 0.947 to 0.776.
- Under that manipulation, thread-wise interleaving has a net timing ratio
  \(R_T=1.023\), with 95% CI \([1.020,1.026]\), and thread-wise remains
  approximately \(3.4\times\) faster than warp-wise.
- Top-level branch interleaving increases thread-wise time by \(15.4\%\), even
  though thread-wise remains approximately \(4.5\times\) faster than
  warp-wise.
- The current positive-root generator does not create widely different
  bisection counts: the observed counts remain approximately 38--39.

The remaining evidence gaps are:

1. The sustained-utilization summary is internally inconsistent: mean SM busy
   is 60--67%, median SM busy is 97--99%, and P10 is 0%, although
   restoration/reset accounts for only 0.6--1.6% of elapsed time.
2. The current report does not include the genuinely parametric-similar
   validation requested by the reviewer and specified in the original plan.
3. The branch experiment has an unexplained warp-wise ordering ratio
   \(R_W=1.765\), so the interaction statistic
   \(\Theta=R_T/R_W\) is not yet a clean causal divergence measure.
4. The current Nsight interpretation does not include the component-level
   memory and stall metrics required to identify the actual bottleneck.
5. The current active-lane efficiency is derived from per-cone work counts; it
   is not a direct measurement of executed active-lane masks.
6. The current divergence conclusion is limited to one SOC dimension and one
   GPU strategy regime.
7. No corrected, comparable divergence experiment has yet been completed for
   diagonally rescaled exponential cones.
8. The current evidence is based on isolated projection loops rather than
   projection ranges observed inside representative solver runs.

---

## 2. Global Experimental Constraints

These constraints apply to every task below.

### 2.1 Hardware and software identity

- Use one explicitly identified NVIDIA H100 GPU per run.
- Record its model, UUID, PCI address, physical index, CUDA-visible index,
  clocks, power limit, temperature, compute mode, and competing processes.
- Record Julia, CUDA.jl, CUDA Toolkit, NVIDIA driver, Nsight Compute, and
  Nsight Systems versions separately.
- Distinguish the installed CUDA Toolkit version from the maximum CUDA version
  reported by the driver.

### 2.2 Immutable source snapshot

Before running any additional experiment:

- save `git rev-parse HEAD`;
- save `git status --short`;
- save `git diff --binary`;
- save all untracked experiment source files;
- record SHA-256 hashes for Julia sources, CUDA sources, generated PTX/shared
  libraries, analysis scripts, and wrapper scripts;
- store compiler commands and optimization flags;
- record whether diagnostic counter macros are enabled.

The run directory must contain enough information to reconstruct a dirty
worktree exactly. A commit hash alone is insufficient.

### 2.3 Statistical unit and replication

- Inferential seeds: exactly 2026--2035.
- Warm-up launches per timing cell: 5.
- Measured launches per timing cell: 10.
- Seed-level statistic: median of the 10 synchronized launch times.
- Statistical analysis: paired log ratios across the 10 workload seeds.
- Report both a paired-\(t\) 95% interval and a 10,000-resample seed-block
  bootstrap interval.
- Never treat cones, warps, kernel launches, profiler passes, or utilization
  samples as independent inferential observations.

### 2.4 Timing

- Generate inputs, select cases, sort, permute, allocate, restore state, and
  reset stored roots outside the timed interval.
- Use CUDA events around exactly one production projection kernel launch.
- Synchronize the stop event before reading elapsed time.
- Use counterbalanced cell order.
- Never use Nsight Compute replay duration as publication runtime.
- Keep diagnostic instrumentation out of every timing binary.

### 2.5 Correctness

Every timed cell must pass:

- finite-output checks;
- sampled CPU/GPU agreement on at least 1,024 cones per seed;
- full-vector thread-wise/warp-wise agreement after inverse permutation;
- grouped/interleaved agreement after inverse permutation;
- primal/polar feasibility;
- complementarity/Moreau and KKT residual checks;
- the common production root-termination condition;
- zero `MAX_ITER` cap hits.

For every reported error, store both the raw error and the dimensionless
mixed-tolerance-scaled error. Define the scaling formula in the CSV metadata.
Do not use the ambiguous label `strategy agreement error` for two different
quantities.

### 2.6 Interpretation

- Treat \(R_T=T_{T,I}/T_{T,G}\) as the primary observed thread-wise layout
  effect.
- Treat \(A=T_{T,I}/T_{W,I}\) as a practical strategy comparison, not as the
  causal cost of divergence.
- Treat \(R_W=T_{W,I}/T_{W,G}\) as an ordering/scheduling diagnostic.
- Use \(\Theta=R_T/R_W\) causally only if the warp-wise ordering control passes
  the invariance gate defined in Experiment C.
- Use a 5% materiality threshold for timing ratios, but report the complete
  confidence interval and the observed percentage effect.
- Do not write “no divergence.” State the measured cost and restrict the claim
  to the tested GPU, cone family, count, dimension, input generator, and
  stopping tolerance.

---

## 3. Experiment Inventory and Priority

| ID | Experiment | Purpose | Priority |
|:---|:---|:---|:---|
| A | Baseline audit and correctness reconciliation | Make the completed run publication-traceable | Required |
| B | Active-window sustained GPU utilization | Repair the mean/median/P10 inconsistency | Required |
| C | Genuinely parametric-similar SOC validation | Directly address the reviewer’s similar-scaling scenario | Required |
| D | Branch profiler and warp-wise ordering controls | Explain \(R_W=1.765\) and assess whether \(\Theta\) is interpretable | Required for causal interaction claims |
| E | Direct active-lane diagnostic | Validate modeled lane efficiency with execution masks | Strong supporting evidence |
| F | SOC dimension sensitivity | Test whether the divergence effect changes with cone dimension | Generalization |
| G | Diagonally rescaled exponential-cone divergence | Extend the study beyond SOCs | Exhaustive cone-family coverage |
| H | Application-level projection profiling | Establish relevance inside solver workloads | External validity |
| I | Integrated analysis and reviewer tables | Produce the only publication-ready summaries | Required |

Run the experiments in the order A, B, C, D, E, F, G, H, I.

### 3.1 Implementation file map

Keep generation, timing, profiling, and analysis responsibilities separate.
Use the following file boundaries:

| File | Action | Responsibility |
|:---|:---|:---|
| `benchmark/rescaled_soc_divergence_2x2.jl` | Modify | Preserve the paired SOC baseline; add ready/start/done duration-mode synchronization and balanced-control entry points |
| `benchmark/rebuttal/soc_divergence_cases.jl` | Modify | Add parametric-similar cases, dimension-aware cases, balanced permutations, and immutable manifests |
| `benchmark/analyze_soc_divergence_2x2.jl` | Modify | Reconcile correctness fields and compute all ten-seed SOC ratios and confidence intervals |
| `benchmark/rescaled_soc_parametric_similar.jl` | Create | Run cold and warm parametric-similar SOC panels without complicating the baseline 2x2 harness |
| `benchmark/rescaled_soc_dimension_divergence.jl` | Create | Run the fixed-count dimension-generalization experiment |
| `benchmark/rebuttal/run_active_window_utilization.sh` | Create | Coordinate the long-lived Julia process, monitor start/stop, timestamps, and raw utilization artifacts |
| `benchmark/rebuttal/profile_soc_divergence.sh` | Create | Collect reproducible Nsight Compute/System profiles for root, branch, parametric, and balanced-control cells |
| `benchmark/rescaled_exp_divergence_2x2.jl` | Create | Generate, validate, time, and profile exponential-cone divergence cases |
| `benchmark/analyze_exp_divergence_2x2.jl` | Create | Compute exponential-cone seed summaries, ratios, confidence intervals, and publication panels |
| `benchmark/profile_application_projections.jl` | Create | Run application-level performance and diagnostic modes with identical solver inputs |
| `src/pdcs_gpu/cuda/massive_block_proj.cu` | Modify under diagnostic guards only | Add sampled active-mask records for thread-wise SOC root loops |
| `src/pdcs_gpu/cuda/sufficient_block_proj.cu` | Modify under diagnostic guards only | Retain comparable root-work diagnostics for warp-wise SOC |
| Exponential-cone CUDA projection sources used by type 27 | Modify under diagnostic guards only | Add branch, iteration, residual, and sampled active-mask counters |
| `test/rebuttal/soc_divergence_cases_test.jl` | Create | Test generators, permutations, hashes, seed changes, and manipulation gates |
| `test/rebuttal/divergence_correctness_test.jl` | Create | Test CPU/GPU, cross-strategy, cross-layout, termination, and diagnostic invariance checks |
| `test/rebuttal/utilization_alignment_test.jl` | Create | Test ready/start/done parsing, exact window trimming, sample counts, and time-ledger reconciliation |
| `test/rebuttal/exp_divergence_cases_test.jl` | Create | Test exponential branch classes, iteration quartiles, permutations, and correctness gates |

Do not change production projection behavior outside compile-time diagnostic
guards. The timing build must be byte-identifiable independently of every
diagnostic build.

---

## 4. Experiment A: Baseline Audit and Correctness Reconciliation

### 4.1 Objective

Freeze the completed SOC run, eliminate ambiguous error labels, verify all
reported ratios from immutable seed-level data, and prevent accidental mixing
with superseded experiments.

### 4.2 Files

- Read:
  `benchmark/results/rebuttal/soc_divergence_2x2/20260726T181035Z_soc_divergence_full/`
- Verify:
  `benchmark/rescaled_soc_divergence_2x2.jl`
- Verify:
  `benchmark/rebuttal/soc_divergence_cases.jl`
- Verify:
  `benchmark/analyze_soc_divergence_2x2.jl`
- Verify:
  `src/pdcs_gpu/cuda/massive_block_proj.cu`
- Verify:
  `src/pdcs_gpu/cuda/sufficient_block_proj.cu`
- Create inside the frozen run:
  `source_snapshot/`
- Regenerate:
  `correctness_reconciled.csv`
- Regenerate:
  `effect_estimates_reconciled.csv`
- Regenerate:
  `publication_tables_reconciled.md`

### 4.3 Required actions

- [ ] Save the complete source snapshot and hashes specified in Section 2.2.
- [ ] Define the raw CPU/GPU error, scaled mixed error, raw cross-strategy
  error, raw cross-layout error, and each residual in machine-readable
  metadata.
- [ ] Recompute every correctness summary directly from raw output.
- [ ] Resolve the contradiction between the nonzero table entries and the
  statement that cross-strategy agreement is exactly zero.
- [ ] Recompute \(A\), \(A_G\), \(R_T\), \(R_W\), and \(\Theta\) from the ten
  seed-level medians.
- [ ] Report ratios to at least four decimal places when a confidence bound is
  close to the 1.05 materiality threshold.
- [ ] Mark every older SOC grouped/interleaved result as superseded.
- [ ] Confirm that all inferential summaries contain exactly seeds 2026--2035.

### 4.4 Acceptance gates

- Every timing seed has one passing correctness record.
- Every reconciled table value is reproducible from a saved CSV.
- The source snapshot reconstructs the exact timing and diagnostic binaries.
- No ambiguous correctness label remains.
- No superseded timing enters any new analysis.

---

## 5. Experiment B: Active-Window Sustained GPU Utilization

### 5.1 Question

During a long, continuously repeated projection workload, what is the
time-aligned distribution of SM activity, and does it support the reviewer’s
suggested 80--90% utilization range?

### 5.2 Cells and replication

Collect utilization for both divergence sources:

```text
experiment = iteration, branch
layout     = grouped, interleaved
strategy   = threadWise, warpWise
seeds      = 2026:2035
duration   = 35 seconds per cell
```

This produces 80 sustained-utilization runs. The first five seconds are a
stabilization interval and are excluded before computing the 30-second
publication window.

### 5.3 Measurement design

The Julia process must be long-lived:

1. initialize CUDA;
2. load or generate the immutable case;
3. compile and warm the target kernel;
4. print a machine-readable `READY` record with a monotonic timestamp;
5. wait for a wrapper signal;
6. execute the 35-second restore/reset/projection loop;
7. print `FIRST_PROJECTION_START`, `LAST_PROJECTION_STOP`, and `DONE`
   timestamps;
8. save a time ledger for projection, restore, reset, host gaps, and total
   wall time.

The wrapper must start `nvidia-smi dmon` only after `READY`, send the start
signal, and stop collection immediately after `DONE`. Use timestamped output
when supported. If `dmon` cannot provide timestamps precise enough to align
the window, collect an additional Nsight Systems or DCGM activity trace.

### 5.4 Metrics

For each seed and cell, report:

- aligned sample count;
- mean, median, P10, P90, and peak SM busy;
- zero-SM sample fraction;
- mean and peak memory-controller activity;
- mean power, clocks, and temperature;
- projection launches per second;
- projection-time fraction;
- restore/reset-time fraction;
- unexplained host/device gap fraction.

Report two clearly labeled windows:

1. **end-to-end repeated-loop window**, including required restore/reset;
2. **kernel-active characterization**, obtained from NVTX-delimited profiler
   data rather than by discarding low `dmon` samples.

### 5.5 Required actions

- [ ] Add a `READY/START/DONE` synchronization interface to the duration-mode
  harness.
- [ ] Record monotonic host timestamps and CUDA-event time totals.
- [ ] Collect all 80 seed/cell runs.
- [ ] Trim only by explicit timestamps; never remove samples merely because
  utilization is zero.
- [ ] Reconcile sample time with the CUDA-event and wall-clock ledger.
- [ ] Compare the new active-window results with the old 60--67% mean,
  97--99% median, and P10=0 summaries.
- [ ] Explain every remaining mean/median discrepancy quantitatively.

### 5.6 Acceptance gates

- At least 30 aligned one-second samples remain in every publication window.
- Projection time plus restore/reset time plus measured gaps agrees with wall
  time within 1%.
- The zero-sample fraction is explained by timestamped events rather than
  discarded as noise.
- The same GPU UUID is observed by Julia and the monitoring tool.
- The report states the measured utilization even if it is below 80%.
- High utilization is treated as supplementary evidence, not proof that
  divergence is absent.

---

## 6. Experiment C: Genuinely Parametric-Similar SOC Validation

### 6.1 Question

When cone inputs and diagonal scalings are genuinely similar across a
parametric family, how much root-work variation occurs inside thread-wise
warps, and how efficiently do thread-wise and warp-wise kernels execute?

### 6.2 Common configuration

```text
projection       = diagonally rescaled SOC, type 22
cone_count       = 1048576
cone_dimension   = 10
strategies       = threadWise, warpWise, blockWise
seeds            = 2026:2035
abs_tol          = 1e-12
rel_tol          = 1e-12
arithmetic       = Float64
warmups          = 5
measured_launches = 10
```

Thread-wise and warp-wise are the primary strategies. Block-wise is a
descriptive reference and is not included in the primary divergence ratio.

### 6.3 Input construction

For each seed, generate one base direction and one centered log diagonal:

\[
u_\star=\frac{z_\star}{\lVert z_\star\rVert_2},
\qquad z_\star\sim\mathcal N(0,I_q),
\]

\[
\ell_\star
=\eta_\star-\frac{\mathbf1^\top\eta_\star}{q}\mathbf1,
\qquad \eta_\star\sim\mathcal N(0,I_q).
\]

For cone \(i\), draw independent normalized perturbations and define

\[
\widetilde u_i
=u_\star+\delta_u
  \frac{\xi_i}{\lVert\xi_i\rVert_2},
\qquad
u_i=\frac{\widetilde u_i}{\lVert\widetilde u_i\rVert_2},
\]

\[
\widetilde\ell_i=\ell_\star+\delta_D\zeta_i,
\qquad
\ell_i=\widetilde\ell_i
-\frac{\mathbf1^\top\widetilde\ell_i}{q}\mathbf1,
\qquad
d_{ij}=e^{\ell_{ij}}.
\]

Reject and redraw a diagonal perturbation if any
\(d_{ij}\notin[10^{-3},10^3]\). Use

\[
\delta_u=\delta_D
\in\{0,10^{-4},10^{-3},10^{-2}\},
\]

and set

\[
t_i=0.20\lVert\operatorname{diag}(d_i)u_i\rVert_2
\]

so that the intended workload takes the positive-root path. The diagnostic
branch code must verify this construction for every cone.

### 6.4 Cold-start panel

For every seed, perturbation level, and strategy:

- reset the stored scalar root to zero;
- measure projection time with the global timing protocol;
- record branch code, interval expansions, Newton attempts/accepts, bisection
  iterations, oracle evaluations, termination reason, and final residual;
- compute within-warp median spread, P90 spread, maximum spread, and modeled
  active-lane efficiency for the natural cone order.

Do not compare different perturbation levels as if they were the same
mathematical workload. The perturbation sweep is descriptive.

### 6.5 Warm-start parametric-sequence panel

For each base seed, generate a sequence of five nonidentical nearby parameter
states using a fixed perturbation amplitude \(10^{-3}\). Use independent,
recorded perturbation directions for successive states.

For state \(k=1,\ldots,4\), measure:

- **cold:** root state reset to zero;
- **warm:** root state copied from the completed projection of state \(k-1\).

Every technical replicate must restore the same prior-state root vector before
the timed launch. Report warm/cold runtime ratio and warm/cold root-work
distributions. Do not use warm-start speedup as evidence that cold-start
divergence is absent.

### 6.6 Utilization and profiling

For all ten seeds at \(\delta=0\) and \(\delta=10^{-2}\), collect:

- aligned 30-second sustained activity for thread-wise and warp-wise;
- Nsight Compute SM and memory Speed-of-Light metrics;
- separate L1/TEX, L2, and DRAM throughput when available;
- occupancy, eligible and issued warps per cycle;
- branch/barrier/scoreboard stall metrics;
- source-correlated active-thread or predication metrics.

### 6.7 Acceptance gates

- All cones take the diagnostically verified intended root path.
- All ten seeds pass correctness and termination checks.
- Every perturbation level contains exactly ten independent seed summaries.
- Similarity parameters and rejection counts are stored in the manifest.
- Cold and warm results are reported in separate panels.
- The conclusion describes observed root-work variation rather than assuming
  that “similar scaling” means identical convergence.

---

## 7. Experiment D: Branch Profiling and Warp-Wise Ordering Controls

### 7.1 Questions

1. Why does the existing branch-interleaved warp-wise workload slow down by
   \(76.5\%\)?
2. Is \(\Theta=R_T/R_W\) a defensible divergence interaction, or is \(R_W\)
   dominated by block composition, locality, or scheduling effects?

### 7.2 Reprofile the completed branch experiment

Profile all four branch cells for all ten seeds:

```text
layout   = grouped, interleaved
strategy = threadWise, warpWise
seeds    = 2026:2035
```

Use one warmed, restored-state target launch per process. Save the complete
metric inventory, `.ncu-rep`, and CSV export.

Collect:

- SM throughput;
- aggregate memory throughput;
- L1/TEX throughput and hit rate;
- L2 throughput and hit rate;
- DRAM throughput;
- global load/store sectors and bytes;
- achieved occupancy;
- eligible and issued warps per cycle;
- instructions per cycle;
- executed instructions;
- branch-target uniformity;
- average active and predicated-on threads where available;
- barrier, branch-resolving, long-scoreboard, short-scoreboard, wait, and
  not-selected stall shares.

Do not infer a latency bottleneck unless the relevant stall counters support
that conclusion.

### 7.3 Warp-wise block-balanced control

Construct two additional exact permutations for the warp-wise kernel. With a
256-thread block, each warp-wise block processes eight cones.

For both difficulty quartiles and branch classes:

- **paired layout:** class order
  \([0,0,1,1,2,2,3,3]\) inside every eight-cone block;
- **alternating layout:** class order
  \([0,1,2,3,0,1,2,3]\) inside every eight-cone block.

Both layouts contain exactly two cones from each class in every warp-wise
block. Thus, block-level class composition, total mathematical work, and cone
multiset are fixed while only cross-warp ordering changes.

Measure both layouts with the warp-wise kernel for:

- same-branch root-work quartiles;
- top-level branch classes;
- all ten seeds;
- five warm-ups and ten measured launches.

### 7.4 Interpretation gate

Define

\[
R_{W,\mathrm{balanced}}
=\frac{T_{W,\mathrm{alternating}}}
       {T_{W,\mathrm{paired}}}.
\]

- If the upper one-sided 95% bound is below 1.05, the balanced control is
  sufficiently invariant for a limited interaction analysis.
- Otherwise, treat warp-wise ordering as mapping-specific behavior and do not
  interpret \(\Theta\) as an isolated causal divergence effect.
- Regardless of this gate, report \(R_T\) and \(A\) directly.

### 7.5 Acceptance gates

- Hashes prove exact multiset identity.
- Every warp-wise block has identical class counts in both control layouts.
- Load/store byte counts are equal unless the profiler identifies a
  data-dependent instruction path that changes them.
- NCU replay duration is excluded from publication timing.
- Any explanation of \(R_W\) is tied to measured counters, not speculation.

---

## 8. Experiment E: Direct Active-Lane Diagnostic

### 8.1 Question

Does the actual root-loop execution show the lane inactivity predicted by the
per-cone oracle-count model?

### 8.2 Diagnostic design

Create a diagnostic-only thread-wise kernel guarded by
`PDCS_PROFILE_ACTIVE_LANES`. It must never be linked into the timing binary.

For a deterministic sample of 4,096 complete thread-wise warps per seed:

- use a warp ballot at each interval-expansion and bisection-loop iteration;
- record the active-mask popcount;
- record the loop type and iteration index;
- record the terminal branch code;
- write to a preallocated per-warp buffer with no global atomics in the root
  loop;
- stop with an explicit invalid diagnostic result if the record capacity is
  exceeded.

Collect this diagnostic for:

- root-work grouped and interleaved layouts;
- branch grouped and interleaved layouts;
- parametric-similar \(\delta=0\) and \(10^{-2}\);
- all ten seeds.

### 8.3 Derived measures

Report:

- active-lane count by loop iteration;
- total active lane-iterations;
- fraction of predicated-off lane-iterations;
- measured lane efficiency

\[
E_w^{\mathrm{mask}}
=\frac{\sum_k a_{w,k}}
       {32K_w},
\]

  where \(a_{w,k}\) is the active-lane count at iteration \(k\) and \(K_w\)
  is the number of recorded iterations for warp \(w\);
- agreement between \(E_w^{\mathrm{mask}}\) and the work-count model \(E_w\);
- separate expansion-loop and bisection-loop summaries.

### 8.4 Acceptance gates

- Diagnostic and uninstrumented projected values agree under the common mixed
  tolerance.
- Diagnostic branch and iteration counters agree with existing per-cone
  counters.
- Every reported warp has a complete record with no buffer overflow.
- Diagnostic-kernel time is never compared with production-kernel time.
- The response distinguishes measured mask efficiency from Nsight
  branch-target uniformity.

---

## 9. Experiment F: SOC Dimension Sensitivity

### 9.1 Question

Does the observed divergence cost change as more coordinates are processed by
each cone?

### 9.2 Fixed-count design

Use one fixed cone count that is large enough to saturate the GPU and feasible
for every dimension:

```text
cone_count     = 262144
cone_dimension = 10, 32, 100
projection     = diagonally rescaled SOC, type 22
strategies     = threadWise, warpWise
layouts        = grouped, interleaved
seeds          = 2026:2035
```

Run two panels:

1. same-branch root-work quartiles;
2. four top-level branch classes.

At each dimension, apply the predeclared family-selection gate using diagnostic
work counts before examining timing. If the positive-root family does not
produce sufficient bisection separation, use the negative-root contingency
and report that the variation is in interval expansion or total oracle work.

### 9.3 Outputs

For every dimension and panel, report:

- per-cone expansion, bisection, and oracle distributions;
- grouped/interleaved within-warp spread and mask efficiency;
- thread-wise and warp-wise timing;
- \(R_T\), \(R_W\), \(A\), and confidence intervals;
- achieved occupancy;
- SM, L1/TEX, L2, and DRAM throughput;
- active-window sustained SM activity.

### 9.4 Acceptance gates

- The same cone count is used at all three dimensions.
- A preflight memory calculation demonstrates at least 20% free HBM headroom.
- Every manipulation passes before timing is analyzed.
- Dimension-dependent conclusions are based on paired seed-level ratios.
- If thread-wise is not the solver-selected strategy at a larger dimension,
  state this explicitly rather than presenting the forced-kernel result as a
  recommended configuration.

---

## 10. Experiment G: Diagonally Rescaled Exponential-Cone Divergence

### 10.1 Objective

Repeat the divergence analysis for primal diagonally rescaled exponential
cones without importing unreconciled timings from older reports.

### 10.2 Common configuration

```text
cone_count      = 1048576
cone_dimension  = 3
projection      = primal diagonal exponential cone, type 27
strategies      = threadWise, warpWise, blockWise
seeds           = 2026:2035
warmups         = 5
measured_launches = 10
arithmetic      = Float64
```

Before this experiment, reconcile the previous hundreds-fold timing
difference by verifying:

- projected cone count;
- timer units;
- timed interval;
- warm-up and synchronization;
- strategy-to-kernel mapping;
- input and diagonal generator;
- stopping tolerances;
- cold or warm root state.

No old exponential-cone timing may be used unless all eight fields match.

### 10.3 EXP-1: Independent heterogeneity sweeps

Run an input sweep:

```text
sigma_x = 0.1, 0.5, 1, 2, 5, 10
sigma_D = 1
```

and a diagonal sweep:

```text
sigma_x = 1
sigma_D = 0.1, 0.5, 1, 2, 5, 10
```

Generate input coordinates and diagonal entries from independent random
streams:

\[
x_j\sim\mathcal N(0,\sigma_x^2),
\]

\[
D_j=\operatorname{clamp}
\left(\lvert\sigma_D Z_j\rvert,10^{-3},10^3\right),
\qquad Z_j\sim\mathcal N(0,1).
\]

For every value, strategy, and seed, report timing, branch fractions,
iteration distributions, convergence failures, and correctness.

### 10.4 EXP-2: Same-iteration and branch-ordering experiments

Using diagnostic counters:

1. generate a pool of converged root-search cones;
2. partition cones into four iteration-count quartiles;
3. construct exact grouped and interleaved permutations of the same multiset;
4. separately construct equal-mass verified top-level branch classes;
5. apply the same 128-cone block-balanced thread-wise layout used for SOCs;
6. apply the warp-wise block-balanced control from Experiment D.

Run thread-wise and warp-wise timing for all ten seeds. Report \(R_T\), \(A\),
the balanced \(R_W\), and \(\Theta\) only when its control gate passes.

### 10.5 EXP-3: Parametric-similar cold and warm sequence

Generate one base input/log-diagonal pair per seed and perturb both with

\[
\delta\in\{0,10^{-4},10^{-3},10^{-2}\}.
\]

Run cold starts at every perturbation level and a five-state nearby sequence
with warm roots copied from the preceding nonidentical state. Report
warm/cold timing ratios and iteration distributions separately.

### 10.6 EXP-4: Profiling and utilization

For all ten seeds, profile:

- the most similar case;
- the most heterogeneous completed case;
- iteration-grouped and iteration-interleaved;
- branch-grouped and branch-interleaved;
- thread-wise and warp-wise strategies.

Collect the profiler and active-window utilization metrics defined in
Experiments B and D.

### 10.7 Acceptance gates

- All sigma-sweep rows are completed or explicitly marked invalid with a
  recorded failure reason.
- Correctness is verified against the CPU exponential-cone projection.
- No SOC scale-invariance conclusion is transferred to exponential cones.
- Workload differences and within-warp divergence effects are analyzed
  separately.
- Exponential-cone conclusions appear in their own tables.

---

## 11. Experiment H: Application-Level Projection Profiling

### 11.1 Objective

Determine whether the synthetic divergence regimes resemble projection work
encountered during complete PDCS runs and measure projection utilization in
context.

### 11.2 Application families

Use ten generated instances or parameter seeds, 2026--2035, for each selected
family:

1. **Fisher market equilibrium:** many three-dimensional exponential cones.
2. **Multi-period portfolio optimization:** many SOC or rescaled-SOC blocks.
3. **Lasso or another high-dimensional SOC model:** a contrasting
   few-cone/high-dimension regime.

Choose one medium and one large size per family from the manuscript benchmark
set. The exact instance dimensions must be copied into the run manifest before
execution and must not be changed after timing is inspected.

### 11.3 Instrumentation

Add NVTX ranges around:

- sparse matrix-vector products;
- SOC projection;
- exponential-cone projection;
- adaptive-step and restart logic;
- residual and termination checks.

In a separate diagnostic build, sample per-cone:

- projection path;
- expansion, Newton, bisection, and oracle counts;
- final residual;
- warm-start usage;
- within-warp work spread under the solver’s actual cone ordering.

Do not collect diagnostic counters in the performance run.

### 11.4 Measurements

For every instance seed, report:

- total solver time;
- total projection time and fraction of solver time;
- number of projection calls;
- selected projection strategy;
- natural-order branch and work distributions;
- natural-order modeled and measured lane efficiency;
- end-to-end and projection-range GPU activity;
- projection-range SM and memory throughput;
- achieved solution tolerance and iteration count.

### 11.5 Acceptance gates

- All compared runs reach the same manuscript stopping criteria.
- NVTX range time reconciles with total solver time.
- Diagnostic and performance builds produce equivalent solutions.
- The natural-order workload is not reordered to manufacture divergence.
- Synthetic-to-application comparisons use distributions and effect sizes,
  not visual similarity alone.

---

## 12. Experiment I: Integrated Statistical Analysis

### 12.1 Primary reported quantities

For each paired grouped/interleaved experiment and seed:

\[
R_{T,s}=\frac{T_{s,T,I}}{T_{s,T,G}},
\qquad
A_s=\frac{T_{s,T,I}}{T_{s,W,I}},
\]

\[
R_{W,s}=\frac{T_{s,W,I}}{T_{s,W,G}},
\qquad
\Theta_s=\frac{R_{T,s}}{R_{W,s}}.
\]

Use geometric means on the log scale. Report:

- geometric mean ratio;
- paired-\(t\) 95% CI;
- seed-block bootstrap 95% CI;
- one-sided 95% bound for the 5% materiality decision;
- all ten seed-level ratios in a supplement.

### 12.2 Decision language

Use the following interpretation order:

1. Report the observed thread-wise layout effect \(R_T\).
2. State whether its one-sided upper bound is below 1.05.
3. Report whether thread-wise remains faster than warp-wise using \(A\).
4. Report \(R_W\) as an ordering-control diagnostic.
5. Report \(\Theta\) as causal only if the balanced warp-wise control passes
   its invariance gate.
6. State the measured utilization distribution.
7. Limit the conclusion to the tested workload.

For the existing SOC results, the appropriate factual structure is:

- same-branch negative-root root-work interleaving produces a 2.3% net
  thread-wise layout effect, below the predeclared 5% margin;
- top-level branch mixing produces a material 15.4% thread-wise slowdown;
- thread-wise nevertheless remains faster than warp-wise for the tested
  \(m=2^{20}\), \(d=10\) H100 workloads;
- the positive-root bisection counts are nearly constant, so the primary
  same-branch stress test varies interval-expansion/oracle work rather than
  widely different bisection counts.

Do not replace these statements with “divergence has no penalty.”

---

## 13. Required Publication Tables

### Table 1: SOC manipulation and timing

| Divergence source | Layout | Work spread | Mask efficiency | Thread time | Warp time | \(R_T\) | \(A\) |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Same-branch root work | Grouped | Seed-block summary | Seed-block summary | Seed-block summary | Seed-block summary | Reference | Reference |
| Same-branch root work | Interleaved | Seed-block summary | Seed-block summary | Seed-block summary | Seed-block summary | Estimate with CI | Estimate with CI |
| Top-level branches | Grouped | Verified paths | Seed-block summary | Seed-block summary | Seed-block summary | Reference | Reference |
| Top-level branches | Interleaved | Verified paths | Seed-block summary | Seed-block summary | Seed-block summary | Estimate with CI | Estimate with CI |

### Table 2: Parametric-similar SOC

| Perturbation | Start | Bisection median/P90/max | Within-warp spread | Thread time | Warp time | \(A\) |
|---:|:---|:---|---:|---:|---:|---:|
| \(0\) | Cold | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ratio with CI |
| \(10^{-4}\) | Cold | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ratio with CI |
| \(10^{-3}\) | Cold | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ratio with CI |
| \(10^{-2}\) | Cold | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ratio with CI |
| \(10^{-3}\) sequence | Warm | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ten-seed summary | Ratio with CI |

### Table 3: Active-window utilization and profiler characterization

| Experiment | Kernel | Layout | SM mean/median/P10/P90/peak | SM SOL | L1/TEX SOL | L2 SOL | DRAM SOL | Occupancy | Issued warps/cycle |
|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| Root work | Thread | Grouped | Aligned 10-seed summary | Counter | Counter | Counter | Counter | Counter | Counter |
| Root work | Thread | Interleaved | Aligned 10-seed summary | Counter | Counter | Counter | Counter | Counter | Counter |
| Root work | Warp | Grouped | Aligned 10-seed summary | Counter | Counter | Counter | Counter | Counter | Counter |
| Root work | Warp | Interleaved | Aligned 10-seed summary | Counter | Counter | Counter | Counter | Counter | Counter |
| Branch | Thread/Warp | Grouped/interleaved | Corresponding summaries | Counter | Counter | Counter | Counter | Counter | Counter |

### Table 4: SOC dimension sensitivity

| Dimension | Root-work family | Thread \(R_T\) with CI | \(A\) with CI | Mask efficiency change | Utilization |
|---:|:---|---:|---:|---:|---:|
| 10 | Frozen family | Ten-seed estimate | Ten-seed estimate | Ten-seed estimate | Aligned summary |
| 32 | Frozen family | Ten-seed estimate | Ten-seed estimate | Ten-seed estimate | Aligned summary |
| 100 | Frozen family | Ten-seed estimate | Ten-seed estimate | Ten-seed estimate | Aligned summary |

### Table 5: Exponential-cone divergence

Report sigma sweeps, iteration ordering, branch ordering, parametric sequence,
profiling, and utilization in separate panels. Do not combine SOC and
exponential-cone timings in one ratio.

### Table 6: Application-level relevance

| Family | Size | Projection type | Strategy | Projection fraction | Natural work spread | Mask efficiency | Projection-range utilization |
|:---|:---|:---|:---|---:|---:|---:|---:|
| Fisher | Medium/Large | Exponential | Recorded | Ten-instance summary | Summary | Summary | Summary |
| Portfolio | Medium/Large | SOC | Recorded | Ten-instance summary | Summary | Summary | Summary |
| Lasso/SOC | Medium/Large | SOC | Recorded | Ten-instance summary | Summary | Summary | Summary |

---

## 14. Artifact Layout

Create one immutable root directory:

```text
benchmark/results/rebuttal/additional_experiments_2/<UTC-run-id>/
├── environment/
│   ├── environment.txt
│   ├── gpu_snapshot.csv
│   ├── compiler_commands.txt
│   └── source_hashes.sha256
├── source_snapshot/
│   ├── git_status.txt
│   ├── git_diff_binary.patch
│   └── untracked_sources/
├── baseline_audit/
├── utilization/
│   ├── raw/
│   ├── aligned/
│   └── utilization_summary.csv
├── soc_parametric/
│   ├── cold/
│   ├── warm/
│   └── manifests/
├── soc_branch_profile/
├── soc_active_lanes/
├── soc_dimensions/
├── exp_divergence/
├── applications/
├── ncu/
│   ├── available_metrics.txt
│   ├── reports/
│   └── csv/
├── nsys/
├── correctness/
├── seed_effects.csv
├── effect_estimates.csv
└── publication_tables.md
```

Every summary row must contain the raw-artifact path from which it was
generated.

---

## 15. Final Execution Checklist

### Evidence repair

- [ ] Freeze the exact source and binary snapshot.
- [ ] Reconcile correctness definitions and values.
- [ ] Regenerate the existing ten-seed SOC effect estimates.
- [ ] Mark all older grouped/interleaved results as superseded.

### Utilization

- [ ] Implement the ready/start/done synchronization.
- [ ] Collect 10-seed active-window utilization for both SOC divergence
  sources and all four thread/warp cells.
- [ ] Reconcile utilization samples with the time ledger.
- [ ] Replace the old utilization summary.

### Parametric SOC

- [ ] Run all four perturbation levels with cold starts for ten seeds.
- [ ] Run the five-state warm-start sequence for ten seeds.
- [ ] Collect root-work, timing, correctness, utilization, and profiler data.

### Branch and causal controls

- [ ] Profile the existing branch experiment for ten seeds.
- [ ] Run the warp-wise block-balanced control for ten seeds.
- [ ] Apply the \(\Theta\) interpretation gate.

### Direct divergence diagnostic

- [ ] Implement and validate active-mask sampling.
- [ ] Collect all specified layouts and parametric cases for ten seeds.
- [ ] Compare measured mask efficiency with modeled efficiency.

### Generalization

- [ ] Complete the fixed-count SOC dimension experiment at dimensions
  10, 32, and 100.
- [ ] Reconcile exponential-cone timing scope.
- [ ] Complete exponential-cone heterogeneity, ordering, parametric,
  profiling, and utilization experiments with ten seeds.
- [ ] Complete application-level projection profiling with ten instance seeds
  per family.

### Publication

- [ ] Generate every table directly from immutable summaries.
- [ ] Verify that no Nsight replay time is presented as runtime.
- [ ] Verify that every timing claim has a ten-seed confidence interval.
- [ ] Verify that utilization uses aligned windows.
- [ ] Verify that SOC and exponential-cone claims remain separate.
- [ ] Verify that the rebuttal reports the 15.4% branch penalty.
- [ ] Verify that no text claims divergence is absent.
- [ ] Verify that every conclusion states its hardware and workload scope.

---

## 16. Completion Criteria

This plan is complete only when:

1. all required correctness gates pass for all ten seeds;
2. the utilization mean/median/P10 inconsistency is resolved;
3. the parametric-similar SOC experiment is complete;
4. the branch \(R_W=1.765\) behavior is either explained by measured counters
   or explicitly left as a mapping-specific ordering effect;
5. the causal use of \(\Theta\) passes the balanced-control gate or is removed;
6. direct active-lane evidence is available for the root and branch loops;
7. SOC dimension sensitivity is reported;
8. exponential-cone timing has one reconciled measurement scope and the full
   ten-seed divergence study is complete;
9. application-level projection behavior is reported;
10. every publication number is traceable to a raw artifact and exact source
    snapshot.

The final R1-3 response should then lead with the measured thread-wise effects,
not with a blanket claim of “no divergence penalty.”
