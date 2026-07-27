# R1-3 Completion Plan: Parametric Similarity, Warp Divergence, and GPU Utilization

## 0. Purpose and Scope

This document specifies the remaining work needed to answer Reviewer R1-3
completely and conservatively. It is intentionally narrower than the earlier
exhaustive plan. The target is the reviewer's specific question:

> Because the rescaled-cone projection uses root finding, different threads may
> take different branches or require different numbers of bisection iterations,
> which may cause warp divergence. This may occur less often in parametric
> instances with similar cone scaling. GPU-utilization statistics during the
> projections, particularly utilization in the 80--90% range or higher, would
> complement Section 4.

The completed experiments already quantify two deliberately heterogeneous
cases:

1. cones following the same root-search branch but having different amounts of
   root-search work; and
2. cones following four different top-level projection branches.

They also include an aligned sustained-utilization measurement. The one missing
experiment that directly follows from the reviewer's wording is a genuinely
parametric-similar SOC experiment in which the similarity of both the cone
vectors and diagonal scalings is controlled continuously.

### 0.1 Required work

The following work is required:

1. audit and correct the terminology and unsupported interpretations in the
   existing report;
2. implement and run the parametric-similar rescaled-SOC experiment defined in
   this document;
3. use exactly ten seeds, 2026--2035;
4. collect endpoint utilization for the parametric-similar experiment;
5. analyze the new and existing results with seed-paired statistics;
6. write the rebuttal using conclusions that are supported directly by the
   measurements.

### 0.2 Work explicitly excluded from the required scope

The following experiments are not required to answer R1-3:

- fixed-count experiments with \(m=100\) or \(m=1{,}000\);
- SOC-dimension sensitivity;
- exponential-cone divergence;
- application-level end-to-end profiling;
- additional GPU architectures;
- more than ten seeds;
- warm-start parametric sequences;
- direct hardware active-lane-mask instrumentation;
- new Nsight Compute profiling.

These experiments may be valuable for a broader paper claim, but they do not
need to delay the R1-3 response. The final response must not claim
generalization to cone types, dimensions, GPUs, or application workloads that
were not tested.

### 0.3 Decision on causal claims

The required response will use the directly measured thread-wise slowdown

\[
R_T=\frac{T_{\mathrm{thread,interleaved}}}
          {T_{\mathrm{thread,grouped}}}
\]

as the practical measure of layout-induced cost. It will not use the
interaction ratio

\[
\Theta=\frac{R_T}{R_W}
\]

as causal proof of thread-level divergence. The completed branch experiment has
\(R_W=1.765\), even though one warp processes one cone in the warp-wise kernel.
Therefore, layout, block composition, scheduling, and locality effects are
present in \(R_W\). Removing the causal interpretation of \(\Theta\) avoids the
need for a new block-balanced control solely for R1-3.

---

## 1. Existing Evidence to Retain

The following completed measurements remain relevant and should not be rerun
unless their raw artifacts fail the audit in Section 2.

### 1.1 Same-root-branch work imbalance

For \(m=2^{20}\), \(d=10\), type-22 rescaled SOCs, and ten seeds:

| Strategy | Grouped (ms) | Interleaved (ms) |
|:--|--:|--:|
| Thread-wise | 5.93 | 6.09 |
| Warp-wise | 19.61 | 20.57 |

The primary thread-wise effect is

\[
R_T=1.023,\qquad 95\%\ \mathrm{CI}=[1.020,1.026].
\]

Thus, deliberately increasing within-warp root-work heterogeneity caused a
measurable but small \(2.3\%\) thread-wise slowdown. The result is below the
predeclared \(5\%\) materiality margin. The bisection count itself was nearly
constant at 38--39 iterations; the observed work variation came primarily from
interval expansion and the resulting oracle evaluations.

### 1.2 Top-level branch mixing

| Strategy | Grouped (ms) | Interleaved (ms) |
|:--|--:|--:|
| Thread-wise | 3.64 | 4.20 |
| Warp-wise | 10.79 | 19.03 |

The thread-wise effect is

\[
R_T=1.154,\qquad 95\%\ \mathrm{CI}=[1.152,1.156].
\]

Therefore, mixing different top-level paths within a hardware warp caused a
material \(15.4\%\) slowdown. This effect must be acknowledged explicitly.
Nevertheless, the thread-wise strategy remained approximately \(4.5\times\)
faster than the warp-wise strategy in the interleaved layout.

### 1.3 Existing aligned utilization

The current report contains 80 aligned sustained-utilization cells:

```text
experiment = same-root-branch work, top-level branch
layout     = grouped, interleaved
strategy   = thread-wise, warp-wise
seeds      = 2026:2035
```

The reported mean utilization values range from \(87.9\%\) to \(98.3\%\).
These results address the reviewer's request for GPU-utilization statistics,
provided that the audit in Section 2 confirms the query field, sample window,
and time ledger.

---

## 2. Work Package A: Audit the Existing Results

### 2.1 Objective

Before adding the new experiment, reconcile the current report with the saved
seed-level data and remove statements that are not supported by the measured
counters.

### 2.2 Immutable run identity

For each existing run, save or verify:

- run ID and UTC timestamp;
- Git commit and dirty-worktree status;
- SHA-256 hashes of the benchmark, generator, analyzer, and CUDA sources;
- Julia version;
- CUDA.jl version;
- CUDA Toolkit version;
- NVIDIA driver version;
- Nsight Compute version, if profiling artifacts are retained;
- GPU model, UUID, and PCI bus ID;
- arithmetic precision;
- compiler flags and CUDA architecture;
- cone count, dimension, tolerance, seeds, warm-ups, and measured launches.

The report must distinguish the two repeated executions by run ID. Because they
used the same seeds, hardware, software, and generator, the second run is a
**repeatability run**, not an independent statistical replication.

### 2.3 Seed accounting

Use exactly:

```text
2026, 2027, 2028, 2029, 2030,
2031, 2032, 2033, 2034, 2035
```

The correct description is:

> All 10/10 seeds passed in each of the two repeatability runs.

Do not write "20/20 seeds" and do not combine the two executions into a
20-seed inferential sample.

### 2.4 Correctness reconciliation

Define all correctness fields mathematically in the report and in the
machine-readable metadata. For a GPU output \(p^{\mathrm{GPU}}\) and CPU
reference \(p^{\mathrm{CPU}}\), report at least:

\[
e_{\mathrm{abs}}
=\left\|p^{\mathrm{GPU}}-p^{\mathrm{CPU}}\right\|_\infty,
\]

\[
e_{\mathrm{scaled}}
=
\max_j
\frac{
  \left|p^{\mathrm{GPU}}_j-p^{\mathrm{CPU}}_j\right|
}{
  \tau_{\mathrm{abs}}
  +\tau_{\mathrm{rel}}
   \max\!\left(
     |p^{\mathrm{GPU}}_j|,
     |p^{\mathrm{CPU}}_j|
   \right)
}.
\]

Also report:

- thread-wise versus warp-wise error on the same ordered input;
- grouped versus interleaved error after inverse permutation;
- primal-cone feasibility residual;
- polar-cone feasibility residual;
- complementarity/Moreau residual;
- termination reason;
- number of `MAX_ITER` events;
- number of nonfinite values.

The cone residuals must use the exact type-22 scaling convention implemented by
the production code. The formulas used by the analyzer must be recorded in the
run manifest rather than being referred to only as "scaled error."

If a table contains a nonzero cross-strategy or cross-layout error, the text
must not claim that the corresponding error is exactly zero.

### 2.5 Utilization terminology and window audit

Record the exact monitoring command. If the queried field is
`utilization.gpu`, label the result **GPU utilization**, not **SM busy**.
`utilization.gpu` is a device-level utilization measure and is not a direct
measurement of active lanes or per-kernel efficiency.

For each utilization cell, retain:

- `READY`, `START`, `FIRST_PROJECTION_START`,
  `LAST_PROJECTION_STOP`, and `DONE` timestamps;
- the exact sampling interval;
- raw samples without deleting zero-utilization observations;
- the number of samples before and after the stabilization window;
- mean, median, P10, P90, peak, and zero-sample fraction;
- projection launches per second;
- cumulative CUDA-event projection time;
- restore/reset time;
- host-gap time;
- total wall time;
- GPU UUID observed by the program and monitoring process.

The time ledger must satisfy

\[
\frac{
\left|
 t_{\mathrm{projection}}
+t_{\mathrm{restore/reset}}
+t_{\mathrm{host\ gaps}}
-t_{\mathrm{wall}}
\right|
}{t_{\mathrm{wall}}}
\le 0.01.
\]

If the raw artifacts do not permit this reconciliation, rerun the affected
utilization cells using the protocol in Section 6. Do not remove low samples
based on their numerical value.

### 2.6 Required textual corrections

Apply all of the following corrections to the final report:

1. Replace "independently reproduced" with "repeated with the same ten seeds."
2. Replace "20/20 seeds passed" with "10/10 seeds passed in each run."
3. Replace "SM busy" with "GPU utilization" when the source is
   `nvidia-smi utilization.gpu`.
4. Replace "all layout differences are at most 2 percentage points" with
   "all mean layout differences are at most 2.7 percentage points."
5. Do not state that high utilization proves divergence is absent.
6. Do not state that branch mixing has no material thread-wise penalty; its
   measured penalty is \(15.4\%\).
7. Do not describe the existing experiment as "maximized divergence." Use
   "deliberately induced" or "adversarially interleaved."
8. Do not call eligible warps "warps waiting on memory." Eligible warps are
   ready to issue.
9. Do not claim that aggregate Memory SOL alone proves L1/L2 saturation.
10. Do not claim that the warp-wise kernel is latency-bound unless the
    supporting stall counters are reported.
11. Do not use Nsight Compute replay duration as publication runtime.
12. Treat \(R_W\) and \(\Theta\) as descriptive unless a block-balanced
    ordering control has been completed.

### 2.7 Acceptance gate

Work Package A passes only if every published value is reproducible from a
saved seed-level file and every interpretation is supported by the named
metric.

---

## 3. Work Package B: Implement the Parametric-Similar Generator

### 3.1 Why a separate harness is recommended

The existing 2-by-2 harness was designed for grouped and interleaved
permutations of a fixed heterogeneous cone multiset. The new experiment varies
the mathematical inputs through a controlled similarity parameter. Mixing
these two designs in one driver would make the cell definitions and artifacts
harder to audit.

Create a dedicated benchmark driver and analyzer, while reusing the existing:

- type-22 production projection kernels;
- GPU selection and environment capture;
- CUDA-event timing helper;
- root-work diagnostic counters;
- correctness checker;
- sustained-utilization wrapper;
- seed-block bootstrap implementation.

Recommended files in the code checkout that produced the existing results:

| File | Action | Purpose |
|:--|:--|:--|
| `benchmark/rescaled_soc_parametric_similar.jl` | Create | Generate, validate, warm, restore, and time all parametric-similarity cells |
| `benchmark/analyze_soc_parametric_similar.jl` | Create | Produce seed summaries, paired ratios, confidence intervals, and publication tables |
| `benchmark/rebuttal/soc_divergence_cases.jl` | Extend | Add a deterministic parametric-similar generator and manifest writer |
| `benchmark/rebuttal/run_active_window_utilization.sh` | Reuse or minimally extend | Run the endpoint sustained-utilization cells |
| `test/rebuttal/soc_parametric_similarity_test.jl` | Create | Test determinism, similarity levels, path gates, and correctness |

If these benchmark files live under a different repository root on the
experiment machine, retain the filenames and directory structure relative to
that code checkout.

No production CUDA algorithm change is expected. A CUDA source may be modified
only if a required diagnostic counter is unavailable, and then only under a
compile-time diagnostic guard. The uninstrumented timing build must remain
behaviorally identical to the production kernel.

### 3.2 Common configuration

Use the same hardware and numerical configuration as the completed experiment:

```text
projection          = type-22 diagonally rescaled SOC
cone_count          = 1048576
cone_dimension      = 10
tail_dimension q    = 9
strategies          = thread-wise, warp-wise
seeds               = 2026:2035
arithmetic          = Float64
abs_tol             = 1e-12
rel_tol             = 1e-12
threads_per_block   = 256
warmup_launches     = 5 per cell
measured_launches   = 10 per cell
seed_statistic      = median of 10 measured launches
root_initialization = zero
```

Only one H100 is used for an inferential run. Record and verify its UUID in
every artifact. Do not distribute different cells across different GPUs.

### 3.3 Base problem for each seed

Let a cone input be \(y_i=(t_i,u_i)\), where
\(u_i\in\mathbb{R}^{q}\) and \(q=9\). For each seed \(s\), draw one base
direction:

\[
z_\star^{(s)}\sim\mathcal N(0,I_q),
\qquad
u_\star^{(s)}
=\frac{z_\star^{(s)}}{\|z_\star^{(s)}\|_2}.
\]

Draw one base log-diagonal:

\[
\eta_\star^{(s)}\sim\mathcal N(0,I_q),
\]

\[
\ell_\star^{(s)}
=
\eta_\star^{(s)}
-
\frac{\mathbf 1^\top\eta_\star^{(s)}}{q}\mathbf 1.
\]

The centered construction gives a diagonal whose geometric mean is one.
Store \(u_\star^{(s)}\), \(\ell_\star^{(s)}\), and their hashes in the
manifest.

### 3.4 Controlled perturbations

Use the four predeclared similarity levels:

\[
\delta\in\{0,10^{-4},10^{-3},10^{-2}\}.
\]

For each cone \(i\), draw an independent direction perturbation
\(\xi_i\sim\mathcal N(0,I_q)\) and define

\[
\widehat{\xi}_i=\frac{\xi_i}{\|\xi_i\|_2},
\]

\[
\widetilde u_i
=u_\star+\delta\widehat{\xi}_i,
\qquad
u_i=\frac{\widetilde u_i}{\|\widetilde u_i\|_2}.
\]

For the diagonal perturbation, draw
\(\zeta_i\sim\mathcal N(0,I_q)\), center it, and normalize its root-mean-square
magnitude:

\[
\zeta_i^c
=
\zeta_i
-
\frac{\mathbf1^\top\zeta_i}{q}\mathbf1,
\]

\[
\widehat\zeta_i
=
\frac{\zeta_i^c}
{\sqrt{\|\zeta_i^c\|_2^2/q}}.
\]

Then define

\[
\widetilde\ell_i
=\ell_\star+\delta\widehat\zeta_i,
\]

\[
\ell_i
=
\widetilde\ell_i
-
\frac{\mathbf1^\top\widetilde\ell_i}{q}\mathbf1,
\qquad
d_{ij}=\exp(\ell_{ij}).
\]

Reject and redraw only the diagonal perturbation if

\[
d_{ij}\notin[10^{-3},10^3]
\]

for any coordinate. Save the number of rejections for every seed and
similarity level.

The same primitive random arrays \(\xi_i\) and \(\zeta_i\) must be reused
across all four \(\delta\) levels within a seed. Consequently, changing
\(\delta\) changes only the perturbation amplitude, not the perturbation
directions. This coupling improves interpretability and supports paired
analysis.

At \(\delta=0\), all cones within a seed must be bitwise identical before
projection. Verify this with hashes and a maximum-difference check.

### 3.5 Root-search path construction

Set

\[
t_i
=0.20
\left\|
\operatorname{diag}(d_i)u_i
\right\|_2.
\]

This construction is intended to place every cone on the positive-root search
path. Before the ten inferential seeds are run, execute a noninferential pilot
with seed 2025 using the fixed coefficient \(0.20\).

The pilot has one purpose: verify that the branch-code interpretation matches
the production code. It must not be used to select a coefficient that creates
a more favorable runtime result. If the fixed construction does not enter the
intended path, stop and correct the mathematical mapping between the generator
and type-22 cone convention before running seeds 2026--2035.

For every inferential cone, record the diagnostic path code. The required gate
is:

\[
\Pr(\text{intended positive-root path})=1
\]

for each seed and each \(\delta\). Do not discard cones that take another path
and do not regenerate an inferential seed conditionally on its measured work.
A path-gate failure invalidates the seed and requires fixing the generator
before restarting the entire ten-seed experiment.

### 3.6 Natural ordering

Keep cones in their deterministic generation order. Do not sort by root-work
count and do not construct grouped/interleaved layouts in this experiment.
Consecutive groups of 32 cones define the natural thread-wise warps.

This is essential: the parametric experiment asks whether naturally similar
problems have similar root-search behavior. Adversarially sorting the cones
would answer a different question already covered by the completed 2-by-2
experiment.

### 3.7 Generator tests

The generator test suite must verify:

1. identical seed and \(\delta\) reproduce identical arrays and hashes;
2. changing the seed changes the base problem and perturbations;
3. all four \(\delta\) levels reuse the same primitive perturbation directions;
4. \(\delta=0\) produces bitwise-identical cones within a seed;
5. the empirical RMS log-diagonal perturbation agrees with \(\delta\);
6. the angular deviation from \(u_\star\) increases with \(\delta\);
7. every diagonal entry is finite and lies in \([10^{-3},10^3]\);
8. the centered log diagonals have numerical mean zero;
9. all inferential cones enter the intended path;
10. all generated values are finite.

---

## 4. Work Package C: Diagnostic Root-Work Experiment

### 4.1 Objective

Measure directly whether increasing parametric similarity reduces variation in
root-finding work among the 32 cones assigned to a thread-wise hardware warp.

### 4.2 Diagnostic records

For every cone, record:

- seed;
- \(\delta\);
- original cone index;
- path code;
- interval-expansion iteration count;
- Newton attempt count, if the implementation uses Newton steps;
- Newton accepted-step count;
- bisection iteration count;
- total root-oracle evaluation count;
- termination reason;
- final bracket width;
- final root residual;
- `MAX_ITER` flag;
- nonfinite flag.

The report must keep expansion and bisection counts separate. The completed
experiment showed that the bisection count was almost fixed while the expansion
count varied. Combining them without the component counts would obscure the
actual source of different execution lengths.

### 4.3 Per-cone and per-seed summaries

For each seed and \(\delta\), report the following for expansion, bisection, and
total oracle evaluations:

- minimum;
- Q10;
- Q25;
- median;
- Q75;
- Q90;
- Q99;
- maximum;
- mean;
- standard deviation;
- coefficient of variation.

### 4.4 Within-warp summaries

For a natural-order warp \(w\) with work counts \(h_{w,1},\ldots,h_{w,32}\),
define:

\[
S_w
=
\max_{\ell=1,\ldots,32} h_{w,\ell}
-
\min_{\ell=1,\ldots,32} h_{w,\ell},
\]

\[
E_w^{\mathrm{model}}
=
\begin{cases}
1,
& \max_\ell h_{w,\ell}=0,\\[4pt]
\displaystyle
\frac{\sum_{\ell=1}^{32}h_{w,\ell}}
{32\max_{\ell=1}^{32}h_{w,\ell}},
& \text{otherwise}.
\end{cases}
\]

Compute these quantities separately for:

- expansion work;
- bisection work;
- total oracle evaluations.

For each seed and \(\delta\), report the median, P90, P99, and maximum of
\(S_w\), and the median, P10, P1, and minimum of
\(E_w^{\mathrm{model}}\).

The report must call \(E_w^{\mathrm{model}}\) a **work-count-based modeled
lane-efficiency proxy**. It is not a hardware measurement of active lanes.

### 4.5 Branch unanimity

For each warp, report:

- number of distinct path codes;
- modal-path fraction.

The intended result is one path code and modal-path fraction one for every
warp. Report the observed values even if the gate fails.

### 4.6 Diagnostic-build invariance

Run one small validation case with both:

- the uninstrumented timing build; and
- the diagnostic build.

Require identical output within the correctness tolerance and identical
termination classifications. Do not use the diagnostic-build runtime as the
publication runtime.

### 4.7 Interpretation

The perturbation sweep is descriptive. Different \(\delta\) levels are
different mathematical workloads, so a timing difference across \(\delta\)
must not be attributed solely to divergence.

The diagnostic experiment must state explicitly whether the distributions of
expansion, bisection, and oracle counts become more concentrated, remain
essentially unchanged, or behave nonmonotonically as \(\delta\) decreases. The
statement must give the numerical within-warp spread at both
\(\delta=10^{-2}\) and the identical \(\delta=0\) endpoint.

A null result is valid: if bisection and oracle work remain nearly identical at
every \(\delta\), report that the root-search control flow was already uniform
over the tested similarity range.

---

## 5. Work Package D: Production Timing Experiment

### 5.1 Timing cells

Run:

```text
4 perturbation levels
× 2 strategies
× 10 seeds
= 80 seed-strategy-perturbation timing cells
```

Each cell contains:

- 5 unmeasured warm-up launches;
- 10 measured projection launches.

This is 1,200 total projection launches including warm-ups. Generation,
host-to-device transfer, sorting, diagnostics, restoration, and root-state
reset must be outside the timed interval.

### 5.2 Fairness requirements

Within each seed and \(\delta\):

- thread-wise and warp-wise kernels receive exactly the same input arrays;
- the arrays have identical order;
- the same tolerance and Float64 arithmetic are used;
- the root state is reset to zero before every measured launch;
- the input is restored from an immutable device template before every launch;
- output buffers are disjoint or identically reset;
- one CUDA stream is used unless the production comparison already uses a
  different documented stream policy;
- every measurement is synchronized by CUDA events.

### 5.3 Counterbalancing

The eight cells in a seed are:

\[
\{T,W\}
\times
\{0,10^{-4},10^{-3},10^{-2}\}.
\]

Warm every cell five times before collecting inferential timings. Then collect
ten measurement rounds. In each round, execute every cell once in a
counterbalanced order. Use an eight-condition Williams design for the first
eight rounds and the first two rows of a cyclically shifted design for rounds
9 and 10. Shift the starting design row by seed index and save the exact order
in the manifest.

This prevents a fixed strategy or \(\delta\) from always being measured early
or late in a run.

### 5.4 Raw timing records

Save one row per launch with:

- run ID;
- seed;
- \(\delta\);
- strategy;
- warm-up or measured status;
- measurement round;
- order position;
- kernel duration in milliseconds;
- GPU UUID;
- SM clock;
- memory clock;
- power;
- temperature;
- error/status fields.

Do not delete slow launches merely because they are statistical outliers. The
seed-level median of all ten valid measured launches is the predeclared
technical-replicate statistic.

### 5.5 Seed-level quantities

Let \(T_{s,K,\delta}\) be the median of the ten measured launches for seed
\(s\), strategy \(K\), and perturbation level \(\delta\).

For each \(\delta\), define the strategy ratio:

\[
A_s(\delta)
=
\frac{T_{s,T,\delta}}{T_{s,W,\delta}}.
\]

Report:

\[
\operatorname{GM}[A(\delta)]
=
\exp\!\left(
\frac{1}{10}
\sum_{s=2026}^{2035}\log A_s(\delta)
\right).
\]

When \(A(\delta)<1\), the thread-wise strategy is faster; its speedup is
\(1/A(\delta)\).

For descriptive endpoint comparisons within a strategy, define:

\[
P_{s,K}
=
\frac{T_{s,K,10^{-2}}}{T_{s,K,0}}.
\]

Report \(P_{s,T}\) and \(P_{s,W}\), but do not call them pure divergence
penalties because the mathematical inputs differ between the endpoints.

### 5.6 Statistical analysis

The statistical unit is the seed, not an individual launch and not a cone.

For every ratio:

1. compute the ratio within each seed;
2. analyze its logarithm;
3. report the geometric mean;
4. compute a 95% two-sided confidence interval using 10,000 seed-block
   bootstrap resamples;
5. use one bootstrap draw to resample all \(\delta\) and strategy cells of a
   seed together;
6. report all ten seed-level ratios in a supplementary table.

For cell times, report:

- mean of the ten seed medians;
- standard deviation across seed medians;
- minimum and maximum seed median;
- median across seed medians.

Do not treat 100 measured launches as 100 independent replicates.

### 5.7 Materiality language

The existing grouped/interleaved experiment retains its predeclared \(5\%\)
materiality threshold for \(R_T\).

For the parametric \(\delta\) sweep, use the following hierarchy:

1. root-work distribution and within-warp spread are the primary outcomes;
2. \(A(\delta)\) is the primary strategy comparison at a fixed \(\delta\);
3. \(P_K\) is a descriptive endpoint comparison;
4. no causal materiality decision is made from \(P_K\) alone.

This separation prevents a change in total arithmetic work from being
misidentified as a divergence effect.

### 5.8 Environmental invalidation and rerun policy

Invalidate an entire seed, not an individual unfavorable cell, if any of the
following occurs:

- the selected GPU UUID changes;
- another process materially uses the GPU;
- an NVIDIA Xid or ECC event occurs;
- the program exits nonzero;
- a correctness or path gate fails;
- a required artifact is missing;
- timestamps show that a measurement was not synchronized;
- thermal or clock throttling invalidates comparability.

After fixing the external or implementation problem, rerun all eight timing
cells for that seed. Record the invalid run and the reason; do not overwrite
it.

---

## 6. Work Package E: Parametric Endpoint GPU Utilization

### 6.1 Purpose

The existing utilization experiment already answers whether the completed
heterogeneous projection workloads kept the GPU active. The endpoint experiment
connects the reviewer's two clauses directly: similar scaling and GPU
utilization.

### 6.2 Cells

Collect sustained utilization for:

```text
delta      = 0, 1e-2
strategy   = thread-wise, warp-wise
seeds      = 2026:2035
duration   = 35 seconds per cell
```

This produces 40 utilization cells. The first 5 seconds form a predeclared
stabilization interval; the following 30 seconds form the publication window.

### 6.3 Long-lived process protocol

For each cell:

1. start one long-lived Julia process;
2. initialize CUDA and select the recorded H100 UUID;
3. load the immutable input;
4. compile and warm the target kernel;
5. print a machine-readable `READY` record with a monotonic timestamp;
6. wait for the wrapper's start signal;
7. print `START`;
8. repeatedly restore input, reset the root state, and launch the projection
   for 35 seconds;
9. record `FIRST_PROJECTION_START` and `LAST_PROJECTION_STOP`;
10. print `DONE`;
11. save the projection/restore/reset/host-gap time ledger.

The monitoring wrapper starts collection only after `READY`, signals the
application to begin, and stops collection immediately after `DONE`.

### 6.4 Monitoring fields

If `nvidia-smi` is used, collect and save at least:

```text
timestamp
uuid
utilization.gpu
utilization.memory
power.draw
clocks.current.sm
clocks.current.memory
temperature.gpu
```

Use:

```text
--format=csv,noheader,nounits
--loop-ms=1000
```

The publication text must call `utilization.gpu` **GPU utilization**. If a true
SM-active metric is collected with DCGM or Nsight Systems, report it separately
with its exact counter name.

### 6.5 Reported utilization statistics

For every seed and cell, report:

- aligned sample count;
- mean GPU utilization;
- median;
- P10;
- P90;
- peak;
- zero-utilization sample fraction;
- mean memory utilization;
- projection launches per second;
- projection-time fraction;
- restore/reset-time fraction;
- unexplained-gap fraction.

Aggregate the seed-level statistics across ten seeds without pooling all
one-second samples as independent replicates.

### 6.6 Interpretation

Use utilization as supporting evidence only:

> During the aligned repeated-projection window, report the numerical mean,
> median, P10, and P90 GPU utilization and then state whether these values show
> that the device remained highly active during the workload.

Do not write:

- high utilization proves that no divergence occurred;
- high utilization means all lanes performed useful work;
- equal utilization means equal kernel efficiency;
- utilization above 80% rules out a runtime penalty.

The reviewer's 80--90% range is a descriptive reference, not a filtering or
acceptance rule. Report the actual value even if it is below 80%.

---

## 7. Correctness and Acceptance Gates for the New Experiment

All ten seeds must pass every gate.

### 7.1 Input and path gates

- Exactly \(m=1{,}048{,}576\) cones are present.
- Every cone has dimension \(d=10\).
- All values are finite.
- All diagonal entries lie in \([10^{-3},10^3]\).
- The \(\delta=0\) cones are bitwise identical within each seed.
- Primitive perturbations are coupled across \(\delta\).
- Every cone takes the intended positive-root path.
- No cone is discarded based on its work count.

### 7.2 Solver gates

- Zero `MAX_ITER` events.
- Zero nonfinite outputs.
- Every root termination has a recognized production stopping reason.
- Final root residual and bracket-width conditions satisfy the production
  tolerances.
- CPU/GPU checks pass on at least 1,024 deterministically sampled cones per
  seed and \(\delta\).
- Thread-wise and warp-wise results agree within the stated mixed tolerance.

### 7.3 Timing gates

- Five warm-ups and ten valid measured launches are present in every timing
  cell.
- All eight timing cells are present for every seed.
- The exact counterbalanced order is saved.
- Timing comes from the uninstrumented production build.
- Restore/reset/generation are outside the timed projection interval.
- No invalid run is silently replaced or overwritten.

### 7.4 Utilization gates

- Forty endpoint utilization cells are present.
- At least 30 aligned one-second samples remain in each publication window.
- Zero-utilization samples are retained.
- The GPU UUID matches the timing experiment.
- Projection, restore/reset, gap, and wall times reconcile within 1%.
- The exact queried utilization field is reported.

### 7.5 Reproducibility gates

- All raw CSV files are retained.
- All source and environment hashes are retained.
- Every publication-table entry is regenerated by one analysis command.
- The analysis includes exactly ten seed-level observations.
- No old or superseded run enters the new inference.

---

## 8. Required Output Artifacts

Use a run directory of the form:

```text
benchmark/results/rebuttal/soc_parametric_similarity/
└── <UTC_RUN_ID>_soc_parametric_similarity/
    ├── manifest.json
    ├── environment.txt
    ├── source_snapshot/
    ├── generator/
    │   ├── seed_level_similarity.csv
    │   ├── rejection_counts.csv
    │   └── input_hashes.csv
    ├── diagnostic/
    │   ├── per_cone/
    │   │   └── seed_<seed>_delta_<delta>.csv.zst
    │   ├── warp_root_work.csv
    │   └── termination_summary.csv
    ├── timing/
    │   ├── launch_level.csv
    │   ├── seed_level.csv
    │   └── invalid_runs.csv
    ├── correctness/
    │   ├── cpu_gpu.csv
    │   ├── cross_strategy.csv
    │   └── residuals.csv
    ├── utilization/
    │   ├── raw/
    │   ├── aligned_seed_level.csv
    │   └── time_ledger.csv
    ├── analysis/
    │   ├── effect_estimates.csv
    │   ├── bootstrap_draws.csv
    │   ├── publication_tables.md
    │   └── publication_figures/
    └── run.log
```

The run manifest must contain:

- configuration values from Section 3.2;
- random-number generator and seed;
- all \(\delta\) values;
- the formula for \(t_i\);
- path-code definitions;
- correctness formulas and tolerances;
- diagnostic-counter definitions;
- timing order;
- bootstrap method;
- utilization query and timestamps.

---

## 9. Required Publication Tables

### 9.1 Table R1-3a: Completed adversarial-layout experiments

| Divergence source | Strategy | Grouped time | Interleaved time | Interleaved/grouped | 95% CI | Interpretation |
|:--|:--|--:|--:|--:|:--|:--|
| Same root branch, heterogeneous work | Thread-wise | 5.93 ms | 6.09 ms | 1.023 | [1.020, 1.026] | Small, below 5% margin |
| Same root branch, heterogeneous work | Warp-wise | 19.61 ms | 20.57 ms | 1.049 | [1.049, 1.050] | Descriptive ordering effect |
| Four top-level branches | Thread-wise | 3.64 ms | 4.20 ms | 1.154 | [1.152, 1.156] | Material 15.4% cost |
| Four top-level branches | Warp-wise | 10.79 ms | 19.03 ms | 1.765 | [1.764, 1.766] | Descriptive layout/control effect |

The warp-wise rows must not be labeled intra-warp divergence penalties.

### 9.2 Table R1-3b: Parametric-similar root-work distribution

Each symbolic entry below is replaced automatically by the analyzer:

| \(\delta\) | Expansion median/P90/max | Bisection median/P90/max | Oracle median/P90/max | Warp spread median/P90 | Modeled efficiency median/P10 |
|--:|:--|:--|:--|:--|:--|
| \(0\) | \(Q_{\mathrm{exp}}(0)\) | \(Q_{\mathrm{bis}}(0)\) | \(Q_{\mathrm{oracle}}(0)\) | \(S(0)\) | \(E(0)\) |
| \(10^{-4}\) | \(Q_{\mathrm{exp}}(10^{-4})\) | \(Q_{\mathrm{bis}}(10^{-4})\) | \(Q_{\mathrm{oracle}}(10^{-4})\) | \(S(10^{-4})\) | \(E(10^{-4})\) |
| \(10^{-3}\) | \(Q_{\mathrm{exp}}(10^{-3})\) | \(Q_{\mathrm{bis}}(10^{-3})\) | \(Q_{\mathrm{oracle}}(10^{-3})\) | \(S(10^{-3})\) | \(E(10^{-3})\) |
| \(10^{-2}\) | \(Q_{\mathrm{exp}}(10^{-2})\) | \(Q_{\mathrm{bis}}(10^{-2})\) | \(Q_{\mathrm{oracle}}(10^{-2})\) | \(S(10^{-2})\) | \(E(10^{-2})\) |

### 9.3 Table R1-3c: Parametric-similar timing

| \(\delta\) | Thread-wise time | Warp-wise time | \(A=T/W\) | 95% CI for \(A\) | Thread-wise speedup \(1/A\) |
|--:|--:|--:|--:|:--|--:|
| \(0\) | \(T_T(0)\) | \(T_W(0)\) | \(A(0)\) | \(\mathrm{CI}_A(0)\) | \(1/A(0)\) |
| \(10^{-4}\) | \(T_T(10^{-4})\) | \(T_W(10^{-4})\) | \(A(10^{-4})\) | \(\mathrm{CI}_A(10^{-4})\) | \(1/A(10^{-4})\) |
| \(10^{-3}\) | \(T_T(10^{-3})\) | \(T_W(10^{-3})\) | \(A(10^{-3})\) | \(\mathrm{CI}_A(10^{-3})\) | \(1/A(10^{-3})\) |
| \(10^{-2}\) | \(T_T(10^{-2})\) | \(T_W(10^{-2})\) | \(A(10^{-2})\) | \(\mathrm{CI}_A(10^{-2})\) | \(1/A(10^{-2})\) |

Cell times must state whether the displayed center is the mean or median of the
ten seed-level medians. Ratios must be geometric means of paired seed-level
ratios.

### 9.4 Table R1-3d: GPU utilization

Include both the existing heterogeneous cases and new parametric endpoints:

| Workload | Strategy | Mean | Median | P10 | P90 | Zero fraction | Launches/35 s |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Same-branch grouped | Thread-wise | 97.8% | 98.1% | 96.9% | 98.8% | \(Z_{\mathrm{root},T,G}\) | 5,775 |
| Same-branch interleaved | Thread-wise | 97.7% | 98.1% | 96.4% | 98.7% | \(Z_{\mathrm{root},T,I}\) | 5,768 |
| Same-branch grouped | Warp-wise | 98.1% | 98.1% | 98.1% | 98.2% | \(Z_{\mathrm{root},W,G}\) | 1,758 |
| Same-branch interleaved | Warp-wise | 98.2% | 98.3% | 98.2% | 98.3% | \(Z_{\mathrm{root},W,I}\) | 1,677 |
| Four-branch grouped | Thread-wise | 87.9% | 87.3% | 86.9% | 90.7% | \(Z_{\mathrm{branch},T,G}\) | 8,608 |
| Four-branch interleaved | Thread-wise | 90.6% | 90.4% | 88.2% | 92.5% | \(Z_{\mathrm{branch},T,I}\) | 7,750 |
| Four-branch grouped | Warp-wise | 96.3% | 96.2% | 96.0% | 97.1% | \(Z_{\mathrm{branch},W,G}\) | 3,147 |
| Four-branch interleaved | Warp-wise | 98.3% | 98.2% | 98.2% | 98.5% | \(Z_{\mathrm{branch},W,I}\) | 1,810 |
| Parametric \(\delta=0\) | Thread-wise | \(U_{\mathrm{mean},T,0}\) | \(U_{\mathrm{med},T,0}\) | \(U_{\mathrm{P10},T,0}\) | \(U_{\mathrm{P90},T,0}\) | \(Z_{T,0}\) | \(L_{T,0}\) |
| Parametric \(\delta=10^{-2}\) | Thread-wise | \(U_{\mathrm{mean},T,2}\) | \(U_{\mathrm{med},T,2}\) | \(U_{\mathrm{P10},T,2}\) | \(U_{\mathrm{P90},T,2}\) | \(Z_{T,2}\) | \(L_{T,2}\) |
| Parametric \(\delta=0\) | Warp-wise | \(U_{\mathrm{mean},W,0}\) | \(U_{\mathrm{med},W,0}\) | \(U_{\mathrm{P10},W,0}\) | \(U_{\mathrm{P90},W,0}\) | \(Z_{W,0}\) | \(L_{W,0}\) |
| Parametric \(\delta=10^{-2}\) | Warp-wise | \(U_{\mathrm{mean},W,2}\) | \(U_{\mathrm{med},W,2}\) | \(U_{\mathrm{P10},W,2}\) | \(U_{\mathrm{P90},W,2}\) | \(Z_{W,2}\) | \(L_{W,2}\) |

The table caption must give the exact monitoring field, one-second sampling
interval, 5-second stabilization interval, 30-second publication window, and
ten-seed aggregation method.

The \(Z\) entries are zero-sample fractions generated by the utilization
analyzer. The \(U\) and \(L\) entries are the new endpoint utilization
statistics and launch counts. They are symbolic output-column names, not
values to be chosen manually.

---

## 10. Optional Figure

One compact figure may be added if space permits:

- horizontal axis: \(\delta\) on a symmetric/log-like categorical scale;
- left panel: median and P90 within-warp oracle-count spread;
- middle panel: thread-wise and warp-wise seed-median projection time;
- right panel: \(A(\delta)=T_{\mathrm{thread}}/T_{\mathrm{warp}}\);
- points: ten seed values;
- line or marker: geometric mean for ratios;
- error bar: 95% seed-block bootstrap confidence interval.

The figure must not draw a continuous fitted curve through the four
\(\delta\) levels unless that model is declared and justified. A connected
categorical plot is sufficient.

---

## 11. Decision Rules for the Rebuttal

### 11.1 Similar parameters produce uniform work

Use this conclusion if smaller \(\delta\) produces visibly smaller oracle-count
spread and larger modeled efficiency:

> In a controlled parametric family, making both the cone vectors and diagonal
> scalings more similar reduced the observed variation in root-search work
> within thread-wise warps. This supports the reviewer's expectation that
> parametric similarity can make root-finding control flow more uniform.

### 11.2 Root work is uniform across the entire sweep

Use this conclusion if all four \(\delta\) levels have nearly identical counts:

> Across the tested parametric-similarity range, expansion and bisection counts
> were already highly concentrated, so increasing parameter similarity did not
> produce a material additional reduction in root-work variation.

This is not a failed experiment. It shows that the production root search is
stable for the tested parametric family.

### 11.3 Similarity does not reduce work variation

Use this conclusion if the counts remain heterogeneous or behave
nonmonotonically:

> Similar cone vectors and diagonal scalings did not by themselves guarantee
> identical root-search work in this controlled family. We therefore report
> the measured work distributions and avoid assuming that parametric
> similarity eliminates divergence.

### 11.4 Strategy comparison

At each fixed \(\delta\), state the measured thread-wise/warp-wise ratio and its
confidence interval. If thread-wise remains faster, report the observed
speedup without claiming universal dominance.

### 11.5 Existing adversarial cases

The rebuttal must preserve both facts:

1. same-root-branch work interleaving caused only a \(2.3\%\) thread-wise
   slowdown, below the \(5\%\) materiality margin;
2. top-level branch mixing caused a material \(15.4\%\) thread-wise slowdown,
   although thread-wise remained approximately \(4.5\times\) faster than
   warp-wise in that tested interleaved case.

### 11.6 Utilization

If the audited mean/median values are at least 80--90%, state that they meet or
exceed the range suggested by the reviewer. Immediately qualify that
device-level utilization is supplementary evidence and does not measure lane
efficiency directly.

---

## 12. Recommended Structure of the Final R1-3 Answer

The final answer should contain four paragraphs and three or four compact
tables.

### Paragraph 1: Acknowledge and explain the experiment

- thank the reviewer for identifying the concern;
- explain one-thread-per-cone versus one-warp-per-cone;
- state that divergence was examined through root-work variation, top-level
  branch mixing, a controlled parametric-similarity family, timing, and
  aligned GPU utilization.

### Paragraph 2: Report the deliberately heterogeneous cases

- report the \(2.3\%\) same-branch slowdown;
- explain that the variation was mainly expansion/oracle work, not bisection;
- report the \(15.4\%\) top-level branch slowdown;
- state that thread-wise remained approximately \(3.4\times\) and
  \(4.5\times\) faster in the corresponding interleaved cases.

### Paragraph 3: Report the parametric-similarity result

- describe the four \(\delta\) levels and ten seeds;
- report expansion, bisection, oracle, and within-warp spread;
- report thread-wise and warp-wise times and paired confidence intervals;
- choose one of the evidence-dependent conclusions in Section 11.

### Paragraph 4: Report utilization and delimit the claim

- report exact audited mean/median/P10/P90 utilization;
- identify the monitoring field as GPU utilization;
- state whether it meets the reviewer's suggested range;
- explain that high utilization complements, rather than replaces, the
  root-work and runtime measurements;
- limit the conclusion to type-22, \(d=10\), \(m=2^{20}\), FP64, and the tested
  H100.

---

## 13. Final Execution Checklist

### Existing evidence

- [ ] Freeze the two completed run directories.
- [ ] Verify source, environment, and GPU hashes.
- [ ] Recompute all existing seed-level timings and ratios.
- [ ] Define every correctness metric mathematically.
- [ ] Correct the repeatability and seed-count language.
- [ ] Audit the aligned-utilization samples and time ledger.
- [ ] Replace `SM busy` with the exact queried metric name.
- [ ] Remove unsupported Nsight and \(\Theta\) causal claims.

### Parametric generator

- [ ] Create the dedicated benchmark driver.
- [ ] Create the deterministic generator with coupled perturbations.
- [ ] Implement \(\delta=\{0,10^{-4},10^{-3},10^{-2}\}\).
- [ ] Run the seed-2025 path-code pilot.
- [ ] Pass all generator tests.
- [ ] Freeze the generator before running seeds 2026--2035.

### Diagnostics

- [ ] Record per-cone path and root-work counters.
- [ ] Separate expansion, bisection, and total oracle work.
- [ ] Compute natural-warp spread and modeled efficiency.
- [ ] Verify 100% intended-path membership.
- [ ] Verify zero `MAX_ITER` and zero nonfinite results.
- [ ] Verify diagnostic-build output invariance.

### Timing

- [ ] Run 80 timing cells.
- [ ] Use five warm-ups and ten measured launches per cell.
- [ ] Save every launch and the counterbalanced order.
- [ ] Compute seed medians and paired ratios.
- [ ] Generate 10,000 seed-block bootstrap confidence intervals.
- [ ] Retain invalid runs and documented rerun reasons.

### Utilization

- [ ] Run 40 endpoint utilization cells.
- [ ] Retain 30 aligned one-second publication samples per cell.
- [ ] Save all raw samples, including zeros.
- [ ] Reconcile the time ledger within 1%.
- [ ] Report mean, median, P10, P90, peak, and zero fraction.

### Publication

- [ ] Generate Tables R1-3a through R1-3d from saved CSV files.
- [ ] Select the conclusion branch from Section 11 using measured results.
- [ ] State the \(15.4\%\) branch-mixing penalty explicitly.
- [ ] State that high GPU utilization is supplementary evidence.
- [ ] Limit generalization to the tested cone, dimension, count, precision,
      GPU, and software environment.
- [ ] Insert the final response into `response_to_reviewers_PDCS.tex`.

---

## 14. Completion Criterion

R1-3 is fully answered when:

1. the existing results have passed the artifact and terminology audit;
2. the controlled parametric-similarity experiment has ten passing seed-level
   summaries at all four \(\delta\) values;
3. per-cone root-work, within-warp work variation, production timing, and
   endpoint utilization are all reported;
4. the response acknowledges both the small same-branch penalty and the
   material top-level branch penalty;
5. every table is reproducible from a retained seed-level artifact; and
6. no conclusion relies on \(m=100\), \(m=1{,}000\), a causal \(\Theta\)
   interpretation, or an unmeasured hardware active-lane metric.
