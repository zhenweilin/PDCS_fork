# R1-3 Warp-Divergence Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans`
> to implement this plan task by task. Track execution with the checkboxes in
> Section 14.

**Goal:** Determine whether deliberately induced intra-warp root-finding
divergence makes the thread-wise diagonally rescaled SOC projection
materially slower than the warp-wise projection.

**Architecture:** Use a fully paired \(2\times2\) design. For each independent
workload seed, generate one fixed cone multiset, derive low-divergence
(`grouped`) and high-divergence (`interleaved`) permutations of that exact
multiset, and run both thread-wise and warp-wise kernels on both layouts. The
primary experiment varies measured same-branch root-search work while holding
the top-level projection path fixed; a secondary experiment varies top-level
branches.

**Tech stack:** Julia 1.10.4, CUDA C++, CUDA.jl, CUDA Toolkit 12.5, NVIDIA H100
80 GB HBM3, Nsight Compute, `nvidia-smi dmon`, CSV, and `Float64` arithmetic.

## Global constraints

- Primary projection: diagonally rescaled second-order cone, projection type
  `22`.
- Primary size: \(m=1{,}048{,}576=2^{20}\) cones of full dimension \(d=10\).
- This size is deliberately in the regime where thread-wise projection is a
  plausible strategy and supplies many resident warps on the H100. Very small
  fixed counts such as \(m=100\) or \(m=1{,}000\) primarily test launch and
  occupancy limits, so they belong to the separate R1-2 strategy map rather
  than the causal R1-3 divergence test.
- Primary strategies: thread-wise (`massive`) and warp-wise (`sufficient`).
  Block-wise is excluded because it does not directly answer the reviewer's
  thread-versus-warp question.
- Force the production `massive_block_proj` and `sufficient_block_proj`
  kernels rather than allowing an automatic strategy selector. Record the
  actual launch geometry; the expected production configuration is 256
  threads per CUDA block.
- Primary timing mode: cold start, with every stored scalar root reset to zero
  before each measured projection.
- Root-search tolerances: `abs_tol = rel_tol = 1e-12`.
- Every strategy and layout for one seed must receive the same cone tuples,
  tolerances, and initial root state.
- Diagnostic instrumentation must be compiled separately and must never be
  enabled during publication timing or utilization collection.
- The current production sources use different safety caps:
  `MAX_ITER = 10_000` in `massive_block_proj.cu` and `MAX_ITER = 100_000` in
  `sufficient_block_proj.cu`. Do not use cap hits to manufacture a hard
  workload. Require zero cap hits and a valid common production stopping
  condition in both kernels, and report the freshly recomputed final residual.
- Input generation, sorting, permutation, allocation, root-state reset, and
  device-to-device restoration of the input are outside the timed projection
  interval.
- The kernels modify both the projected array and scalar root state in place.
  Restore the array from an immutable device template and reset the root state
  before every warm-up, measured launch, profiler launch, and utilization-loop
  launch.
- Do not change per-cone tolerances, insert artificial delays, or tune a
  generator after observing runtime differences.
- Nsight replay time must never be reported as kernel runtime.
- Raw results, environment metadata, and exact case manifests must be retained.
- A failed correctness or manipulation check invalidates the corresponding
  timing result.

---

## 1. Reviewer question and experimental estimands

The reviewer is concerned that different threads may take different
root-finding branches or require different numbers of bisection iterations.
The practical question is:

> Under a deliberately high-divergence workload, is thread-wise projection
> materially slower than warp-wise projection?

The causal question is:

> Does increasing within-warp divergence hurt thread-wise projection more than
> it hurts warp-wise projection, after controlling for the cone multiset and
> its total mathematical work?

The GPU mappings make warp-wise projection the natural control:

- **Thread-wise:** one CUDA thread projects one cone. Thirty-two independent
  cones occupy one hardware warp, so cone-to-cone branch or iteration
  differences can cause intra-warp divergence.
- **Warp-wise:** one hardware warp projects one cone. All lanes cooperate on
  the same scalar root search, so cone-to-cone iteration differences occur
  across warps rather than among lanes of one warp.

For kernel \(K\in\{T,W\}\) and layout \(L\in\{G,I\}\), let
\(T_{s,K,L}\) be the seed-level median synchronized projection time for
workload seed \(s\). Here:

- \(T\): thread-wise;
- \(W\): warp-wise;
- \(G\): grouped, the low-divergence layout;
- \(I\): interleaved, the high-divergence layout.

The \(2\times2\) design is:

| Kernel | Grouped: low divergence | Interleaved: high divergence |
|:---|---:|---:|
| Thread-wise | \(T_{s,T,G}\) | \(T_{s,T,I}\) |
| Warp-wise | \(T_{s,W,G}\) | \(T_{s,W,I}\) |

Two primary estimands are required.

### 1.1 Absolute high-divergence comparison

\[
A_{I,s}=\frac{T_{s,T,I}}{T_{s,W,I}},
\qquad
A_{G,s}=\frac{T_{s,T,G}}{T_{s,W,G}}.
\]

The primary absolute estimand is \(A_s\equiv A_{I,s}\);
\(A_{G,s}\) is its low-divergence reference.

- \(A_s>1\): thread-wise is slower than warp-wise under high divergence.
- \(A_s<1\): thread-wise remains faster under high divergence.

### 1.2 Divergence-susceptibility interaction

\[
R_{T,s}=\frac{T_{s,T,I}}{T_{s,T,G}},\qquad
R_{W,s}=\frac{T_{s,W,I}}{T_{s,W,G}},
\]

\[
\Theta_s
=\frac{R_{T,s}}{R_{W,s}}
=\frac{T_{s,T,I}/T_{s,T,G}}
       {T_{s,W,I}/T_{s,W,G}}.
\]

- \(\Theta_s>1\): interleaving hurts thread-wise more than warp-wise.
- \(\Theta_s\approx1\): no thread-wise-specific relative degradation is
  resolved.
- \(\Theta_s<1\): the data do not show an additional thread-wise penalty.

Both \(A\) and \(\Theta\) must be reported. It is possible for divergence to
cause a small relative penalty, \(\Theta>1\), while thread-wise remains faster
in absolute time, \(A<1\).

\(R_W\) is an ordering and scheduling control, not a warp-wise divergence
penalty: the warp-wise kernel assigns one whole warp to one cone, so cones
with different iteration counts do not create cross-cone lane divergence
inside that warp.

---

## 2. Files and artifacts

Repository root: `/Users/zhenweilin/fork/PDCS`.

### 2.1 Source files

- Create:
  `benchmark/rescaled_soc_divergence_2x2.jl`
  — case generation, exact permutations, correctness checks, timing, and
  profiling entry points.
- Create:
  `benchmark/analyze_soc_divergence_2x2.jl`
  — seed-level aggregation, confidence intervals, manipulation checks, and
  publication tables.
- Modify under a compile-time diagnostic guard:
  `src/pdcs_gpu/cuda/massive_block_proj.cu`
  — per-cone branch and root-search counters for the thread-wise kernel.
- Modify under the same guard:
  `src/pdcs_gpu/cuda/sufficient_block_proj.cu`
  — corresponding counters for source-level validation of the warp-wise
  kernel.
- Update after execution:
  `doc/pdcs_overleaf/R1-2_r1-3.md`
  — reviewer response populated only with validated results.

### 2.2 Run-directory schema

Write every run under:

```text
benchmark/results/rebuttal/soc_divergence_2x2/<UTC-run-id>/
```

Each run directory must contain:

```text
environment.txt
case_manifest.csv
cone_ids.csv.zst
permutations.csv.zst
timings_raw.csv
timings_seed_summary.csv
root_work_raw.csv.zst
root_work_summary.csv
manipulation_checks.csv
correctness.csv
effect_estimates.csv
ncu/*.ncu-rep
ncu/*.csv
ncu/available_metrics.txt
nvidia_smi/*.txt
utilization_summary.csv
publication_tables.md
```

`environment.txt` must record:

- Git commit SHA and dirty-worktree status;
- GPU UUID, model, and physical index;
- Julia, CUDA.jl, CUDA Toolkit, driver, and Nsight Compute versions;
- GPU clocks, power state, power limit, temperature, and compute mode;
- compiler commands and diagnostic macro state;
- all seeds, tolerances, dimensions, cone counts, and generator grids.

---

## 3. Primary experiment: same-branch root-work divergence

This experiment isolates loop-exit divergence while keeping every cone on one
root-search path. The preferred candidate family uses the positive-\(t\)
branch and targets variation in bisection-loop counts. The production
positive-\(t\) cold path, however, always starts from the same bracket
\([0,0.5]\) and uses one global tolerance. Therefore, changing the cone data
does not by itself guarantee different bisection counts. The diagnostic
manipulation gate below decides whether a genuine iteration contrast exists
before any publication timing is inspected.

A same-negative-root candidate family is predeclared as a contingency because
that production path includes data-dependent interval expansion. The chosen
family is fixed using diagnostic counts from a pilot seed, not runtime, and
then used for all inferential seeds. Thus, in either family, the top-level
path is constant and the tested contrast is measured root-search work rather
than a mixture of feasible, polar, positive-root, and negative-root branches.

### 3.1 Cone notation

For cone \(i\), write

\[
x_i=(t_i,u_i),\qquad u_i\in\mathbb R^q,\qquad q=d-1=9,
\]

and

\[
\widehat D_i
=\operatorname{diag}(1,d_{i1},\ldots,d_{iq}).
\]

The first diagonal coordinate is fixed to one.

### 3.2 Tail-direction generator

For each candidate cone, draw

\[
z_i\sim\mathcal N(0,I_q)
\]

from a seed- and cone-index-addressable random stream and set

\[
u_i=\frac{z_i}{\|z_i\|_2}.
\]

If \(\|z_i\|_2=0\), discard and redraw that cone using the next recorded
counter value in the same random stream.

### 3.3 Diagonal-scaling generator

For a prescribed log-scale standard deviation \(s\), draw

\[
\eta_i\sim\mathcal N(0,I_q)
\]

and define

\[
\ell_{ij}
=s\eta_{ij}-\frac{s}{q}\sum_{k=1}^q\eta_{ik},
\]

\[
d_{ij}=\exp(\ell_{ij}).
\]

Accept the draw only if every \(d_{ij}\in[10^{-3},10^3]\); otherwise redraw
the complete vector \(\eta_i\) from the next recorded counter in the same
stream. This bound-preserving rejection rule retains

\[
\prod_{j=1}^q d_{ij}=1,
\]

so changing \(s\) changes within-cone conditioning rather than merely
multiplying the full diagonal by one scalar. Clipping individual entries is
not allowed because it would destroy the centered-log invariant.

Define

\[
a_i=\|\operatorname{diag}(d_i)u_i\|_2.
\]

For a boundary ratio \(r\in(0,1)\), set

\[
t_i=r\,a_i.
\]

Then \(t_i>0\) and \(t_i<a_i\), so the cone is neither feasible nor polar and
must take the positive-\(t\) root-search branch.

### 3.4 Candidate grid and pool

For each independent workload seed, generate a candidate pool of

\[
4m=4{,}194{,}304
\]

cones, distributed as evenly as possible over

\[
r\in\{0.02,0.10,0.20,0.50,0.80,0.95\},
\]

\[
s\in\{0,0.25,0.50,1.00,1.50,2.00\}.
\]

Use deterministic round-robin assignment when \(4m\) is not divisible by the
36 grid cells. Record the \((r,s)\) cell and immutable cone ID for every
candidate.

### 3.5 Predeclared negative-root contingency

For the same generated \((u_i,d_i)\), define

\[
b_i=\|\operatorname{diag}(d_i)^{-1}u_i\|_2
\]

and set

\[
t_i=-r\,b_i,\qquad 0<r<1.
\]

Because \(b_i>-t_i=r b_i\), the point is not in the polar region; because
\(t_i<0\), it is not feasible. Every accepted cone must therefore enter the
same negative-\(t\) root-search path. That path starts from
\([\xi_L,\xi_R]=[0.5,1]\) and repeatedly doubles \(\xi_R\) while the root is
not bracketed, so cone data can change both interval-expansion work and the
subsequent bisection work.

The contingency uses the same initial and expanded \((r,s)\) grids and the
same candidate-pool size as the positive family. It is not a second
opportunity to search for a favorable timing result: all candidate-family
selection is completed from diagnostic counters before uninstrumented timing.

### 3.6 Diagnostic pass

Run a separately compiled diagnostic kernel once on the candidate pool. For
each cone, record:

- top-level path;
- interval-expansion iterations;
- warm-state tests and attempted and accepted Newton steps;
- bisection-loop body count \(k_i\);
- total root-oracle evaluations;
- termination reason: residual, bracket width, or iteration cap;
- whether the strategy-specific `MAX_ITER` was reached;
- final bracket endpoints and normalized bracket width;
- a freshly recomputed final scalar residual;
- output finiteness.

Record the normalized final bracket width using the production expression

\[
w_i^{\mathrm{bracket}}
=\frac{\xi_{R,i}-\xi_{L,i}}
       {1+\xi_{R,i}+\xi_{L,i}}.
\]

Use per-cone local counters rather than atomics in the root loops. In the
thread-wise diagnostic kernel, the cone-owning thread writes one record. In
the warp-wise diagnostic kernel, lane 0 writes one record after warp
agreement. Never time or profile this diagnostic build.

Use the thread-wise diagnostic counts to define \(h_i\) and the difficulty
strata, because the intervention targets lane-to-lane work in that kernel.
Use the warp-wise diagnostic pass to verify its path, termination, residual,
and work decomposition independently; do not average the two kernels'
counters into the ordering key.

The diagnostic pass must not supply publication timing.

Retain only cones satisfying all of the following:

1. the recorded path matches the candidate family being tested
   (positive-\(t\) or negative-\(t\) root search);
2. `MAX_ITER` is false;
3. the output and residual are finite;
4. either the scalar residual meets `abs_tol` or the normalized final bracket
   width meets `rel_tol`, matching the production stopping rule;
5. the sampled CPU/GPU projection check in Section 7 passes.

### 3.7 Pilot family selection and difficulty quartiles

Reserve seed 2001 as a diagnostic-only pilot; it is not one of the 20
inferential seeds. Apply the following decision rule without examining any
uninstrumented timing:

1. Test the positive-root family on the initial grid.
2. If its gate fails, test the predeclared expanded positive grid.
3. If that still fails, test the negative-root family on the initial grid.
4. If necessary, test the expanded negative-root grid.
5. Freeze the first family and grid that pass all separation and
   manipulation criteria below. If none passes, stop and report that the
   intended same-branch root-work contrast could not be established.

For the selected family, define the primary per-cone work count \(h_i\) by

\[
h_i=
\begin{cases}
k_i, & \text{positive-root family},\\
o_i, & \text{negative-root family},
\end{cases}
\]

where \(o_i\) is the total number of root-oracle evaluations, including
interval expansion and bisection. Always retain and report
\((e_i,k_i,o_i)\), where \(e_i\) is the interval-expansion count; \(h_i\) is
only the preregistered ordering variable.

Sort retained cones by the family-specific lexicographic key

```text
positive: (bisection_iterations, total_oracle_evaluations, immutable_cone_id)
negative: (total_oracle_evaluations, expansion_iterations,
           bisection_iterations, immutable_cone_id)
```

Let \(N_{\mathrm{ret}}\) be the retained count and define

\[
N=4\left\lfloor\frac{N_{\mathrm{ret}}}{4}\right\rfloor.
\]

Require \(N\ge m\). If needed, discard only the final
\(N_{\mathrm{ret}}-N\le3\) entries to make the count divisible by four, and
partition the first \(N\) sorted entries into four contiguous groups of
exactly \(N/4\) ranks:

\[
Q_1,\ Q_2,\ Q_3,\ Q_4.
\]

\(Q_1\) contains the easiest cones and \(Q_4\) the hardest. Let
\(M=N/4\) and \(n_q=m/4=262{,}144\). From each quartile select the
deterministic, evenly spaced ranks

\[
j_a
=\left\lfloor\frac{(a+1/2)M}{n_q}\right\rfloor,
\qquad a=0,\ldots,n_q-1.
\]

This selects exactly \(m/4\) cones across the full rank span of each quartile
rather than taking only one edge of it. If \(N<m\), that candidate-family
stage fails before layout construction.

Before constructing layouts, require

\[
\operatorname{median}(h_i\mid Q_4)
-\operatorname{median}(h_i\mid Q_1)\ge4.
\]

For the expanded-grid stages in the decision rule, discard the pilot
candidate pool and regenerate it using

\[
r\in\{0.005,0.01,0.02,0.10,0.50,0.90,0.99\},
\]

\[
s\in\{0,0.50,1.00,1.50,2.00,2.50,3.00\}.
\]

After the pilot freezes the family and grid, each inferential seed must
independently pass the same separation gate and the Section 3.9 manipulation
gate. A failed seed is reported and invalidates the confirmatory 10-seed
experiment; it is not silently replaced. Do not continue changing generators
after inspecting timing results.

### 3.8 Exact grouped and interleaved layouts

Let \(q_{c,j}\) be the \(j\)-th selected cone in quartile
\(c\in\{0,1,2,3\}\). All indices below are zero based.

#### Grouped layout

For

\[
s=0,\ldots,\frac{m}{128}-1,\quad
c=0,\ldots,3,\quad
\ell=0,\ldots,31,
\]

define

\[
G_{32(4s+c)+\ell}=q_{c,32s+\ell}.
\]

Every consecutive group of 32 cones, corresponding to one thread-wise
hardware warp, belongs to one difficulty quartile.

#### Interleaved layout

For

\[
w=0,\ldots,\frac{m}{32}-1,\quad
r'=0,\ldots,7,\quad
c=0,\ldots,3,
\]

define

\[
I_{32w+4r'+c}=q_{c,8w+r'}.
\]

Every thread-wise hardware warp contains eight cones from each difficulty
quartile.

Apply each permutation identically to:

- the input vector;
- the diagonal vector and its squared or inverse-derived payloads;
- the immutable cone ID;
- the zero-valued cold-start state;
- diagnostic metadata.

Rebuild monotone `head_start` offsets for the new payload order; do not
permute the numerical offset values themselves. Hash the cone tuples before
and after layout construction in addition to checking IDs.

Save both permutations and their inverses. Verify

\[
\operatorname{sort}(\mathrm{ID}_G)
=\operatorname{sort}(\mathrm{ID}_I)
\]

and require the inverse permutation to recover every source array bit for bit.

Each 128-cone microtile contains 32 cones from every quartile in both layouts.
With the production 256-thread block, two complete microtiles occupy one
thread-wise block, so every block contains 64 cones from each quartile in both
layouts. The intended intervention is therefore the distribution of equal
work strata among lanes within a warp, not the total stratum composition of a
block.

### 3.9 Manipulation check

For thread-wise warp \(w\), define the primary-work spread

\[
S_w
=\max_{\ell=1,\ldots,32}h_{w\ell}
 -\min_{\ell=1,\ldots,32}h_{w\ell},
\]

and modeled active-lane efficiency

\[
E_w
=\frac{\sum_{\ell=1}^{32}h_{w\ell}}
       {32\max_{\ell=1,\ldots,32}h_{w\ell}}.
\]

Require \(\max_\ell h_{w\ell}>0\). If every lane executes the same measured
root work, \(E_w=1\); early lane completion lowers \(E_w\). This is a
workload-derived model, not a hardware utilization counter. For the positive
family it is explicitly a bisection-only model. For the negative family it is
an oracle-evaluation model; because expansion and bisection evaluations need
not have identical instruction costs, report separate expansion and
bisection distributions as well.

The manipulation passes only if all of the following hold:

\[
\operatorname{median}(S_w^I)
>\operatorname{median}(S_w^G),
\]

\[
\operatorname{median}(E_w^G)\ge0.90,
\]

\[
\operatorname{median}(E_w^I)
\le\operatorname{median}(E_w^G)-0.10.
\]

If any condition fails, the timing comparison cannot support a conclusion
about same-branch root-work divergence.

---

## 4. Secondary experiment: top-level branch divergence

This experiment isolates divergence caused by different top-level
`if`/`else` paths. It is secondary because the reviewer's root-iteration
concern is addressed more directly by Section 3.

### 4.1 Common cone data

Generate one reproducible multiset of \(m\) tail vectors and diagonals using
the generator in Sections 3.2--3.3 with \(s=1\). Define

\[
a_i=\|\operatorname{diag}(d_i)u_i\|_2,\qquad
b_i=\|\operatorname{diag}(d_i)^{-1}u_i\|_2.
\]

Assign exactly \(m/4\) cones to each class:

\[
t_i=
\begin{cases}
  1.25a_i, & \text{feasible},\\
 -1.25b_i, & \text{polar/zero},\\
  0.20a_i, & \text{positive-\(t\) root search},\\
 -0.20b_i, & \text{negative-\(t\) root search}.
\end{cases}
\]

The diagnostic branch code, not the construction label alone, must confirm
each class before timing.

### 4.2 Branch-grouped layout

Use the grouped formula from Section 3.8 with \(c\) denoting branch class.
Each thread-wise warp contains one top-level path.

### 4.3 Branch-interleaved layout

Use the interleaved formula from Section 3.8 with \(c\) denoting branch class.
Each thread-wise warp contains eight cones from every top-level path.

The two layouts must be exact permutations of the same cone tuples. Their
timing and effect estimates use the same \(A\), \(R_T\), \(R_W\), and
\(\Theta\) definitions as the primary experiment.

### 4.4 Branch-divergence manipulation check

Let \(p_{w\ell}\) be the diagnostic path code of lane \(\ell\) in a
thread-wise warp. For each warp, report the number of distinct path codes and
the modal-path fraction. Require:

- grouped: one distinct code and modal fraction \(1\) in every complete warp;
- interleaved: four distinct codes and modal fraction \(8/32=0.25\) in every
  complete warp;
- identical global counts of all four path codes in both layouts.

If the diagnostic path differs from the construction label for any cone, the
branch-divergence experiment fails before timing.

---

## 5. Applied validation: genuinely parametric-similar cones

This experiment addresses the reviewer's observation that cones in a
parametric sequence may have similar scaling. It is a descriptive realism
check, not the primary causal divergence test.

For each workload seed, generate and save one base pair
\((u_\star,\ell_\star)\) with

\[
u_\star=\frac{z_\star}{\|z_\star\|_2},
\qquad z_\star\sim\mathcal N(0,I_q),
\]

\[
\ell_\star
=\eta_\star-\frac{\mathbf1^\top\eta_\star}{q}\mathbf1,
\qquad \eta_\star\sim\mathcal N(0,I_q).
\]

Accept the base draw only if every
\(\exp(\ell_{\star j})\in[10^{-3},10^3]\); otherwise redraw
\(\eta_\star\). This prevents the perturbation-level rejection rule below
from becoming impossible at \(\delta_D=0\).

For each cone, draw

\[
\xi_i,\zeta_i\sim\mathcal N(0,I_q)
\]

from independent recorded streams and define

\[
\widetilde u_i
=u_\star
+\delta_u\|u_\star\|_2
   \frac{\xi_i}{\|\xi_i\|_2},
\]

\[
u_i=\frac{\widetilde u_i}{\|\widetilde u_i\|_2},
\]

\[
\widetilde\ell_i
=\ell_\star+\delta_D\zeta_i,
\]

\[
\ell_i
=\widetilde\ell_i
-\frac{\mathbf1^\top\widetilde\ell_i}{q}\mathbf1,
\qquad
d_{ij}=e^{\ell_{ij}}.
\]

Reject and redraw \(\zeta_i\) if any
\(d_{ij}\notin[10^{-3},10^3]\), preserving the centered-log invariant. Reject
and redraw \(\xi_i\) if either direction norm in the construction is zero.

Use

\[
\delta_u=\delta_D
\in\{0,10^{-4},10^{-3},10^{-2}\}
\]

and

\[
t_i=0.20\|\operatorname{diag}(d_i)u_i\|_2.
\]

For each perturbation level, report:

- thread-wise and warp-wise cold-start projection time;
- bisection-count median, 90th percentile, maximum, and within-warp spread;
- \(A=T_T/T_W\);
- sustained GPU activity and kernel-level throughput.

Do not use comparisons between different perturbation levels as the primary
causal estimate of divergence, because they are different mathematical
workloads. Warm-start experiments are outside the primary R1-3 design and
must be reported separately if retained.

---

## 6. Replication, counterbalancing, and timing

### 6.1 Independent workload seeds

Use 10 independent seeds:

```text
2026, 2027, ..., 2035
```

The workload seed is the statistical unit. Cones, warps, repeated launches,
and Nsight replay passes are not independent observations.

### 6.2 Technical replicates

For every seed and \(2\times2\) cell:

1. allocate and prepare all buffers;
2. copy each layout once to an immutable device template;
3. before every warm-up, restore the mutable input from that template and
   reset all scalar roots to zero;
4. perform five such unmeasured warm-up launches;
5. collect ten synchronized CUDA-event timings in ten counterbalanced rounds;
6. before every measured launch, again restore the input and zero the root
   state in the same CUDA stream, then record the start event only after those
   operations have completed in stream order;
7. define \(T_{s,K,L}\) as the median of those ten launches.

Ten launches are technical replicates used to stabilize the seed-level
estimate; inferential sample size remains 20.

### 6.3 Counterbalanced execution order

Use the following four-cell Williams schedule, in which every cell precedes
every other cell exactly once across the four rows:

```text
Order 0: TG, TI, WI, WG
Order 1: TI, WG, TG, WI
Order 2: WG, WI, TI, TG
Order 3: WI, TG, WG, TI
```

For seed index \(j=0,\ldots,19\) and timing round
\(r=0,\ldots,9\), use `Order ((j + r) mod 4)`. Each round contributes one
timing to each cell. This counterbalances all four cells rather than always
timing one strategy or one layout first.

Within each difficulty quartile or branch class, apply a seed-addressable
Fisher--Yates shuffle before the exact grouped/interleaved placement. Use the
same shuffled class lists for both layouts and both kernels.

### 6.4 Timed interval

Use CUDA events around exactly one projection kernel launch:

```text
restore_input_from_immutable_device_template()
zero_root_state()
record(start)
launch_projection()
record(stop)
synchronize(stop)
elapsed_time(start, stop)
```

Exclude:

- input generation;
- diagnostic instrumentation;
- sorting and quartile selection;
- allocation;
- input restoration;
- root-state reset;
- Host--Device and Device--Host copies;
- correctness checks.

The projection mutates its input, so every measured launch must restore the
same immutable source data before timing. Use the same CUDA stream for the
restoration, reset, events, and projection. Confirm with a stream-level test
that the start event cannot overtake the restoration. Audit every kernel
argument once: any workspace value that is read before being overwritten must
also be initialized to the same state before each launch.

### 6.5 Runtime environment

Before and after each seed block:

- verify no unrelated process is using the target GPU;
- record GPU temperature, clocks, P-state, power draw, and memory use;
- record the GPU UUID from both Julia and `nvidia-smi`;
- abort and repeat the entire seed block if the device reports an error or if
  a correctness check fails.

Do not discard a seed merely because its timing is unfavorable.

Warm-start interactions are deliberately excluded. A source audit found a
potential semantic difference in an exact warm-root recovery path between the
two kernels, so a warm-start comparison requires a separate correctness study
before it can be interpreted as a performance experiment.

---

## 7. Correctness and invariance checks

Every seed must pass all checks below before its timing enters analysis.

### 7.1 Projection correctness

- Sample at least 1,024 cones from every generated experiment.
- Compare GPU output with the CPU diagonally rescaled SOC projection.
- Use a mixed absolute-relative componentwise test:

\[
\max_j
\frac{|x_{\mathrm{GPU},j}-x_{\mathrm{CPU},j}|}
     {5\times10^{-8}
      +5\times10^{-8}
       \max(|x_{\mathrm{GPU},j}|,|x_{\mathrm{CPU},j}|)}
\le1.
\]

- Check finiteness, primal-cone feasibility, polar-residual feasibility,
  complementarity/Moreau decomposition, the projection KKT residual, and
  absence of `MAX_ITER`.

### 7.2 Branch correctness

- Every Section 3 cone must record the positive-\(t\) root-search path.
- If the predeclared negative-root contingency is selected, replace the
  preceding condition by requiring the negative-\(t\) root-search path for
  every Section 3 cone.
- Every Section 4 cone must record the path assigned by its class.
- Branch counts must be identical before and after permutation.

### 7.3 Permutation invariance

- Grouped and interleaved ID multisets must be identical.
- Cone-tuple hashes must match between grouped and interleaved layouts.
- Applying the saved inverse permutation must recover the input, diagonal,
  root-state, and metadata arrays bit for bit.
- After inverse permutation, grouped and interleaved projected values must
  pass the mixed absolute-relative test in Section 7.1.

### 7.4 Instrumentation invariance

For a fixed sample, diagnostic and uninstrumented builds must produce outputs
that pass the mixed absolute-relative test in Section 7.1. Diagnostic counters
must not be enabled for timing, Nsight Compute, or sustained-utilization runs.

### 7.5 Strategy agreement and termination

- After inverse permutation, compare thread-wise and warp-wise outputs using
  the Section 7.1 mixed tolerance.
- Require the recorded termination to satisfy the common production rule in
  both implementations: final residual at most `abs_tol` or normalized
  bracket width at most `rel_tol`. Report both quantities regardless of which
  condition stopped the loop.
- Require zero iteration-cap hits despite the two source files' different
  safety-cap constants.
- Do not retain a cone merely because both kernels return the same incorrect
  result; the independent CPU and Moreau checks remain mandatory.

---

## 8. Statistical analysis

### 8.1 Seed-level ratios

For each divergence source and seed, compute:

\[
A_s=\frac{T_{s,T,I}}{T_{s,W,I}},
\quad
R_{T,s}=\frac{T_{s,T,I}}{T_{s,T,G}},
\quad
R_{W,s}=\frac{T_{s,W,I}}{T_{s,W,G}},
\quad
\Theta_s=\frac{R_{T,s}}{R_{W,s}}.
\]

Estimate ratios on the log scale:

\[
\widehat A
=\exp\left(\frac1{20}\sum_{s=1}^{20}\log A_s\right),
\]

\[
\widehat\Theta
=\exp\left(\frac1{20}\sum_{s=1}^{20}\log\Theta_s\right).
\]

Report the corresponding percentage effects:

\[
100(\widehat A-1)\%,\qquad
100(\widehat\Theta-1)\%.
\]

### 8.2 Confidence intervals

Use two seed-blocked methods:

1. a paired \(t\)-interval on seed-level log ratios;
2. a 10,000-resample seed-level bootstrap sensitivity analysis.

Every bootstrap resample must carry all four \(2\times2\) cells for the
selected seed. Never resample individual cones, warps, launches, or profiler
replay passes as if they were independent workloads.

Treat the same-branch root-work interaction and absolute high-divergence ratio
as the two prespecified primary quantities answering R1-3. The branch-mixing
and parametric-similar studies are secondary and descriptive; do not promote
them to confirmatory evidence if the primary manipulation fails.

### 8.3 Predeclared materiality margin

Use a one-sided 5% material penalty margin.

For the absolute high-divergence ratio \(A\):

- if the upper one-sided 95% confidence bound is below \(1.05\), rule out
  thread-wise being at least 5% slower than warp-wise under the tested
  high-divergence workload;
- if the lower one-sided 95% bound is above \(1.05\), conclude that thread-wise
  has a material absolute slowdown;
- otherwise report the result as inconclusive.

For the interaction \(\Theta\):

- if the upper one-sided 95% confidence bound is below \(1.05\), rule out at
  least 5% extra susceptibility of thread-wise to interleaving;
- if the lower one-sided 95% bound is above \(1.05\), conclude that
  interleaving imposes a material thread-wise-specific penalty;
- otherwise report the result as inconclusive.

Do not equate a nonsignificant \(p\)-value with evidence of no penalty.

### 8.4 Readability summaries

For each cell, report mean, median, and sample standard deviation across the 20
seed-level medians. Ratio estimates and their confidence intervals, rather
than overlap of per-launch standard deviations, determine the inferential
conclusion.

---

## 9. Nsight Compute characterization

Collect Nsight Compute metrics for seeds 2026, 2027, and 2028 after
uninstrumented timing is complete. Profile one warmed target kernel per
process and keep profiling runs separate from timing inference. Compile the
optimized, uninstrumented PTX with `-lineinfo`, and place exactly one target
projection launch inside the NVTX push/pop range `PDCS_PROJECTION`. The
`--profile-one` mode must perform its restored-state warm-ups outside that
range and issue exactly one restored-state launch inside it.

Use the following sections when available:

```text
SpeedOfLight
Occupancy
SchedulerStats
WarpStateStats
SourceCounters
```

Record:

| Quantity | Preferred metric or section |
|:---|:---|
| Branch-target uniformity | `smsp__sass_average_branch_targets_threads_uniform.pct` |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Eligible warps/cycle | `smsp__warps_eligible.sum.per_cycle_active` |
| SM throughput | SpeedOfLight SM throughput |
| Memory throughput | SpeedOfLight memory throughput |
| Issued warps/cycle | SchedulerStats |
| Branch and barrier stalls | WarpStateStats and SourceCounters |
| Active threads at root-loop source lines | SourceCounters source view |

First save the metric inventory for the installed Nsight Compute version:

```bash
: "${PDCS_RUN_DIR:?Set PDCS_RUN_DIR to the immutable run directory}"
mkdir -p "$PDCS_RUN_DIR/ncu"

ncu --query-metrics \
  > "$PDCS_RUN_DIR/ncu/available_metrics.txt"
```

A concrete primary profiling command is:

```bash
: "${PDCS_RUN_DIR:?Set PDCS_RUN_DIR to the immutable run directory}"

CUDA_VISIBLE_DEVICES=2 ncu \
  --target-processes all \
  --kernel-name regex:massive_block_proj \
  --nvtx \
  --nvtx-include "PDCS_PROJECTION/" \
  --launch-count 1 \
  --section SpeedOfLight \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section SourceCounters \
  --export "$PDCS_RUN_DIR/ncu/iteration_interleaved_thread_seed2026" \
  ./.julia-bin/julia --project=. \
  benchmark/rescaled_soc_divergence_2x2.jl \
    --experiment iteration \
    --layout interleaved \
    --strategy threadWise \
    --cone-count 1048576 \
    --cone-dimension 10 \
    --seed 2026 \
    --profile-one
```

Repeat for grouped/interleaved and thread-wise/warp-wise. Use the native
warp-wise kernel name for the warp-wise runs. Do not select the target with
`--launch-skip`: that is fragile when initialization or warm-up launch counts
change. Save both `.ncu-rep` and CSV exports.

Branch-target uniformity is supplementary. It may not capture lanes that are
inactive because their root search has already converged. The measured timing,
per-cone root-work counts, within-warp spread, and modeled active-lane
efficiency are the primary evidence. In particular, a near-100% global
branch-target-uniformity metric does not rule out loop-exit imbalance; inspect
the active-thread information at the annotated root-loop source lines.

If Nsight reports `ERR_NVGPUCTRPERM`, record the failure and report profiling
metrics as unavailable for that environment. Do not rerun the full workflow
with broad privileges by default.

---

## 10. Sustained GPU-utilization collection

For each primary \(2\times2\) cell, loop the workload for at least 30 seconds,
which yields at least 30 one-second `dmon` samples after warm-up.
Before each projection, restore the immutable input and reset the root state.
The reset occurs outside the projection NVTX range.

Use:

```bash
: "${PDCS_RUN_DIR:?Set PDCS_RUN_DIR to the immutable run directory}"
mkdir -p "$PDCS_RUN_DIR/nvidia_smi"

PDCS_GPU_INDEX=2
PDCS_DMON_OUTPUT="$PDCS_RUN_DIR/nvidia_smi/dmon_iteration_interleaved_thread_seed2026.txt"
PDCS_DMON_PID=

cleanup_dmon() {
  if [ -n "${PDCS_DMON_PID:-}" ]; then
    kill "$PDCS_DMON_PID" 2>/dev/null || true
    wait "$PDCS_DMON_PID" 2>/dev/null || true
  fi
}
trap cleanup_dmon EXIT INT TERM

nvidia-smi dmon -i "$PDCS_GPU_INDEX" -s pucvmet -d 1 \
  > "$PDCS_DMON_OUTPUT" &
PDCS_DMON_PID=$!

CUDA_VISIBLE_DEVICES="$PDCS_GPU_INDEX" \
./.julia-bin/julia --project=. \
  benchmark/rescaled_soc_divergence_2x2.jl \
    --experiment iteration \
    --layout interleaved \
    --strategy threadWise \
    --cone-count 1048576 \
    --cone-dimension 10 \
    --seed 2026 \
    --duration 30

cleanup_dmon
PDCS_DMON_PID=
trap - EXIT INT TERM
```

Here `PDCS_GPU_INDEX` is the physical `nvidia-smi` index; when
`CUDA_VISIBLE_DEVICES=2`, Julia normally exposes that device as CUDA-visible
index 0. Record the UUID to prove the mapping. The wrapper's cleanup trap
stops the monitor even if Julia fails.

Report:

- sustained SM-busy mean, median, 10th percentile, 90th percentile, and peak;
- SM and memory Speed-of-Light throughput;
- achieved occupancy;
- eligible warps/cycle;
- projection launches completed during the 30-second interval;
- time spent in input/root-state restoration versus projection launches;
- clocks, power, temperature, GPU UUID, physical and CUDA-visible indices, and
  competing-process snapshots.

The coarse `dmon` interval necessarily includes the device-to-device reset
copies between projections. Disclose this explicitly and use NVTX-delimited
Nsight data to characterize the projection kernel alone.

Compare the sustained SM-busy distribution explicitly with the reviewer's
suggested 80--90% range, but treat that comparison as descriptive rather than
as a divergence acceptance gate. Interpret it together with kernel-level
throughput. A memory-bound kernel may have low SM arithmetic throughput and
high memory throughput while still using the GPU efficiently.

High utilization alone does not prove that divergence is absent. Utilization
is supplementary to \(A\), \(\Theta\), and the manipulation check.

---

## 11. Publication tables

Generate the following tables directly from immutable seed-level summaries.

### 11.1 Manipulation check

| Divergence source | Selected family / work count | Layout | Independent seeds | Within-warp spread [IQR] | Within-warp coherence measure [IQR] | Manipulation gate |
|:---|:---|:---|---:|---:|---:|:---|
| Same-branch root work | Positive / bisections, or negative / oracle evaluations | Grouped | 20 | \(S^G\) | \(E^G\) | Defined by Section 3.9 |
| Same-branch root work | Same frozen family | Interleaved | 20 | \(S^I\) | \(E^I\) | Defined by Section 3.9 |
| Top-level branch | Four verified paths | Grouped | 20 | Branch-path spread | Modal-path fraction | Verified branch grouping |
| Top-level branch | Same four paths | Interleaved | 20 | Branch-path spread | Modal-path fraction | Verified branch interleaving |

### 11.2 Absolute timing

| Divergence source | Layout | Thread-wise time | Warp-wise time | Thread/Warp geometric-mean ratio \(A\) with 95% CI | Interpretation rule |
|:---|:---|---:|---:|---:|:---|
| Same-branch root work | Grouped | Seed-level summary | Seed-level summary | Reference \(A_G\) | Descriptive |
| Same-branch root work | Interleaved | Seed-level summary | Seed-level summary | Primary \(A\) | Section 8.3 |
| Top-level branch | Grouped | Seed-level summary | Seed-level summary | Reference \(A_G\) | Descriptive |
| Top-level branch | Interleaved | Seed-level summary | Seed-level summary | Secondary \(A\) | Section 8.3 |

### 11.3 Divergence effects

| Divergence source | Thread interleaving ratio \(R_T\) with 95% CI | Warp ordering-control ratio \(R_W\) with 95% CI | Interaction \(\Theta\) with 95% CI | 5% decision |
|:---|---:|---:|---:|:---|
| Same-branch root work | Seed-block estimate | Seed-block estimate | Primary interaction | Section 8.3 |
| Top-level branch | Seed-block estimate | Seed-block estimate | Secondary interaction | Section 8.3 |

### 11.4 Utilization

| Kernel | Layout | Sustained SM busy, mean/median/P10/P90/peak | SM SOL | Memory SOL | Occupancy | Eligible warps/cycle |
|:---|:---|---:|---:|---:|---:|---:|
| Thread-wise | Grouped | 30-second measurement | Nsight Compute | Nsight Compute | Nsight Compute | Nsight Compute |
| Thread-wise | Interleaved | 30-second measurement | Nsight Compute | Nsight Compute | Nsight Compute | Nsight Compute |
| Warp-wise | Grouped | 30-second measurement | Nsight Compute | Nsight Compute | Nsight Compute | Nsight Compute |
| Warp-wise | Interleaved | 30-second measurement | Nsight Compute | Nsight Compute | Nsight Compute | Nsight Compute |

Do not publish a table containing unresolved labels such as
`parametric-similar` unless its generator follows Section 5 and its manifest
is retained.

---

## 12. Interpretation rules

Apply these rules before drafting the reviewer response.

### Outcome A: no material absolute or interaction penalty

Let \(U_{.95}(X)\) and \(L_{.95}(X)\) denote the upper and lower one-sided 95%
confidence bounds for ratio \(X\). If

\[
U_{.95}(A)<1.05,\qquad U_{.95}(\Theta)<1.05,
\]

state that the experiment rules out a 5% thread-wise slowdown relative to
warp-wise and a 5% extra thread-wise susceptibility under the tested
high-divergence workload.

### Outcome B: divergence penalty exists but thread-wise remains faster

If

\[
L_{.95}(\Theta)>1.05,\qquad U_{.95}(A)<1,
\]

state that interleaving measurably degrades thread-wise efficiency, but the
thread-wise kernel remains faster in absolute time for the tested cone count
and dimension.

### Outcome C: thread-wise becomes materially slower

If \(L_{.95}(A)>1.05\), state that
thread-wise is materially slower than warp-wise under the tested
high-divergence workload. Discuss whether the strategy selector should avoid
thread-wise projection when measured or predicted iteration spread is high.

### Outcome D: inconclusive

If either confidence interval crosses the 5% decision boundary, report the
result as inconclusive. Do not write "no divergence penalty" merely because a
null-hypothesis test is nonsignificant.

Every conclusion must be limited to:

- the tested GPU;
- the tested cone dimensions and counts;
- the tested diagonal and boundary-ratio generators;
- cold-start root search unless a separate warm-start experiment is reported.

---

## 13. Reviewer-response structure after execution

The final response should use this order:

1. explain why thread-wise can suffer intra-warp divergence while warp-wise is
   the control;
2. state that the same cone multiset was used in grouped and interleaved
   layouts;
3. report the selected same-branch family and the manipulation check showing
   that measured root-work spread increased and modeled active-lane
   efficiency decreased;
4. report the absolute high-divergence ratio \(A\);
5. report the interaction \(\Theta\);
6. report sustained GPU activity and the relevant Nsight bottleneck;
7. state the predeclared 5% conclusion;
8. limit the claim to the tested workloads and note that profiler branch
   metrics are supplementary to timing and iteration counters.

Do not lead with five loosely defined workload labels. Do not use the old
independent \(\sigma=0.05\) versus \(\sigma=2\) comparison as evidence about
similar versus heterogeneous cones. Do not use warm-start speedup as evidence
that divergence is absent.

---

## 14. Execution checklist

### Task 1: Reproducible case generator and manifest

**Files:**

- Create: `benchmark/rescaled_soc_divergence_2x2.jl`
- Create during execution:
  `benchmark/results/rebuttal/soc_divergence_2x2/<UTC-run-id>/case_manifest.csv`

- [ ] Implement the Section 3 Gaussian direction and centered log-diagonal
  generators with seed- and cone-index-addressable streams.
- [ ] Implement both predeclared positive- and negative-root candidate
  families and the pilot-only family-selection rule.
- [ ] Implement the base-parametric generator in Section 5.
- [ ] Add a deterministic test that two runs with seed 2026 produce identical
  cone IDs, inputs, diagonals, and manifest hashes.
- [ ] Add a test that changing only the seed changes the manifest hash.
- [ ] Record every generator constant in `case_manifest.csv`.

**Acceptance:** repeated generation with the same seed is bitwise identical;
the manifest contains no implicit default parameter.

### Task 2: Diagnostic root-search counters

**Files:**

- Modify: `src/pdcs_gpu/cuda/massive_block_proj.cu`
- Modify: `src/pdcs_gpu/cuda/sufficient_block_proj.cu`
- Create during execution: `root_work_raw.csv.zst`
- Create during execution: `root_work_summary.csv`

- [ ] Add compile-time guarded output slots for branch code, expansion count,
  warm-state tests, Newton attempts/accepts, bisection count, oracle count,
  termination reason, final bracket/normalized width, `MAX_ITER`, and a
  freshly recomputed final residual.
- [ ] Make the cone-owning thread write each thread-wise record and lane 0
  write each warp-wise record, without root-loop atomics.
- [ ] Add hand-constructed feasible, polar, positive-root, and negative-root
  tests and verify their branch codes.
- [ ] Add same-negative-root cases with different known interval-expansion or
  total-oracle counts and verify counter ordering.
- [ ] Compare diagnostic and uninstrumented outputs on 1,024 cones.

**Acceptance:** counters change on the intended cases, projected values agree
under the Section 7.1 mixed tolerance, and the uninstrumented build contains
no diagnostic stores.

### Task 3: Exact paired layouts

**Files:**

- Modify: `benchmark/rescaled_soc_divergence_2x2.jl`
- Create during execution: `permutations.csv.zst`

- [ ] Implement the zero-based grouped and interleaved formulas in Section
  3.8.
- [ ] Implement the same formulas for the four branch classes in Section 4.
- [ ] Save forward and inverse permutations.
- [ ] Test ID/hash equality, rebuild monotone offsets, and verify bitwise
  inverse recovery for every payload array.
- [ ] Run the manipulation gate before allowing timing mode.

**Acceptance:** layouts differ only by a verified permutation, and the
same-branch root-work manipulation satisfies every Section 3.9 condition.

### Task 4: Correctness gate

**Files:**

- Modify: `benchmark/rescaled_soc_divergence_2x2.jl`

- [ ] Compare 1,024 sampled GPU projections per experiment and seed with the
  CPU implementation.
- [ ] Check finiteness, primal and polar feasibility, complementarity/Moreau
  residuals, KKT residual, branch class, final root residual, and `MAX_ITER`.
- [ ] Check grouped/interleaved output equality after inverse permutation.
- [ ] Check thread-wise/warp-wise output equality after inverse permutation.
- [ ] Refuse to write timing output when any correctness check fails.

**Acceptance:** every timed seed has a passing row in `correctness.csv`.

### Task 5: Uninstrumented timing

**Files:**

- Modify: `benchmark/rescaled_soc_divergence_2x2.jl`
- Create: `timings_raw.csv`
- Create: `timings_seed_summary.csv`

- [ ] Implement five warm-ups, ten measured launches, state restoration, and
  CUDA-event synchronization.
- [ ] Implement the four-order Williams schedule across all ten timing rounds.
- [ ] Run seeds 2026 through 2035 for the iteration experiment.
- [ ] Run seeds 2026 through 2035 for the branch experiment.
- [ ] Store all technical replicates and seed-level medians.

**Acceptance:** every experiment has 10 complete paired seed blocks, each
containing all four \(2\times2\) cells.

### Task 6: Profiling and sustained utilization

**Files:**

- Create: `ncu/*.ncu-rep`
- Create: `ncu/*.csv`
- Create: `nvidia_smi/*.txt`
- Create: `utilization_summary.csv`

- [ ] Profile seeds 2026--2028 for all four primary cells.
- [ ] Compile the optimized uninstrumented PTX with `-lineinfo`, select one
  target launch with NVTX, and export the requested Nsight Compute sections
  and raw metric names.
- [ ] Run every primary cell for at least 30 seconds with input and root
  restoration.
- [ ] Record dmon samples, reset-versus-projection time, completed launch
  counts, device-index mapping, and competing-process snapshots.
- [ ] Keep profiler replay timing out of `timings_raw.csv`.

**Acceptance:** every reported profiler value is traceable to a saved raw
artifact and exact case manifest.

### Task 7: Seed-blocked analysis

**Files:**

- Create: `benchmark/analyze_soc_divergence_2x2.jl`
- Create: `effect_estimates.csv`
- Create: `publication_tables.md`

- [ ] Compute \(A_s\), \(R_{T,s}\), \(R_{W,s}\), and \(\Theta_s\) for every
  complete seed.
- [ ] Compute log-scale geometric means, paired \(t\)-intervals, and 10,000
  seed-block bootstrap intervals.
- [ ] Apply the predeclared 5% decision rules without inspecting or changing
  the threshold.
- [ ] Generate all Section 11 tables directly from the saved summaries.

**Acceptance:** analysis fails if a seed block is incomplete, a manipulation
gate failed, or a correctness result is missing.

### Task 8: Reviewer response and manuscript update

**Files:**

- Modify: `doc/pdcs_overleaf/R1-2_r1-3.md`
- Modify after author approval:
  `doc/pdcs_overleaf/response_to_reviewers_PDCS.tex`

- [ ] Write the response in the order specified in Section 13.
- [ ] Include both \(A\) and \(\Theta\) with confidence intervals.
- [ ] Include the manipulation and utilization tables.
- [ ] Restrict claims according to Section 12.
- [ ] Cross-check every number against `publication_tables.md`.

**Acceptance:** no causal claim depends on an unpaired workload comparison,
no profiler replay time is reported as runtime, and every reported number has
a raw-artifact path.
