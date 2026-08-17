# R3.5 full report: coordinate-wise diagonal versus scalar-per-cone rescaling on Fisher markets

## Executive summary

This report answers the R3.5 reviewer question with a controlled Fisher-market
experiment. The formal comparison is between two representations of the same
Ruiz--Pock--Chambolle preconditioner: unrestricted coordinate-wise diagonal
factors and one shared scalar factor per structured cone. It is not a
comparison against an unscaled solver.

We first correct the experimental formulation so that the utility coefficients
appear directly in the exponential-cone mapping. This correction matters
because the earlier lifted form places the utilities in separate equality
rows and leaves the exponential-cone template with only two unit entries per
buyer. The corrected direct model exposes the within-cone coordinate
imbalance that diagonal rescaling is intended to address.

On five paired dense instances with 100 buyers and 5,000 goods, corresponding
to 500,000 allocation variables, diagonal rescaling passes the independent
primal verification gate in 5/5 runs. Scalar-per-cone passes in 0/5 runs, even
though the solver internally reports `OPTIMAL` in all five scalar runs. The
scalar mode requires 10.78 times as many iterations and 8.64 times as much
solver time in geometric mean. Its per-seed time ratio is 6.33--12.58.

The result supports retaining coordinate-wise diagonal rescaling as the robust
default and exposing scalar-per-cone as a user-selectable option. It does not
claim that diagonal scaling always wins or that problem dimension alone
determines the winner.

## Question

The 62-instance CBF experiment shows that scalar-per-cone rescaling often
reduces projection cost, but also has heterogeneous convergence effects. This
supplement asks whether retaining separate coordinate-wise diagonal factors
can be important on a large application with many exponential cones.

The comparison is **not** diagonal rescaling versus no rescaling. Both modes
run the same Ruiz--Pock--Chambolle pipeline:

- `diagonal`: `scalar_cone_rescaling=false`, retaining separate factors for
  the coordinates of every structured cone;
- `scalar_cone`: `scalar_cone_rescaling=true`, replacing those factors by one
  common scalar per cone.

All other solver settings are identical.

If the coordinate-wise candidate factors in a structured cone block $B$ are
$s_j$, the scalar mode applies

\[
s_B=\max_{j\in B}s_j,
\qquad s_j\leftarrow s_B\quad(j\in B).
\]

This scalarization is performed in every Ruiz round and in the
Pock--Chambolle pass before the scaled data are formed. The maximum keeps the
operation deterministic and conservative with respect to the candidate
factors. The same rule is used for structured primal and dual cone blocks.
For this Fisher model, each exponential cone is three-dimensional.

The computational attraction is that a common factor preserves the cone
under uniform scaling and permits the ordinary constant-scale projection
path. The tradeoff is that it removes two degrees of freedom from every
three-dimensional exponential cone. Coordinate-wise diagonal scaling instead
uses the more general weighted projection but can separately equilibrate its
coordinates.

## Formulation correction

For buyers $i=1,\ldots,m$, goods $j=1,\ldots,n$, utilities
$U_{ij}\ge 0$, positive buyer weights $w_i$, and supplies $b_j$, the
Eisenberg--Gale objective can be written as

\[
\max_{X\ge 0}\ \sum_{i=1}^m w_i
\log\!\left(\sum_{j=1}^n U_{ij}X_{ij}\right)
\quad\text{s.t.}\quad \sum_{i=1}^m X_{ij}=b_j.
\]

We use the exponential cone

\[
K_{\exp}=\operatorname{cl}\{(a,b,c): b>0,\ b\exp(a/b)\le c\}.
\]

Therefore $p_i\le\log t_i$ is equivalent to
$(p_i,1,t_i)\in K_{\exp}$.

The previous Fisher benchmark uses the valid lifted model

\[
 U_iX_i-t_i=0,
 \qquad (p_i,1,t_i)\in K_{\exp}.
\]

Its exponential-cone matrix is a fixed template whose nonzero entries are all
one; the instance utilities occur in separate equality rows. This is not a
modeling error, but it obscures the coordinate imbalance that the R3.5
experiment is intended to study.

More explicitly, if the variables for buyer $i$ are $(p_i,t_i)$, its cone
affine map is

\[
\begin{bmatrix}1&0\\0&0\\0&1\end{bmatrix}
\begin{bmatrix}p_i\\t_i\end{bmatrix}
-\begin{bmatrix}0\\-1\\0\end{bmatrix}
=\begin{bmatrix}p_i\\1\\t_i\end{bmatrix}.
\]

Globally, the lifted exponential-cone matrix is

\[
Q=\left[0_{3m\times mn}\quad
I_m\otimes\begin{bmatrix}1&0\\0&0\\0&1\end{bmatrix}\right],
\qquad
d=\mathbf 1_m\otimes\begin{bmatrix}0\\-1\\0\end{bmatrix}.
\]

Thus $\operatorname{nnz}(Q)=2m$, and every nonzero in $Q$ is exactly one.
The middle coordinate equal to one comes from the constant term $-d$, not
from $Q$. The data have not disappeared: $U$ is in the lifted utility
equalities, $w$ is in the objective, and $b$ is in the supply
right-hand side.

The supplement therefore eliminates the utility auxiliary variable and uses

\[
\begin{aligned}
\min_{X,p}\quad&-\sum_iw_ip_i,\\
\sum_iX_{ij}&=b_j,\\
(p_i,1,\sum_jU_{ij}X_{ij})&\in K_{\exp},\\
X&\ge0.
\end{aligned}
\]

Now `U_ij` occurs directly in the third coordinate row of buyer `i`'s
exponential cone. We deliberately do not absorb `w_i` into the cone: keeping
weights only in the objective isolates the effect of the utility mapping from
additional random weight scaling.

The direct cone block for buyer $i$, ordered as $(X_{i,:},p_i)$, is

\[
\begin{bmatrix}
0_{1\times n}&1\\
0_{1\times n}&0\\
U_{i,:}&0
\end{bmatrix}
\begin{bmatrix}X_{i,:}^{\mathsf T}\\p_i\end{bmatrix}
-\begin{bmatrix}0\\-1\\0\end{bmatrix}
=
\begin{bmatrix}
p_i\\1\\U_{i,:}X_{i,:}^{\mathsf T}
\end{bmatrix}.
\]

An alternative valid formulation absorbs the weights into the cone:

\[
(-z_i,w_i,w_i\sum_jU_{ij}X_{ij})\in K_{\exp},
\qquad \min\sum_i z_i.
\]

We do not use that alternative in the formal experiment because it introduces
the products $w_iU_{ij}$ into the cone map. Leaving $w_i$ in the
objective gives the cleanest test of whether a single cone scalar can replace
coordinate-wise factors for the direct utility map.

The direct formulation has `mn+m` variables, `n` zero-cone rows, `m`
exponential cones, and `mn+nnz(U)+m` matrix nonzeros. Relative to the lifted
form it eliminates exactly `m` variables, `m` equality rows, and `2m` matrix
nonzeros. Structure and analytical-equivalence tests pass 51/51 assertions.
On the small solved gate, direct and lifted objectives differ by less than
`7e-8`.

For the formal dense case $m=100,n=5000$, the direct model has 500,100
variables, 5,300 conic rows, and 1,000,100 matrix nonzeros. The 500,000
allocation variables are lower-bounded by zero; the remaining 100 variables
are the logarithmic utility variables $p_i$.

## Protocol

### Instance generation and fixed settings

| Item | Formal setting |
|:---|:---|
| Generator | `../PDCS_fork/benchmark/large_scale_fisher_market/fisher_market_common.jl` |
| Buyers / goods | $m=100$, $n=5000$ |
| Utility density | 1.0 |
| Allocation variables | 500,000 |
| Seeds | 2026, 2027, 2028, 2029, 2030 |
| Numeric type | Float64 |
| Matrix storage | in-memory sparse CSC |
| Solver | cuPDCS GPU |
| Rescaling | Ruiz--Pock--Chambolle in both modes |
| Compared switch | `scalar_cone_rescaling=false/true` only |
| Solver tolerance | `1e-6` absolute and relative |
| Time limit | 600 seconds per solve |
| Termination check | every 1,000 iterations |
| Pairing | same numerical digest and same physical GPU within each seed |
| Order | counterbalanced across seeds |
| Saved data | scalar records, logs, hashes, and summaries; no generated matrix or solution vector |

The generator uses a local seeded random-number generator. Buyer weights and
positive utility values are sampled in Float64, and every good has supply
$0.25m=25$. A numerical digest covers the weights, sparse utility indices and
values, and supply. The analyzer requires the two modes for a seed to have
the same digest.

The runner uses adaptive restart, adaptive step-size weight, aggressive
reflection, resolving, averaged iterates, and duality-gap restart in both
modes. Acceleration and KKT restart are disabled in both. No solver setting
other than `scalar_cone_rescaling` changes.

The formal family has 100 buyers, 5,000 goods, dense utilities, and 500,000
nonnegative allocation variables. Five deterministic seeds, 2026--2030, are
run in both modes. Every seed's pair runs sequentially on the same physical
NVIDIA H100 GPU; mode order is counterbalanced across seeds. Arithmetic is
Float64, the solver target is `1e-6`, and the time limit is 600 seconds.

The instance is regenerated and solved in memory. Only scalar TOML records,
logs, hashes, and summaries are retained. Success requires finite primal
values and independent supply, nonnegativity, and exponential-cone checks.
The common primal violation gate is `2e-5`; a solver-reported `OPTIMAL` status
alone is not accepted.

### Independent primal verification

For a recovered allocation $X$ and logarithmic utility vector $p$, the
post-solve checker recomputes

\[
r_{\mathrm{supply}}=
\frac{\max_j|\sum_iX_{ij}-b_j|}{\max(1,|b_j|)},
\qquad
r_{\mathrm{nonneg}}=\max(0,-\min_{ij}X_{ij}),
\]

and, in log space to avoid overflow,

\[
r_{\exp}=\max_i\max\!\left(
0,\ p_i-\log\left(\sum_jU_{ij}X_{ij}\right)
\right).
\]

A row is independently verified only if the primal contains no nonfinite
values, the recomputed objective is finite, the solver termination is
accepted, and

\[
\max(r_{\mathrm{supply}},r_{\mathrm{nonneg}},r_{\exp})\le 2\times10^{-5}.
\]

The `2e-5` audit gate is exactly 20 times the requested `1e-6` solver
tolerance. It is common to both modes and was frozen in `experiment.toml`
before formal analysis.

## Preliminary checks

The formulation test suite passes 51/51 assertions. It checks sparse matrix
dimensions and positions, direct-versus-lifted analytical equivalence,
objective placement, cone indices, and independent verification behavior.

A solved smoke instance with $m=20$, $n=32$, density 0.5, and seed 2026
provides a correctness-scale comparison:

| Mode | Verified | Iterations | Solver time (s) | Objective | Supply residual | EXP violation |
|:---|:---:|---:|---:|---:|---:|---:|
| Diagonal | yes | 7,000 | 7.19 | -20.395530579 | `1.27e-7` | `4.81e-7` |
| Scalar per cone | yes | 6,500 | 5.98 | -20.395530640 | `1.58e-8` | `6.14e-7` |

The direct diagonal solution and the lifted diagonal solution have objectives
-20.395530579 and -20.395530645, an absolute difference below `7e-8`.
This supports the implementation equivalence. The scalar mode is slightly
faster on this small instance, which is consistent with the expectation that
the cheaper projection path can help when the lost equilibration is harmless.
It is a single smoke case and is not used as a statistical performance claim.

## Formal result

| Seed | Diagonal verified | Scalar verified | Diagonal iter. | Scalar iter. | Diagonal time (s) | Scalar time (s) | Scalar/diagonal time | Diagonal EXP violation | Scalar EXP violation |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | yes | no | 50,000 | 559,000 | 23.95 | 202.23 | 8.45 | `2.24e-7` | `9.50e-5` |
| 2027 | yes | no | 40,000 | 627,000 | 18.64 | 234.43 | 12.58 | `1.17e-6` | `3.32e-5` |
| 2028 | yes | no | 52,000 | 432,000 | 25.06 | 158.64 | 6.33 | `6.46e-7` | `8.21e-5` |
| 2029 | yes | no | 56,000 | 602,000 | 23.48 | 221.60 | 9.44 | `3.93e-7` | `3.40e-5` |
| 2030 | yes | no | 56,000 | 521,000 | 24.62 | 186.28 | 7.57 | `6.47e-6` | `1.71e-4` |

Coordinate-wise diagonal rescaling verifies 5/5 rows; scalar-per-cone verifies
0/5 under the same independent gate, although all five scalar runs report
`OPTIMAL` internally. Median iterations are 52,000 versus 559,000, and median
solver times are 23.95 versus 202.23 seconds. The geometric-mean
scalar/diagonal ratios are 10.78 for iterations and 8.64 for solver time; the
per-seed time ratios range from 6.33 to 12.58.

| Aggregate | Diagonal | Scalar per cone | Scalar/diagonal |
|:---|---:|---:|---:|
| Independently verified | 5/5 | 0/5 | -- |
| Solver reports `OPTIMAL` | 5/5 | 5/5 | -- |
| Median iterations | 52,000 | 559,000 | -- |
| Median solver time | 23.95 s | 202.23 s | -- |
| Geometric-mean iteration ratio | -- | -- | 10.78 |
| Geometric-mean solver-time ratio | -- | -- | 8.64 |
| Maximum EXP violation | `6.47e-6` | `1.71e-4` | -- |

Every allocation vector is finite and has zero reported nonnegativity
violation. Supply residuals are also small: the maximum diagonal supply
residual is `7.83e-7`, while scalar residuals are below `1.38e-10`. The scalar
failures are therefore specifically caused by the exponential-cone condition,
not by NaNs, explicit projection failure, allocation negativity, or the supply
equalities.

The largest paired relative difference in recomputed objective is
`1.07e-6` (more exactly `1.0714e-6`). This apparent objective agreement does
not rescue the scalar runs: objective proximity is not a substitute for
primal cone feasibility. The independent check is necessary precisely because
all scalar runs carry an internal `OPTIMAL` label.

## Why the large direct model separates the modes

The first and second coordinates of buyer $i$'s exponential cone have a
single coefficient or a fixed constant, whereas the third coordinate is a
dense linear combination of 5,000 allocation variables. A single scalar
shared by the three coordinates cannot independently balance these different
row structures. Coordinate-wise scaling can.

The measured behavior matches that mechanism. Scalar-per-cone does not fail
by producing nonfinite iterates; instead it needs roughly an order of
magnitude more PDHG iterations and terminates with a systematically larger
log-space exponential-cone violation. The cheaper constant-scale projection
is therefore dominated by slower outer convergence on this family.

This also explains why the lifted model is a poor diagnostic for this
question. In the lifted form, the utility row is separated from
$(p_i,1,t_i)$, whose matrix template contains only unit coefficients. The
direct form is equivalent as an optimization problem but exposes the actual
utility map inside the structured cone, where the diagonal-versus-scalar
choice acts.

## Relation to the 62-instance R3.5 campaign

The Fisher experiment supplements rather than replaces the main 62-instance
CBF campaign. In that campaign, scalar-per-cone has a favorable aggregate
paired wall-time ratio of 0.9029, wins 40 of 62 instances, and preserves the
same verified-run count as diagonal scaling. However, its median ratio is
0.9913 and individual regressions occur. Neither total dimension nor maximum
cone dimension gives a reliable automatic selection threshold.

The two bodies of evidence answer different parts of the reviewer question:

| Evidence | What it establishes |
|:---|:---|
| 62-instance CBF corpus | Scalar-per-cone can reduce aggregate cost and should be available to users, but its benefit is heterogeneous. |
| Small Fisher smoke case | Both modes can solve the same direct formulation; scalar can be slightly cheaper when equilibration loss is harmless. |
| Large direct Fisher family | Coordinate-wise diagonal factors can be crucial for convergence speed and independently verified cone feasibility. |

The combined policy follows directly: retain diagonal rescaling as the robust,
backward-compatible default; expose scalar-per-cone as an opt-in performance
option; do not use dimension alone to switch automatically.

The already completed no-rescaling experiment is intentionally outside this
report. Here, `diagonal` and `scalar_cone` both mean that rescaling is on.
There is no `none` arm in the formal Fisher comparison.

## Conclusion and boundary

This Fisher family provides a concrete large-scale case where
coordinate-wise diagonal rescaling is not dispensable. Once a buyer's full
utility row appears in the third exponential-cone coordinate, one scalar for
the entire cone cannot separately balance the one-coefficient `p_i` row and
the many-coefficient utility row. The saved projection work is overwhelmed by
roughly an order of magnitude more PDHG iterations.

The defensible claim is application-specific:

> On the direct 500,000-allocation-variable Fisher family, coordinate-wise
> diagonal rescaling is essential for reliable `1e-6`-target solutions and is
> 6.3--12.6 times faster than scalar-per-cone rescaling across five seeds.

This does not establish a universal dimension threshold. The 62-instance CBF
corpus still shows that total dimension alone does not predict the winning
mode. Together, the experiments justify keeping coordinate-wise diagonal
rescaling as the robust default while exposing scalar-per-cone as an opt-in
performance option for workloads on which its cheaper projections do not
damage convergence.

## Scope and limitations

The conclusion should be read with the following boundaries:

- The formal result covers one generated Fisher family, five seeds, one model
  scale, dense utilities, and one GPU architecture.
- The two modes were paired within seed and counterbalanced, but each seed was
  run once per mode; the study does not quantify repeated-run timing variance.
- The experiment isolates the utility mapping by leaving buyer weights in the
  objective. It does not test the weight-absorbed conic formulation.
- The common time limit is 600 seconds. All runs terminated before the limit,
  so the reported comparison is not censored by it.
- The independent audit is primal. It checks supply, nonnegativity, and
  exponential-cone membership, but it is not a separately implemented full
  KKT verifier.
- The evidence demonstrates a large structured instance where diagonal
  rescaling matters; it does not identify a universal dimension threshold.

These limitations do not alter the paired observation, but they constrain the
breadth of the claim. A broader Fisher grid over buyer count, good count,
density, and utility dynamic range would be the appropriate follow-up if a
predictive automatic selector is desired.

## User-facing configuration

The default remains coordinate-wise diagonal rescaling:

```julia
set_optimizer_attribute(model, "rescaling_method", :ruiz_pock_chambolle)
set_optimizer_attribute(model, "scalar_cone_rescaling", false)
```

Users can opt into one scalar per structured cone without disabling
rescaling:

```julia
set_optimizer_attribute(model, "rescaling_method", :ruiz_pock_chambolle)
set_optimizer_attribute(model, "scalar_cone_rescaling", true)
```

The explicit method alias
`:ruiz_pock_chambolle_scalar_cone` selects the same scalar-per-cone behavior.
The experiment driver also leaves the per-solve time limit selectable through
`--time-limit SECONDS`.

## Reviewer-ready response

> We thank the reviewer for asking whether a single rescaling scalar can be
> used for each structured cone. We implemented this option by replacing the
> coordinate-wise Ruiz--Pock--Chambolle candidate factors in every cone block
> with their block maximum. We compared it with the original coordinate-wise
> diagonal factors while holding the formulation and all other solver options
> fixed.
>
> In the general 62-instance CBF campaign, scalar-per-cone rescaling reduces
> aggregate synchronized wall time by 9.7%, but the effect is heterogeneous
> and neither total dimension nor maximum cone dimension reliably predicts the
> winning mode. We therefore expose scalar-per-cone rescaling as a user option
> rather than making it the default.
>
> To explain why the diagonal mode should be retained, we also evaluated a
> direct Fisher-market formulation in which each buyer's utility row appears
> in the third coordinate of its exponential cone. On five paired instances
> with 500,000 allocation variables, coordinate-wise diagonal rescaling passes
> independent primal verification in 5/5 runs, whereas scalar-per-cone passes
> in 0/5 runs under the same `2e-5` audit gate. Scalar-per-cone requires 10.78
> times as many iterations and 8.64 times as much solver time in geometric
> mean; its per-seed time ratio is 6.33--12.58. A small smoke instance is solved
> by both modes and is slightly faster with scalar-per-cone rescaling.
>
> These results show the intended tradeoff: a shared cone scalar can reduce
> projection cost when within-cone equilibration is unimportant, but
> coordinate-wise factors can be decisive on a large imbalanced conic map.
> Accordingly, we retain diagonal rescaling as the robust default and provide
> scalar-per-cone rescaling as an explicit opt-in option.

## Reproducibility record

The formal run used Julia 1.12.6 and NVIDIA H100 80GB HBM3 GPUs with driver
595.71.05. The driver reported CUDA 13.2 compatibility; the driver script
selected the CUDA 12.6 runtime path and existing `sm_90` production projection
artifacts. GPUs used for the experiment were empty at the environment audit.

Recorded repository commits:

- experiment repository: `6b08bf5d7ca0588b74d33df531a3044f1a2b0e2e`;
- sibling generator repository: `e6dd048152468ac5045a562e39800527690dedeb`.

Recorded SHA-256 hashes:

| Artifact | SHA-256 |
|:---|:---|
| `experiment.toml` | `0c95999b75f5208da0e76d4c80ef6776f6f3fbbc40172826221a9df59c73f4ed` |
| `FisherDirectFormulation.jl` | `21cd9bbeede9f894ac4ea9805cdb5dc1bd2442cd569c357fd00584cc6359ec35` |
| `run_case.jl` | `a96e8114fa8bc19ed1957109224f8816cbb77794f5dea157cddb49fb63728870` |
| `analyze_results.jl` | `8e2ba87f05039f08af382c8f110576c5f0c0fc86bc33c2f27ca2917527896621` |
| `run_experiment.sh` | `08ede0ad1031373075a2eedc0ee0a44fdd65d5541048250a42a71845fb3dea18` |

Run the formulation tests with the existing sibling Julia environment; no
package installation is required:

```bash
JULIA_DEPOT_PATH=../PDCS_fork/.julia-depot \
  ../PDCS_fork/.julia-bin/julia --startup-file=no --project=. \
  benchmark/R3.5_fisher_market/test_formulations.jl
```

Re-run the formal paired experiment on selected GPUs with:

```bash
bash benchmark/R3.5_fisher_market/run_experiment.sh \
  --gpus 2,3 --time-limit 600
```

Use a new `--output` directory, or pass `--force true`, to avoid silently
reusing existing records. The formal artifact directory is approximately
180 KiB because instances and solution vectors are regenerated in memory and
are intentionally not retained.

## Artifacts

- Frozen protocol: [`experiment.toml`](experiment.toml)
- Direct formulation: [`FisherDirectFormulation.jl`](FisherDirectFormulation.jl)
- Solver runner: [`run_case.jl`](run_case.jl)
- Same-GPU counterbalanced driver: [`run_experiment.sh`](run_experiment.sh)
- Exact audit: [`analyze_results.jl`](analyze_results.jl)
- Generated table: [`results/formal_diagonal_scalar/analysis/RESULTS.md`](results/formal_diagonal_scalar/analysis/RESULTS.md)
- Aggregate statistics: [`results/formal_diagonal_scalar/analysis/summary.toml`](results/formal_diagonal_scalar/analysis/summary.toml)
- Machine-readable pairs: [`results/formal_diagonal_scalar/analysis/pairs.csv`](results/formal_diagonal_scalar/analysis/pairs.csv)
- Captured environment: [`results/formal_diagonal_scalar/environment/environment.txt`](results/formal_diagonal_scalar/environment/environment.txt)
- Main 62-instance report: [`../R3.5_scalar_cone_rescaling_results.md`](../R3.5_scalar_cone_rescaling_results.md)
