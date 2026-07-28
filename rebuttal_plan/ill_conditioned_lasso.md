# AE-11: Experimental Plan for Controlled Ill-Conditioned Large-Scale Lasso Instances

## 1. Objective

The Associate Editor writes:

> The study would be more informative if it included at least one
> ill-conditioned or structurally different family of instances.

We address this comment by introducing a sparse Lasso instance family with an
exactly controlled **active-set condition number**. The experiments are
designed to distinguish and measure:

1. deterioration of local curvature on the optimal support;
2. the sensitivity of PDCS iterations and runtime to ill-conditioning;
3. the extent to which diagonal rescaling mitigates ill-conditioning;
4. the performance of cuPDCS relative to other first-order and interior-point
   solvers at a common KKT accuracy; and
5. whether the conclusions extend to large-scale sparse instances.

Throughout this document, the canonical Lasso formulation is

\[
    \min_{x\in\mathbb R^n}
    F(x):=
    \|Ax-b\|_2^2+\lambda\|x\|_1.
\]

This is the normalization used by the manuscript and by the SOCP
reformulation supplied to PDCS. Consequently, the gradient of the squared-loss
term is \(2A^\top(Ax-b)\). The generator, solver interfaces, verification
code, and manuscript must all use this convention; the meaning of
\(\lambda\) must not change implicitly across solvers.

### 1.1 Exact SOCP reformulation solved by PDCS

Introduce the positive and negative parts

\[
    x=x_1-x_2,
    \qquad
    x_1,x_2\in\mathbb R_+^n,
\]

and introduce the residual variable

\[
    y=A(x_1-x_2)-b.
\]

Because

\[
    \|x\|_1
    =
    \min_{\substack{x=x_1-x_2\\x_1,x_2\ge0}}
    \mathbf 1^\top(x_1+x_2),
\]

the Lasso problem is equivalent to

\[
    \begin{aligned}
    \min_{x_1,x_2,y,r}\quad&
        2r+\lambda\mathbf 1^\top(x_1+x_2)\\
    \mathrm{s.t.}\quad&
        y=A(x_1-x_2)-b,\\
    &   \|y\|_2^2\le 2r,\\
    &   x_1,x_2\in\mathbb R_+^n.
    \end{aligned}
\]

At every optimum, \(2r=\|y\|_2^2\), so the objective reduces exactly to
\(\|Ax-b\|_2^2+\lambda\|x\|_1\).

The equivalence is exact in both directions. Given any Lasso vector \(x\), set

\[
    x_1=\max\{x,0\},
    \qquad
    x_2=\max\{-x,0\},
    \qquad
    y=Ax-b,
    \qquad
    r=\frac12\|y\|_2^2.
\]

This produces an SOCP-feasible point with the same objective value. Conversely,
for any SOCP-feasible point, \(x=x_1-x_2\) satisfies

\[
    \|x\|_1\le\mathbf 1^\top(x_1+x_2),
    \qquad
    \|Ax-b\|_2^2=\|y\|_2^2\le2r.
\]

Therefore, the original Lasso objective at the recovered \(x\) is no larger
than the SOCP objective. Together, these two directions prove equality of the
optimal values and justify recovering \(x=x_1-x_2\) from the PDCS solution.

The quadratic epigraph constraint is represented by one standard
second-order cone. Define

\[
    \mathcal K_{\mathrm{soc}}^{m+2}
    :=
    \left\{(t,z)\in\mathbb R\times\mathbb R^{m+1}:
    t\ge\|z\|_2\right\}.
\]

Then

\[
    \|y\|_2^2\le2r
\]

is equivalent to

\[
    \boxed{
    \begin{bmatrix}
        (1+r)/\sqrt2\\
        (1-r)/\sqrt2\\
        y
    \end{bmatrix}
    \in\mathcal K_{\mathrm{soc}}^{m+2}.
    }
\]

Indeed,

\[
    \left(\frac{1+r}{\sqrt2}\right)^2
    -
    \left(\frac{1-r}{\sqrt2}\right)^2
    -
    \|y\|_2^2
    =
    2r-\|y\|_2^2.
\]

The SOC membership also enforces the appropriate nonnegative leading
coordinate. Therefore, the complete problem passed to PDCS consists only of:

- the linear objective
  \(2r+\lambda\mathbf 1^\top(x_1+x_2)\);
- the affine equality
  \(y=A(x_1-x_2)-b\);
- the nonnegative-cone constraint
  \((x_1,x_2)\in\mathbb R_+^{2n}\); and
- one second-order cone of dimension \(m+2\).

After PDCS returns \(x_1\) and \(x_2\), recover the Lasso coefficient vector as

\[
    x=x_1-x_2.
\]

The independent verifier must evaluate both the SOCP residuals and the
original Lasso KKT residual at this recovered \(x\).

---

## 2. Why the Full-Matrix Condition Number Is Insufficient

The global spectral condition number is

\[
    \kappa_2(A)^2
    =
    \frac{\lambda_{\max}(A^\top A)}
         {\lambda_{\min}(A^\top A)}.
\]

For a high-dimensional Lasso problem with \(n>m\), \(A^\top A\) is singular,
and this condition number is necessarily infinite. This does not imply that
the Lasso instance is computationally intractable. The local optimization
difficulty is more directly related to the restricted Hessian on the optimal
support

\[
    S:=\{j:x_j^*\neq 0\},
\qquad
    H_S:=A_S^\top A_S.
\]

The primary controlled variable in this study is therefore

\[
    \boxed{
    \kappa_{\mathrm{active}}
    :=
    \kappa_2(A_S)^2
    =
    \frac{\lambda_{\max}(A_S^\top A_S)}
         {\lambda_{\min}(A_S^\top A_S)}
    }.
\]

This quantity characterizes local conditioning after the correct support has
been identified. We also report the KKT margin to verify that any observed
deterioration is not caused by ambiguity in active-set identification.

PDCS solves an SOCP reformulation of Lasso rather than applying ISTA directly
to the original objective. We must consequently report:

- \(\kappa_{\mathrm{active}}\) for the original Lasso matrix;
- the corresponding active-block condition estimate after PDCS diagonal
  rescaling; and
- common primal, dual, gap, and KKT residuals for the SOCP formulation.

---

## 3. Candidate Constructions and Final Choice

### 3.1 Dense SVD construction

A direct construction is

\[
    A_S=U\Sigma V^\top
\]

with prescribed singular values. At large scale, however, this requires
generating and storing dense orthogonal matrices. It changes the sparsity
structure of the original experiments and incurs at least \(O(ms)\) generation
and storage costs. It is not used in the main experiment.

### 3.2 Simple column scaling

Another possibility is to multiply columns by widely varying scales. This is
easy to generate, but the resulting ill-conditioning is dominated by unequal
column norms and may be almost entirely removed by diagonal rescaling. Such an
experiment would primarily test the preconditioner rather than the effect of
correlated variables on local curvature. We therefore do not use this
construction.

### 3.3 Sparse correlated column pairs

The selected construction uses sparse correlated column pairs. It has the
following advantages:

- exact control of \(\kappa_{\mathrm{active}}\);
- unit Euclidean norm for every active column;
- ill-conditioning that is not produced by simple column scaling;
- \(O(\operatorname{nnz}(A))\) generation and storage cost;
- no dense matrix or SVD requirement;
- scalability to millions of rows and columns; and
- a known optimal solution obtained through a reverse KKT construction.

---

## 4. Exactly Controlled Sparse Active Block

Let \(s=|S|\) be even. For each \(i=1,\ldots,s/2\), generate two sparse
orthonormal vectors \(u_i,v_i\in\mathbb R^m\), with different groups also
mutually orthogonal:

\[
    u_i^\top u_j=v_i^\top v_j=u_i^\top v_j=0
    \quad (i\neq j),
\]

\[
    \|u_i\|_2=\|v_i\|_2=1,
    \qquad
    u_i^\top v_i=0.
\]

For a correlation parameter \(\rho\in[0,1)\), define the \(i\)-th pair of
active columns by

\[
    a_{2i-1}=u_i,
    \qquad
    a_{2i}=\rho u_i+\sqrt{1-\rho^2}\,v_i.
\]

Both columns have unit norm, and their Gram block is

\[
    \begin{bmatrix}
    a_{2i-1}^\top a_{2i-1}
      &a_{2i-1}^\top a_{2i}\\
    a_{2i}^\top a_{2i-1}
      &a_{2i}^\top a_{2i}
    \end{bmatrix}
    =
    \begin{bmatrix}
    1&\rho\\
    \rho&1
    \end{bmatrix}.
\]

Thus,

\[
    A_S^\top A_S
    =
    \operatorname{blockdiag}
    \left(
    \begin{bmatrix}1&\rho\\ \rho&1\end{bmatrix},
    \ldots,
    \begin{bmatrix}1&\rho\\ \rho&1\end{bmatrix}
    \right).
\]

Each block has eigenvalues \(1+\rho\) and \(1-\rho\), so

\[
    \kappa_{\mathrm{active}}
    =
    \frac{1+\rho}{1-\rho}.
\]

For a target condition number \(K\), set

\[
    \boxed{\rho(K)=\frac{K-1}{K+1}}.
\]

The main experiment uses

\[
    K\in\{1,10^2,10^4,10^6\}.
\]

The corresponding correlations are:

| \(K\) | \(\rho=(K-1)/(K+1)\) |
|---:|---:|
| \(1\) | \(0\) |
| \(10^2\) | \(0.9801980198019802\) |
| \(10^4\) | \(0.9998000199980002\) |
| \(10^6\) | \(0.9999980000020000\) |

\(K=10^8\) may be included only as a numerical stress test. It must not enter
the primary conclusions unless all solvers can verify the instance and the
final KKT residual reliably in double precision.

---

## 5. Scalable Generation of Sparse Basis Vectors

### 5.1 Exactly orthogonal construction

The main experiment first uses an exactly orthogonal construction so that the
realized condition number equals the target value exactly.

Let each basis vector contain \(d\) nonzeros. Assign two disjoint row sets to
each pair \((u_i,v_i)\):

\[
    R_{u_i},R_{v_i}\subseteq[m],
    \qquad
    |R_{u_i}|=|R_{v_i}|=d.
\]

All such row sets are disjoint across pairs. Generate Rademacher entries on
each support:

\[
    [u_i]_j\in\{-1/\sqrt d,+1/\sqrt d\},
    \quad j\in R_{u_i},
\]

\[
    [v_i]_j\in\{-1/\sqrt d,+1/\sqrt d\},
    \quad j\in R_{v_i}.
\]

This produces exactly sparse, orthonormal unit vectors without
Gram--Schmidt. The column sparsities satisfy

\[
    \operatorname{nnz}(a_{2i-1})=d,
    \qquad
    \operatorname{nnz}(a_{2i})=2d.
\]

The active block therefore has

\[
    \operatorname{nnz}(A_S)=\frac32sd
\]

nonzeros, and both generation time and storage are \(O(sd)\).

### 5.2 Row permutation

After construction, apply one common random permutation of the \(m\) rows,
determined by the workload seed. This preserves the Gram matrix and its
condition number while avoiding visibly contiguous artificial row blocks.

### 5.3 Structural-naturalness sensitivity panel

Disjoint supports provide exact control but create an artificial block
structure. A smaller overlapping-support sensitivity panel tests whether the
conclusions depend on this structure:

- hold \(m,n,s,d,K\) fixed;
- retain \(80\%\) private support for each basis vector;
- draw the remaining \(20\%\) from a shared row pool;
- orthogonalize \(v_i\) locally against \(u_i\) within each pair;
- report the measured condition number obtained from
  \(A_S^\top A_S\), rather than claiming that it equals \(K\) exactly; and
- test whether the qualitative trend from the main experiment remains.

This sensitivity panel remains separate from the exactly controlled family
and must not be pooled with it when computing aggregate statistics.

---

## 6. Known Optimum and KKT Guarantee

### 6.1 Support and optimal solution

Set

\[
    S=\{1,\ldots,s\}.
\]

For each workload seed, generate a sign vector
\(q\in\{-1,+1\}^s\). Use opposite signs within each correlated pair:

\[
    q_{2i}=-q_{2i-1}.
\]

This choice is essential. The vector \((1,-1)\) is the eigenvector associated
with the smaller eigenvalue \(1-\rho\) of each Gram block. Opposite signs
therefore force the known optimum and the KKT construction to excite the
ill-conditioned direction. With equal signs, the construction would align
with the larger eigenvalue \(1+\rho\), potentially making an instance with a
large formal condition number artificially easy.

Define

\[
    x_j^*=c_jq_j,\quad j\in S,
    \qquad
    x_j^*=0,\quad j\notin S,
\]

where \(c_j\) is drawn independently from \([1,2]\), followed by a common
rescaling so that

\[
    \|x_S^*\|_2=1.
\]

Within the same workload seed, every value of \(K\) reuses the same \(x^*\).
Consequently, support size, sign pattern, and solution scale do not confound
the conditioning effect.

### 6.2 Reverse construction from the KKT conditions

Let

\[
    G:=A_S^\top A_S.
\]

For

\[
    \|Ax-b\|_2^2+\lambda\|x\|_1,
\]

the active-coordinate KKT condition is

\[
    2A_S^\top r^*
    =
    -\lambda q,
    \qquad
    r^*:=Ax^*-b.
\]

Define

\[
    h:=A_SG^{-1}q,
    \qquad
    r^*:=-\frac{\lambda}{2}h.
\]

Then

\[
    2A_S^\top r^*=-\lambda q.
\]

Finally, set

\[
    \boxed{b=A_Sx_S^*-r^*.}
\]

Because \(G\) is block diagonal with \(2\times2\) blocks, no large inverse is
needed. For each block, use

\[
    \begin{bmatrix}
    1&\rho\\
    \rho&1
    \end{bmatrix}^{-1}
    =
    \frac{1}{1-\rho^2}
    \begin{bmatrix}
    1&-\rho\\
    -\rho&1
    \end{bmatrix}.
\]

The generator must compute \(G^{-1}q\) block by block. It must never explicitly
invert the full matrix \(A_S^\top A_S\).

### 6.3 Inactive columns

For each \(j\notin S\), generate an initial sparse vector
\(\widetilde a_j\) with \(d\) random nonzeros. To satisfy the strict inactive
KKT condition without making the column dense, add one pivot row \(p\) and set

\[
    [a_j]_p
    =
    -\frac{\sum_{\ell\in\operatorname{supp}(\widetilde a_j)}
    [r^*]_\ell[\widetilde a_j]_\ell}{[r^*]_p}.
\]

The pivot must satisfy:

- \(p\notin\operatorname{supp}(\widetilde a_j)\);
- \([r^*]_p\neq0\); and
- \(|[r^*]_p|\) lies in the upper half of the nonzero magnitudes of \(r^*\),
  preventing an excessively large pivot coefficient.

Normalize \(a_j\) after applying the correction. Since the inner product is
zero before normalization, it remains zero afterward:

\[
    a_j^\top r^*=0.
\]

The inactive KKT condition therefore has a fixed normalized margin:

\[
    \frac{\lambda-2|a_j^\top r^*|}{\lambda}=1.
\]

This design isolates active conditioning rather than active-set
identification degeneracy.

If a random column has no pivot satisfying the safety conditions, discard it
and generate another column from the next segment of the same seeded random
stream. Record every retry. Instance generation fails if more than \(0.1\%\)
of inactive columns require a retry.

### 6.4 Optimality and uniqueness

The final instance satisfies

\[
    2A_S^\top(Ax^*-b)+\lambda\operatorname{sign}(x_S^*)=0
\]

and

\[
    \|2A_{S^c}^\top(Ax^*-b)\|_\infty=0<\lambda.
\]

Because \(A_S\) has full column rank and the inactive KKT inequalities are
strict, \(x^*\) is the unique optimal solution.

---

## 7. Data Scaling and Confounding Controls

As \(K\) increases, \(G^{-1}\) amplifies the direction associated with its
smallest eigenvalue. If \(\lambda\) remains fixed, \(\|r^*\|_2\) and
\(\|b\|_2\) can increase with \(K\). This is a genuine consequence of
ill-conditioning, but it could also be criticized as a data-scale confound.
We therefore use two explicitly separated panels.

### 7.1 Panel A: Fixed problem parameters

The main experiment holds fixed:

- \(m,n,s,d\);
- \(x^*\);
- every column norm;
- \(\lambda=10^{-2}\);
- the workload seed; and
- solver parameters and stopping criteria.

Only \(K\) changes. This panel measures the complete numerical effect of
ill-conditioning.

### 7.2 Panel B: Fixed optimal-residual scale

The scale-sensitivity panel enforces

\[
    \|r^*\|_2=R_0=1.
\]

First compute

\[
    h=A_SG^{-1}q,
\]

then set

\[
    \lambda_K=\frac{2R_0}{\|h\|_2},
    \qquad
    r^*=-\frac{\lambda_K}{2}h.
\]

The optimal residual norm is then the same for every \(K\). This panel must
report \(\lambda_K\) and must not be pooled statistically with Panel A. If the
iteration and runtime trends agree across both panels, the evidence that the
effect is caused by active conditioning rather than merely by the scale of
\(b\) is substantially stronger.

### 7.3 Required scale diagnostics

Record the following for every instance:

\[
    \|A\|_2,\quad
    \|A\|_F,\quad
    \|b\|_2,\quad
    \|b\|_\infty,\quad
    \|x^*\|_2,\quad
    \|r^*\|_2,\quad
    \lambda,\quad
    \lambda/\lambda_{\max},
\]

where, under the manuscript's squared-loss convention,

\[
    \lambda_{\max}:=2\|A^\top b\|_\infty.
\]

---

## 8. Experimental Sizes and Random Replication

### 8.1 Workload seeds

Use exactly ten workload seeds in the main experiment:

```text
2026, 2027, 2028, 2029, 2030,
2031, 2032, 2033, 2034, 2035
```

For a fixed seed, different values of \(K\) use:

- the same support;
- the same \(x^*\);
- the same base sparsity pattern;
- the same inactive-column random stream; and
- only a different \(\rho(K)\) and the resulting \(b\).

The observations across \(K\) are therefore paired.

### 8.2 Pilot study

Run the following pilot, which does not enter publication statistics:

| Item | Setting |
|:---|:---|
| \(m\) | \(2\times10^4\) |
| \(n\) | \(1\times10^5\) |
| \(s\) | \(200\) |
| \(d\) | \(20\) |
| \(K\) | \(1,10^2,10^4,10^6\) |
| Seeds | 2026, 2027 |
| Tolerances | \(10^{-3},10^{-6}\) |

The pilot passes only if:

- every generator-side KKT check passes;
- the relative error between the measured and target active condition numbers
  is at most \(10^{-8}\);
- every solver reads the same instance;
- at least two non-PDCS solvers finish instances through \(K=10^4\); and
- the observed runtimes are sufficient to confirm the main-experiment time
  limits.

The pilot may be used to repair the generator, file formats, and measurement
procedure. It must not be used to select favorable values of \(K\), seeds, or
problem sizes on the basis of solver rankings.

### 8.3 Medium-scale all-solver comparison

| Item | Setting |
|:---|:---|
| \(m\) | \(5\times10^4\) |
| \(n\) | \(5\times10^5\) |
| \(s\) | \(500\) |
| \(d\) | \(20\) |
| Target nnz/column | Approximately 20--21 |
| \(K\) | \(1,10^2,10^4,10^6\) |
| Seeds | 2026--2035 |
| Tolerances | \(10^{-3},10^{-6}\) |

Compare:

- cuPDCS;
- PDCS CPU;
- SCS indirect;
- SCS GPU, if the current version supports this SOCP reliably;
- ABIP;
- CuClarabel;
- COPT; and
- MOSEK.

Report COPT and MOSEK both with presolve disabled and with presolve enabled.
The no-presolve results compare the core methods on the original formulation.
The presolved results represent practical performance.

### 8.4 Large-scale scalability experiment

| Item | Setting |
|:---|:---|
| \(m\) | \(5\times10^5\) |
| \(n\) | \(5\times10^6\) |
| \(s\) | \(2000\) |
| \(d\) | \(20\) |
| Target nnz | Approximately \(10^8\) |
| \(K\) | \(1,10^2,10^4,10^6\) |
| Seeds | 2026--2030 |
| Tolerances | \(10^{-3},10^{-6}\) |

Run only solvers whose memory requirements fit the designated hardware.
Solvers that are not run because of known memory requirements must be labeled
`not attempted: memory requirement exceeds hardware`, not as algorithmic
failures.

The primary conclusions come from the ten-instance medium-scale experiment.
The five-instance large-scale panel demonstrates scalability and is not
pooled with the medium-scale results when computing shifted geometric means.

---

## 9. Solver Configuration and Fairness

### 9.1 Hardware

Within each scale panel, run all solvers on the same machine whenever
possible. GPU solvers use the same NVIDIA H100 GPU with 80 GB of memory.
Record for CPU solvers:

- CPU model;
- physical core and hardware-thread counts;
- actual number of solver threads;
- system memory; and
- BLAS implementation and thread configuration.

If some solvers must run on different machines, explicitly label the results
as a cross-hardware comparison. Do not attribute the entire performance
difference to the numerical algorithm.

### 9.2 Time limits

For both medium- and large-scale experiments:

- use a two-hour limit at tolerance \(10^{-3}\);
- use a five-hour limit at tolerance \(10^{-6}\).

These limits match the current Lasso experiments. Exits caused by licensing,
input conversion, or file loading must not be classified as numerical
failures.

### 9.3 Timing boundaries

Record separately:

- instance-generation time;
- serialization time;
- input-parsing time;
- presolve time;
- solver-setup time;
- numerical solve time; and
- end-to-end time.

Use numerical solve time in the main paper table and provide end-to-end time
in the appendix or supplementary material. GPU timing must include all
required synchronization. Initial JIT compilation is excluded from solve
time, but the implementation must be warmed up with a small instance of the
same type before timing.

### 9.4 Initial points

All solvers use their standard cold start. No solver, including cuPDCS, may
receive \(x^*\) or a nearby point as a warm start. The known optimum is used
only for independent verification.

---

## 10. Common Termination and Solution-Quality Verification

The solvers use different internal stopping tests, so their reported statuses
are not directly comparable. Every returned solution must be evaluated by one
independent verifier.

For

\[
    F(x)=\|Ax-b\|_2^2+\lambda\|x\|_1,
\]

define

\[
    r(x)=Ax-b
\]

and the minimum KKT stationarity residual

\[
    e_{\mathrm{stat}}(x)
    =
    \min_{g\in\partial\|x\|_1}
    \|2A^\top r(x)+\lambda g\|_\infty.
\]

Compute it coordinate-wise as

\[
    [e_{\mathrm{stat}}(x)]_j=
    \begin{cases}
    |2[A^\top r(x)]_j+\lambda\operatorname{sign}(x_j)|,
       & |x_j|>\tau_{\mathrm{zero}},\\
    \max\{2|[A^\top r(x)]_j|-\lambda,0\},
       & |x_j|\le\tau_{\mathrm{zero}}.
    \end{cases}
\]

Use

\[
    \tau_{\mathrm{zero}}=10^{-10}
    \max\{1,\|x\|_\infty\}.
\]

The normalized stationarity error is

\[
    \widehat e_{\mathrm{stat}}
    =
    \frac{e_{\mathrm{stat}}}
    {1+2\|A^\top r(x)\|_\infty+\lambda}.
\]

Because \(x^*\) is known, also report

\[
    e_x=
    \frac{\|x-x^*\|_2}{1+\|x^*\|_2},
\]

\[
    e_F=
    \frac{|F(x)-F(x^*)|}{1+|F(x^*)|},
\]

and support recovery:

\[
    \operatorname{precision}
    =
    \frac{|\widehat S\cap S|}{|\widehat S|},
    \qquad
    \operatorname{recall}
    =
    \frac{|\widehat S\cap S|}{|S|}.
\]

SOCP solvers must also pass the common primal-feasibility,
dual-feasibility, and relative-duality-gap tests used elsewhere in the
manuscript. An instance is counted as solved only when all common metrics
meet the target tolerance.

If a solver reports success but the independent verifier rejects its
solution, classify the result as `inaccurate returned solution`, not as a
timeout.

---

## 11. Condition-Number Verification and Preconditioning Diagnostics

### 11.1 Theoretical condition number

For the exactly orthogonal construction,

\[
    K_{\mathrm{theory}}=(1+\rho)/(1-\rho).
\]

### 11.2 Measured condition number

Because \(s\le2000\), form the small matrix

\[
    G=A_S^\top A_S
\]

and use a symmetric eigensolver to compute

\[
    K_{\mathrm{measured}}
    =
    \lambda_{\max}(G)/\lambda_{\min}(G).
\]

For \(K\le10^6\), require

\[
    \frac{|K_{\mathrm{measured}}-K_{\mathrm{theory}}|}
    {K_{\mathrm{theory}}}
    \le 10^{-8}.
\]

Any instance that fails this gate must not enter the solver benchmark.

### 11.3 Before and after PDCS rescaling

Record the row- and column-scaling vectors used by PDCS. Apply the same
transformation to the corresponding active block and compute

\[
    K_{\mathrm{before}},
    \qquad
    K_{\mathrm{after}}.
\]

Run cuPDCS in two configurations:

- production configuration: rescaling enabled;
- diagnostic configuration: rescaling disabled.

The no-rescaling configuration is used only to interpret the effect of
preconditioning and not to represent product performance. Report

\[
    \text{conditioning reduction}
    =
    K_{\mathrm{before}}/K_{\mathrm{after}}.
\]

---

## 12. Outcome Measures and Statistical Analysis

### 12.1 Per-instance records

Record:

- solver status;
- solve time;
- end-to-end time;
- iteration count;
- matrix-vector product count;
- projection count;
- peak CPU/GPU memory;
- independent KKT error;
- primal error;
- dual error;
- relative gap;
- \(e_x\);
- \(e_F\);
- support precision and recall;
- \(K_{\mathrm{before}}\);
- \(K_{\mathrm{after}}\); and
- \(\|b\|_2\), \(\|r^*\|_2\), and \(\lambda/\lambda_{\max}\).

### 12.2 Statistical unit

The workload seed is the statistical unit. Matrix columns, iterations,
kernel launches, and repeated timing measurements are not independent
observations.

### 12.3 Aggregate summaries

For every solver, \(K\), and tolerance, report:

- solved count out of ten;
- shifted geometric mean runtime;
- geometric mean iteration count;
- geometric mean matrix-vector product count;
- median independent KKT error;
- median \(e_x\); and
- maximum peak memory.

For runtime, use the shifted geometric mean

\[
    \operatorname{sgm}(t_1,\ldots,t_N)
    =
    \exp\left(\frac1N\sum_{i=1}^N\log(t_i+s_0)\right)-s_0,
\]

with the fixed shift

\[
    s_0=1\text{ second}.
\]

For runtime profiles, assign failed instances the time limit, while reporting
success rates separately. Do not merge different failure types.

### 12.4 Paired conditioning effects

For each solver and seed, compute the log-ratios relative to \(K=1\):

\[
    \Delta_{\mathrm{time}}(K)
    =
    \log
    \frac{t(K)+1}{t(1)+1},
\]

\[
    \Delta_{\mathrm{iter}}(K)
    =
    \log
    \frac{\operatorname{iter}(K)+1}
         {\operatorname{iter}(1)+1}.
\]

Across the ten paired seeds, report:

- the geometric mean ratio;
- a paired-\(t\) 95% confidence interval; and
- a 10,000-resample seed-block bootstrap confidence interval.

If failure rates are substantial, do not draw unconditional speed
conclusions from the successful subset alone. Report a performance profile
and success rate together.

---

## 13. Tables and Figures

### 13.1 Main-paper table

Use one compact table for each tolerance:

| \(K\) | Solver | Solved | SGM time | GM iterations | Median KKT | Peak memory |
|---:|:---|---:|---:|---:|---:|---:|

If space is limited, retain cuPDCS, SCS indirect, ABIP, CuClarabel, COPT*, and
MOSEK* in the main text and place the complete table in the appendix.

### 13.2 Conditioning-sensitivity figure

Use

\[
    \log_{10}K
\]

on the horizontal axis. Plot separately:

- shifted geometric mean runtime;
- geometric mean iteration count; and
- success rate.

Use logarithmic axes where appropriate and display seed-level points or
paired lines rather than only aggregate means.

### 13.3 Preconditioning diagnostic figure

Plot

\[
    K_{\mathrm{before}}
    \quad\text{against}\quad
    K_{\mathrm{after}}
\]

and the cuPDCS rescaling-on/rescaling-off iteration ratio. This figure
quantifies how much ill-conditioning is mitigated by diagonal rescaling.

### 13.4 Performance profiles

Construct the medium-scale instance set from all \(K\), seeds, and
tolerances. Produce:

- a runtime performance profile; and
- a matrix-vector product performance profile.

Only solutions accepted by the common verifier count as successful. Treat
failures using the time limit and explain this convention in the caption.

---

## 14. Execution Sequence

### Phase 1: Generator validation

- [ ] Implement the sparse correlated-pair generator.
- [ ] Implement the reverse KKT construction of \(x^*\), \(r^*\), and \(b\).
- [ ] Implement sparse pivot correction for inactive columns.
- [ ] Verify column norms, sparsity, support, Gram blocks, and theoretical
      condition numbers.
- [ ] Verify active and inactive KKT residuals at \(x^*\).
- [ ] Verify that paired instances with a common seed differ only in permitted
      fields.
- [ ] Save instance metadata and SHA-256 hashes.

### Phase 2: Pilot

- [ ] Run two seeds, four values of \(K\), and two tolerances.
- [ ] Verify consistent input conversion across all solvers.
- [ ] Compare each solver status with the independent verifier.
- [ ] Confirm the time limits and feasibility of the main problem sizes.
- [ ] Freeze the generator and solver configurations.

### Phase 3: Medium-scale main experiment

- [ ] Generate 40 paired instances.
- [ ] Run every solver at both tolerances.
- [ ] Run the cuPDCS no-rescaling diagnostic.
- [ ] Collect raw logs, solutions, metadata, and system information.
- [ ] Verify every result independently.

### Phase 4: Scale sensitivity

- [ ] Run Panel B with \(\|r^*\|_2=1\).
- [ ] Compare runtime and iteration trends between Panels A and B.
- [ ] Determine whether the conclusions depend on \(\|b\|\).

### Phase 5: Structural-naturalness sensitivity

- [ ] Run the overlapping-support panel.
- [ ] Report the measured rather than theoretical condition number.
- [ ] Compare the trends from the exact-block and overlapping-support
      constructions.

### Phase 6: Large-scale experiment

- [ ] Generate five seeds of large paired instances.
- [ ] Run every solver that fits in memory.
- [ ] Separate not attempted, out-of-memory, timeout, numerical failure, and
      inaccurate-solution outcomes.

### Phase 7: Statistical and manuscript outputs

- [ ] Generate the seed-level master CSV.
- [ ] Generate aggregate tables, paired ratios, and confidence intervals.
- [ ] Generate conditioning-sensitivity plots and performance profiles.
- [ ] Draft the AE-11 response.
- [ ] Add the instance definition and main findings to the Lasso subsection.
- [ ] Add the full generator, solver settings, and result tables to the
      appendix.

---

## 15. Data and Reproducibility Requirements

Create a separate directory for each run containing at least:

```text
run_id/
  environment.txt
  git_head.txt
  git_status.txt
  git_diff.patch
  generator_config.json
  solver_config.json
  instance_manifest.csv
  instance_hashes.sha256
  raw_logs/
  raw_solutions/
  verifier_results.csv
  seed_level_results.csv
  summary_results.csv
  figures/
```

Each row of `instance_manifest.csv` contains:

```text
instance_id,seed,panel,m,n,s,d,K,rho,nnz,
lambda,lambda_max,b_norm2,rstar_norm2,xstar_norm2,
kappa_theory,kappa_measured,kkt_active,kkt_inactive,
matrix_hash,b_hash,xstar_hash
```

Every solver must reference the same `instance_id` and hashes. If different
file formats are required, read the converted file back and verify that its
numerical contents for \(A,b,\lambda\) are equivalent to the source instance.

---

## 16. Failure Taxonomy

Use only the following result labels:

- `solved_verified`;
- `solver_claimed_solved_but_inaccurate`;
- `timeout`;
- `out_of_memory`;
- `numerical_failure`;
- `input_or_conversion_failure`;
- `license_failure`; and
- `not_attempted_hardware_limit`.

Tables must keep these statuses separate. Only `solved_verified` counts as a
successful solve.

---

## 17. Interpretation Rules

Conclusions must be determined by the data:

1. If cuPDCS iterations increase with \(K\) while success remains stable,
   state that cuPDCS exhibits the expected sensitivity to ill-conditioning
   but remains robust over the tested range.
2. If diagonal rescaling substantially reduces \(K_{\mathrm{after}}\) and the
   iteration count, attribute the benefit to the algorithm together with
   rescaling, rather than to the core PDHG iteration alone.
3. If every solver deteriorates with \(K\), emphasize that the new family
   constitutes a genuinely harder benchmark.
4. If cuPDCS fails at \(K=10^6\), report the observed limitation. Do not
   remove that condition-number level after seeing the result.
5. If Panel A deteriorates but Panel B does not, explain that the main driver
   may be data scaling rather than the active condition number alone.
6. If the exact-block and overlapping-support panels show different trends,
   do not make a general robustness claim in the main text.
7. Do not draw conclusions from a single seed or runtime observation.

---

## 18. Structure of the AE-11 Response

The final response should contain four elements:

1. thank the Associate Editor and acknowledge that the original experiment
   used independent uniform nonzeros without an explicitly ill-conditioned
   family;
2. define
   \(\kappa_{\mathrm{active}}=\kappa_2(A_S)^2\) and explain how the sparse
   correlated pairs control it exactly;
3. state that the dimensions, sparsity, support, column norms, random seeds,
   solver settings, and common stopping criteria are held fixed; and
4. summarize the verified effects on cuPDCS runtime, iterations, success rate,
   and rescaling without claiming that the solver is insensitive to
   conditioning.

Before the experiment is complete, do not write specific solver rankings,
speedups, success rates, or maximum solved condition numbers into the response.

After completion, the response can follow this structure:

> Thank you for this suggestion. We added a family of sparse,
> ill-conditioned Lasso instances whose local optimization difficulty is
> controlled through the active-set condition number
> \(\kappa_{\mathrm{active}}=\kappa_2(A_S)^2\). For each correlated column
> pair, the Gram block is
> \(\left[\begin{smallmatrix}1&\rho\\\rho&1\end{smallmatrix}\right]\), with
> \(\rho=(K-1)/(K+1)\), so the target condition number is exactly \(K\).
> Across \(K\in\{1,10^2,10^4,10^6\}\), we hold the dimensions, sparsity,
> support, column norms, random seeds, and solver settings fixed and evaluate
> all returned solutions using a common KKT verifier. We report the verified
> changes in runtime, iteration count, and success rate as the conditioning
> deteriorates. We also report the condition number before and after diagonal
> rescaling, thereby separating sensitivity to conditioning from the benefit
> of preprocessing.

---

## 19. Manuscript Locations to Modify

After the experiments are complete, modify:

1. the Lasso subsection in `PDCS_arxiv.tex`:
   - define the ill-conditioned family;
   - add the main result table or sensitivity figure; and
   - add a restrained summary of the findings;
2. `PDCS_arxiv_appendix_application.tex`:
   - provide the complete sparse-pair generator;
   - give the reverse KKT construction; and
   - report all solver settings and complete tables; and
3. AE-11 in `response_to_reviewers_PDCS.tex`:
   - summarize the new experiment;
   - cite the corresponding manuscript section, table, and figure; and
   - state only findings verified by the common verifier.

---

## 20. Final Acceptance Criteria

The AE-11 experiment is complete only when all of the following hold:

- [ ] Every main instance has a known, unique \(x^*\) that passes the KKT
      verifier.
- [ ] For \(K\le10^6\), the relative error in the measured active condition
      number is at most \(10^{-8}\).
- [ ] Different values of \(K\) under the same seed form valid paired
      instances.
- [ ] The medium-scale study contains ten independent workload seeds.
- [ ] Every returned solution is evaluated by the same independent verifier.
- [ ] Timeout, out-of-memory, numerical failure, and inaccurate-solution
      outcomes remain separate.
- [ ] Runtime, iterations, matrix-vector products, success rate, and KKT
      quality are all reported.
- [ ] Condition numbers before and after PDCS rescaling and the on/off
      diagnostic are reported.
- [ ] Panel B either rules out or reveals a data-scale confound.
- [ ] At least one less artificial overlapping-support panel supports or
      appropriately limits the main conclusion.
- [ ] Raw data, environment information, configuration files, hashes, and
      analysis artifacts are sufficient for reproduction.
- [ ] The reviewer response contains no unsupported robustness or superiority
      claim.
