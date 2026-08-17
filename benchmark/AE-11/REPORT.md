# AE-11 reviewer response: condition-number robustness

## Question addressed

The reviewer asked for evidence on an ill-conditioned or structurally
different family. We use a controlled wide Lasso family and change only

\[
\kappa_+(A)=\sigma_{\max}(A)/\sigma_{\min}^{+}(A)
\in\{1,10^2,10^4,10^6,10^8\}.
\]

This is the nonzero spectral condition number; `A'*A` is singular because
`m<n`. For every seed, all five matrices share dimensions, CSC pattern,
singular vectors, response vector, penalty policy, and solver settings.

## Simplified protocol

We run only the default cuPDCS configuration with diagonal rescaling enabled.
There is no solver comparison and no rescaling on/off ablation in the official
AE-11 result. Those questions are outside this reviewer experiment.

The 25 instances use `m=50,000`, `n=250,000`, five seeds, fixed `alpha=1.5`,
Float64 arithmetic, cold starts, and a one-hour limit per run. The objective is

\[
\|Ax-b\|_2^2+\lambda\|x\|_1,
\qquad \lambda=1.5\|A^\top b\|_\infty.
\]

Since the zero-solution threshold is `2||A'*b||_inf`, `alpha=1.5` is not a
trivial zero-solution setting. The final audit additionally requires every
reported solution to have finite, positive `||x||_1`.

The solver is asked for `2e-7`. Success is decided only by a separate Float64
verifier using normalized KKT `<=1e-6`, normalized dual infeasibility
`<=1e-6`, and relative duality gap `<=1e-5`. A solver-reported `optimal`
status alone is insufficient.

## Result

The frozen numerical table is generated at
[`results/robustness_alpha1p5_1e-6/analysis/RESULTS.md`](results/robustness_alpha1p5_1e-6/analysis/RESULTS.md),
with machine-readable values in `summary.csv` and `summary.toml` beside it.

The exact configured-grid audit passes: all 25 logical rows are present,
unique, Float64, hash-consistent, nonzero, and independently verified.

| Condition number | Verified | Median solve (s) | Maximum solve (s) | Median iterations | Maximum KKT | Maximum relative gap |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | 5/5 | 6.06 | 6.34 | 2,000 | `1.46e-11` | `1.69e-10` |
| `1e2` | 5/5 | 6.08 | 6.48 | 2,000 | `1.37e-9` | `1.67e-10` |
| `1e4` | 5/5 | 14.38 | 19.78 | 6,000 | `2.77e-10` | `2.43e-11` |
| `1e6` | 5/5 | 19.21 | 24.63 | 8,000 | `2.84e-7` | `8.36e-9` |
| `1e8` | 5/5 | 30.77 | 34.77 | 14,000 | `3.16e-7` | `1.60e-8` |

Across all 25 runs, the largest independently measured KKT residual is
`3.16e-7`, the largest relative gap is `1.60e-8`, and the smallest solution
`l1` norm is `0.114`. The measured condition numbers have at most `1.79e-15`
relative construction error. No run times out or returns NaN/Inf; the longest
solve takes 34.77 seconds, far below the one-hour cap.

Therefore, for this fixed practical Lasso family, cuPDCS remains reliably
solvable at `1e-6` verification accuracy as the controlled nonzero spectral
condition number increases from `1` to `1e8`.

## Interpretation boundary

This experiment supports a robustness statement for this fixed, practically
regularized Lasso family. It does not claim that every penalty choice is easy,
that runtime is independent of conditioning, or that `K=1e8` has the same
numerical margin as `K=1`. The claim is deliberately limited to whether the
same cuPDCS configuration solves all configured instances within one hour and
passes the common `1e-6` certificate.

Conditioning still increases work: median solve time rises from 6.06 to 30.77
seconds and median iterations from 2,000 to 14,000 between `K=1` and `K=1e8`.
The robustness claim is therefore about verified completion, not invariant
runtime.

## NaN diagnostic during tolerance selection

Exploratory runs at `alpha=1` produced numerical instability: asking the
solver for `1e-7` produced two
`numerical_error_nan` exits. This did not come from the independent verifier
or imply a failed explicit Lasso/cone projection: returned primal vectors had
no nonfinite entries. The solver source emits exit code 8 specifically when
its internal relative gap becomes NaN, after the adaptive iteration state has
already become unstable. With a fixed `2e-7` target, 24/25 `alpha=1` rows
verified and one `K=1e8` row timed out after its adaptive weight collapsed to
about `5.7e-90`. These observations motivated freezing the still-nonzero
practical ratio `alpha=1.5` before the final full-grid run. The official grid
uses this one penalty and one internal target for every row; it has no
seed-specific retry or tolerance. All exploratory rows are retained outside
the official result directory and are not counted.

## Reproducibility and storage

The driver uses the existing Julia binary and depot under `../PDCS_fork` with
offline package mode. It regenerates each deterministic sparse matrix in
memory, solves it directly, verifies the returned vector, writes only scalar
TOML metrics/hashes, and releases the instance. No matrices, model files, or
solution vectors are saved.
