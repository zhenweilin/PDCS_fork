# R3.5 Fisher-market diagonal-rescaling experiment

This experiment uses the direct Fisher-market exponential-cone formulation

\[
\begin{aligned}
\min_{X,p}\quad &-\sum_i w_i p_i,\\
\sum_i X_{ij} &= b_j,\\
(p_i,1,\sum_j U_{ij}X_{ij})&\in K_{\exp},\\
X&\ge 0.
\end{aligned}
\]

It eliminates the lifted utility variable and its equality. Consequently the
utility coefficients `U_ij` occur directly in the third coordinate row of
each exponential-cone block. The formulation has `m*n + m` variables, `n`
zero-cone rows, `m` exponential cones, and
`m*n + nnz(U) + m` matrix nonzeros.

The older lifted formulation in `../PDCS_fork/benchmark/large_scale_fisher_market`
is retained only as an equivalence baseline. It has `m*n + 2m` variables and
places `U` in `m` separate utility equalities, leaving a fixed all-one
nonzero pattern in the exponential-cone template.

Run the structure and analytical-equivalence tests with the existing Julia
environment (no package installation is needed):

```bash
JULIA_DEPOT_PATH=../PDCS_fork/.julia-depot \
  ../PDCS_fork/.julia-bin/julia --startup-file=no --project=. \
  benchmark/R3.5_fisher_market/test_formulations.jl
```

The main R3.5 comparison holds the direct formulation and all algorithmic
settings fixed and changes only how the Ruiz--Pock--Chambolle scaling factors
are represented inside a structured cone:

- `diagonal`: `rescaling_method=:ruiz_pock_chambolle` and
  `scalar_cone_rescaling=false`, so cone coordinates retain separate diagonal
  factors;
- `scalar_cone`: `rescaling_method=:ruiz_pock_chambolle` and
  `scalar_cone_rescaling=true`, so every coordinate in one cone shares one
  scalar factor.

Both modes use rescaling. `:none` is deliberately excluded from the formal
R3.5 comparison.

## Result

The same-GPU, counterbalanced five-seed experiment on `m=100`, `n=5000`
(500,000 allocation variables) is complete. Coordinate-wise diagonal
rescaling verifies 5/5 rows, while scalar-per-cone verifies 0/5 under the
common independent `2e-5` primal gate. Median solver times are 23.95 and
202.23 seconds; the geometric-mean scalar/diagonal time ratio is 8.64.

See [`REPORT.md`](REPORT.md) for the modeling argument and interpretation, and
[`results/formal_diagonal_scalar/analysis/RESULTS.md`](results/formal_diagonal_scalar/analysis/RESULTS.md)
for the generated numerical table.

Reproduce with the existing Julia environment and no package installation:

```bash
bash benchmark/R3.5_fisher_market/run_experiment.sh --gpus 0,1
```

The time limit remains user-selectable through `--time-limit SECONDS`.
