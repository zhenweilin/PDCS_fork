# Dense ill-conditioned Lasso benchmark

This benchmark compares `cuclarabel`, `cuscs`, and `cupdcs` on exactly the
same dense Lasso instances. It uses the objective

```text
minimize  ||A*x - b||_2^2 + lambda*||x||_1.
```

## Matrix family and condition numbers

Every formal matrix is 1000 by 1000, dense, Float64, symmetric, and positive
definite. A shared seeded Gaussian QR factorization produces one orthogonal
matrix `Q`. For each requested condition number `kappa`, the eigenvalues are

```text
sigma_i = 1 - (i - 1)/(999) * (1 - 1/kappa),  i = 1,...,1000,
A       = Q * Diagonal(sigma) * transpose(Q).
```

Thus the eigenvalues (and, because A is SPD, the singular values) are in
descending arithmetic order from 1 to `1/kappa`, and `cond_2(A) = kappa`.
The formal condition-number grid is

```text
10, 100, 1e3, 1e4, 1e5, 1e6.
```

All six cases reuse the same `Q`, sparse reference signal, support, and noise
direction. Only the arithmetic spectrum changes. The source parameters are in
`illcondition_lasso.toml`; change that file rather than copying constants into
solver scripts.

## Guarantee that all solvers see the same problem

`generate_instances.jl` runs once and serializes each `(A, b, xstar, lambda)`
instance under `instances/`. It computes the complete eigenspectrum to verify
the requested condition number and arithmetic spacing. The generated manifest
records SHA-256 hashes of A, b, xstar, and the eigenvalue vector.

Each solver deserializes the same cache and verifies all four hashes before
building its model. Solver scripts never regenerate A. This is stronger than
merely giving the three solvers the same random seed.

## Julia and solver environments

Every job uses the same Julia executable previously validated for the
large-scale Lasso experiments:

```text
benchmark/large_scale_lasso/.tools/julia-1.10.4/bin/julia
```

The version is checked at runtime and is not selected from a Julia version in
this folder's TOML or Manifest. `cupdcs` and `cuclarabel` use
`benchmark/large_scale_lasso/.gpu_solver_env`; `cuscs` uses the companion
`benchmark/large_scale_lasso/.gpu_scs_env` because its published GPU artifact
uses CUDA 11.8. All three use the official JuMP/MathOptInterface API. JuMPRW is
not used.

## Files

- `illcondition_lasso.toml`: authoritative family, condition grid, tolerance,
  and time-limit configuration.
- `ill_conditioned_lasso_cases.jl`: dense SPD generator, verification, hashes,
  and Lasso stationarity metric.
- `generate_instances.jl`: creates the shared serialized instances and
  `instances/generated_instances.toml`.
- `verify_instances.jl`: independently checks cache sizes, hashes, dimension,
  and dense storage before submission.
- `test_illcondition_lasso.jl`: fast regression tests for dense storage,
  arithmetic spacing, condition numbers, and formal configuration.
- `solve_case.jl`: the only model construction and result implementation used
  by all solvers.
- `cuclarabel_test.jl`, `cuscs_test.jl`, `cupdcs_test.jl`: thin compatibility
  wrappers selecting a solver; they do not duplicate model code.
- `run_solver_h100.sbatch`: one aggregate H100 campaign for a selected solver.
- `submit_h100.sh`: regenerates/verifies the common caches and submits the three
  solver campaigns together.

## Running on the cluster

From the repository root:

```bash
benchmark/illcondition_lasso/submit_h100.sh
```

This submits exactly three jobs, one for each solver. Each job requests one
H100 and sequentially runs all six condition numbers. Every case has a
one-hour solver limit, tolerance `1e-6`, and an additional 30-minute driver
grace period; the aggregate job limit is 48 hours. GPU memory/utilization is
sampled every five minutes.

Results are placed under:

```text
benchmark/illcondition_lasso/results/current/cuclarabel/
benchmark/illcondition_lasso/results/current/cuscs/
benchmark/illcondition_lasso/results/current/cupdcs/
```

The TOML result for each case records the verified problem hashes, Julia/GPU
metadata, termination status, timings, and these common comparison criteria:

- relative primal infeasibility of the epigraph and SOC constraints;
- relative Lasso stationarity (the common dual/KKT criterion);
- relative primal-dual objective gap;
- the maximum of the three relative criteria.

No primal vector, dual vector, or slack vector is stored. The vectors exist
only temporarily in memory to compute the scalar criteria.

To run one case manually inside an H100 allocation, select the appropriate
large-scale-Lasso project and call, for example:

```bash
benchmark/large_scale_lasso/.tools/julia-1.10.4/bin/julia \
  --project=benchmark/large_scale_lasso/.gpu_solver_env \
  benchmark/illcondition_lasso/cupdcs_test.jl \
  --manifest benchmark/illcondition_lasso/instances/generated_instances.toml \
  --instance-id kappa-1e4 \
  --output /tmp/cupdcs-kappa-1e4.toml \
  --pdcs-root "$PWD" --tolerance 1e-6 --time-limit 3600
```
