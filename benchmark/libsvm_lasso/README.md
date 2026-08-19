# Compact Lasso SOCP benchmarks

This directory implements the exact compact reformulation

```text
minimize    ||A*x-b||_2^2 + lambda*||x||_1
variables   x in R^n, u in R^n, r in R
subject to  -u <= x <= u
            [(1+r)/sqrt(2), (1-r)/sqrt(2), A*x-b] in SOC^(m+2).
```

The objective is `2r + lambda*sum(u)`. At an optimum,
`u=abs(x)` and `2r=||A*x-b||^2`, so the SOCP is exact.

Unlike the older lifted model, this formulation has no split
`x_plus/x_minus`, residual variable `y`, fixed variable `w=1`, or
residual-defining equality. It has:

- `2n+1` variables;
- one `2n`-dimensional nonnegative block;
- one SOC of dimension `m+2`;
- `m+2n+2` canonical rows;
- `nnz(A)+4n+2` canonical matrix nonzeros.

The direct bulk builder stores `A` once in the SOC mapping and constructs
final CSC arrays without scalar JuMP constraint assembly or an intermediate CBF
file.

## Files

- `realistic_lasso.jl`: LIBSVM streaming loader, compact JuMP builder, and
  parallel direct-CSC builder.
- `run_penalty_sweep.jl`: in-memory build/solve driver for CPU or GPU PDCS.
- `datasets.toml`: source URLs, file names, dimensions, and label convention.
- `test_realistic_lasso.jl`: parser, JuMP, CSC layout, objective, and
  multithreaded-construction tests.
- `test_pdcs_bulk_lasso.jl`: verifies zero-copy CSC handoff to PDCS.
- `test_penalty_sweep.jl`: driver and option tests.

Downloaded archives and result logs are not committed.

## Penalty rules

For LIBSVM data,

```text
lambda_ref = ||A' * b||_inf
lambda     = alpha * lambda_ref
alpha      = {1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000}.
```

Because the loss has no factor `1/2`, the zero solution threshold is
`lambda >= 2||A'b||_inf`. The default `--dataset all` reproduction set is
`news20`, `E2006-log1p`, and `rcv1-train`. Other entries in
`datasets.toml` remain selectable explicitly.

The paper synthetic generator in `../large_scale_lasso/` uses its original
rule `lambda=||A'b||_inf`; it does not multiply by alpha.

## Environment and tests

From the repository root, the existing PDCS environment is sufficient:

```bash
julia --project=. benchmark/libsvm_lasso/test_realistic_lasso.jl
julia --project=. benchmark/libsvm_lasso/test_penalty_sweep.jl
PDCS_SKIP_GPU_PRECOMPILE=1 \
  julia --project=. benchmark/libsvm_lasso/test_pdcs_bulk_lasso.jl
```

A small standalone JuMP environment is also included:

```bash
julia --project=benchmark/libsvm_lasso -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/libsvm_lasso \
  benchmark/libsvm_lasso/test_realistic_lasso.jl
```

## Data placement

Put each compressed archive in a data directory using the exact `file` name
from `datasets.toml`. The loader streams bz2/xz/tar-xz archives directly; it
does not extract a permanent dense copy. For example:

```text
/data/libsvm_lasso/news20.binary.bz2
/data/libsvm_lasso/log1p.E2006.train.bz2
/data/libsvm_lasso/rcv1_train.binary.bz2
```

The catalog contains the official URL and expected compressed byte count.
Every load checks the byte count before parsing.

## Build-only smoke test

This parses a bounded subset, constructs the direct CSC conic form, and does
not require a GPU:

```bash
julia --project=. benchmark/libsvm_lasso/run_penalty_sweep.jl \
  --dataset news20 --mode build --modeling bulk \
  --raw-dir /data/libsvm_lasso --max-rows 1000 --workers 4 \
  --alpha 0.1 --output-dir /tmp/compact_lasso_smoke
```

Use `--modeling jump` only as a small-instance compatibility comparison.

## GPU solve on another machine

Run only after choosing an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PDCS_SKIP_GPU_PRECOMPILE=1 \
julia --project=. benchmark/libsvm_lasso/run_penalty_sweep.jl \
  --dataset news20 --mode pdcs-gpu --modeling bulk \
  --pdcs-root "$PWD" --raw-dir /data/libsvm_lasso \
  --alpha 0.1 --workers 8 --index-type int64 \
  --time-limit 3600 --rel-tol 1e-6 --abs-tol 1e-6 --verbose 2 \
  --output-dir benchmark/libsvm_lasso/results/news20_alpha_0p1
```

Omit `--alpha` to run the full nine-value grid. Use `--dataset all` for
the three-case default set. Run one process per GPU and keep separate output
directories/raw solver logs for every candidate.

Each dataset is loaded and modeled once per process. Updating alpha replaces
only the objective coefficients; constraints and the CSC matrix are reused.
PDCS clears solver state before each solve, avoiding warm-start contamination
between alpha values.
