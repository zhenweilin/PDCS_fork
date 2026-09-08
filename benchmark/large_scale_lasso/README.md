# Synthetic compact Lasso generator

This directory reproduces the paper's synthetic Lasso family with the compact
SOCP formulation documented in `../libsvm_lasso/README.md`.

For every manifest entry it deterministically generates sparse `A`, a
half-sparse signal, and `b=A*x+1e-6`. The paper penalty is

```text
lambda = ||A' * b||_inf
```

with no alpha multiplier. The CBF model has variables `(x,u,r)`, the two
epigraph inequalities `u-x>=0` and `u+x>=0`, and one SOC containing
`A*x-b` directly. Thus the instance-related matrix is not hidden in a
residual-defining equality.

The committed `lasso_table5.toml` contains five deterministic replicates of
each `(m,n)` pair:

```text
(10000,100000), (70000,700000), (400000,7000000),
(700000,7000000), (750000,7500000)
```

all at density `1e-4`.

## Verify the compact builder

From the repository root:

```bash
julia --project=. benchmark/large_scale_lasso/test_compact_lasso.jl
```

## Regenerate the manifest

The manifest records the generator, project, and environment hashes:

```bash
julia --project=. benchmark/large_scale_lasso/large_scale_lasso.jl \
  generate-config --preset table5 --master-seed 20260728 \
  --config benchmark/large_scale_lasso/lasso_table5.toml
```

## Generate one compact CBF instance

```bash
julia --project=. benchmark/large_scale_lasso/large_scale_lasso.jl \
  build-data --config benchmark/large_scale_lasso/lasso_table5.toml \
  --instance table5-m10000-n100000-r01 \
  --output-dir /data/compact_lasso_cbf \
  --results /data/compact_lasso_cbf/results.toml
```

Omit `--instance` to generate all 25 cases. Outputs and raw results are not
tracked by Git. Use `--allow-environment-mismatch` only when intentionally
changing Julia/package versions; it disables the reproducibility hash gate.

For direct in-memory PDCS experiments, use
`../libsvm_lasso/realistic_lasso.jl:build_lasso_conic_data` with a
`LassoData` object. That path avoids CBF serialization and generic JuMP/MOI
copying.

## H100 solver campaign

The GPU campaign tests cuPDCS, cuSCS, and CuClarabel on H100 GPUs. Prepare its
pinned Julia 1.10.4 environment once from the repository root:

```bash
benchmark/large_scale_lasso/prepare_gpu_solver_env.sh
```

Submit a one-replicate, smallest-scale smoke test before a full campaign:

```bash
cluster_scripts/submit_lasso_gpu_solvers.sh smoke
cluster_scripts/submit_lasso_gpu_solvers.sh full
```

To stage each full solver task behind the matching smoke-array task:

```bash
cluster_scripts/submit_lasso_gpu_solvers.sh after-smoke SMOKE_ARRAY_JOB_ID
```

Each Slurm array task requests one whole `gpu:h100`. Cases run in increasing
`(m,n)` order. If any replicate at a scale fails validation, reaches a time
limit, or raises an exception, cuSCS and CuClarabel finish that scale and skip
all larger scales. cuPDCS always attempts every configured scale and replicate.
The full defaults are five replicates, a one-hour per-case solver limit, and
exact `1e-6` solver and independent-validation tolerances. A cuPDCS case is
accepted only when its exported point also has an internal relative KKT maximum
at or below `1e-6`. Results are written below the ignored `results/` directory;
each case retains a TOML summary and raw log, including an explicit failed-case
record when a solver process crashes before it can write one.

After all jobs finish, verify H100 provenance, five-replicate coverage, and the
solver-specific early-stop rule, while producing a combined TSV:

```bash
benchmark/large_scale_lasso/.tools/julia-1.10.4/bin/julia \
  benchmark/large_scale_lasso/verify_gpu_campaign.jl \
  --cupdcs results/full-cupdcs/cupdcs \
  --cuscs results/full-external/cuscs \
  --cuclarabel results/full-external/cuclarabel \
  --output results/verified --require-complete true
```
