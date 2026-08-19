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
