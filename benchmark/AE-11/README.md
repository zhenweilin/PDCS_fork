# AE-11: simple condition-number robustness experiment

The official AE-11 experiment now asks one narrow question: does the default
cuPDCS solver return independently verified solutions when only the nonzero
spectral condition number of a fixed Lasso family is changed?

## Official grid

- Problem: `min ||A*x-b||_2^2 + lambda*||x||_1`.
- Dimensions: `m=50,000`, `n=250,000`, about 16 million sparse nonzeros.
- Condition numbers: `K = 1, 1e2, 1e4, 1e6, 1e8`.
- Seeds: `2026:2030`, giving 25 runs.
- Penalty: fixed practical ratio `lambda = 1.5||A'*b||_inf` (`alpha=1.5`).
  The zero-solution threshold is `2||A'*b||_inf`, so this is not a trivial
  zero-solution experiment; all returned solutions must also have positive
  `||x||_1`.
- Solver: cuPDCS with the production diagonal rescaling enabled.
- Solver tolerance: `2e-7` for every run.
- Independent acceptance: normalized KKT at most `1e-6`, normalized dual
  infeasibility at most `1e-6`, and relative duality gap at most `1e-5`.
- Time limit: one hour per run.

For each seed, dimensions, sparse pattern, singular vectors, and `b` are
fixed across all five values of `K`; only the nonzero singular-value spectrum
changes. Matrices and solutions are constructed and used only in memory. The
package saves scalar TOML metrics, hashes, logs, and summaries, never instance
data or solution vectors.

## Reproduce

All Julia environments are reused from `../PDCS_fork`; the driver runs in
offline package mode and does not install anything.

```bash
cd benchmark/AE-11
./run_robustness.sh --gpus 0,1
```

Useful options are `--gpus`, `--time-limit`, `--output`, and `--force`.
Defaults reproduce the frozen protocol in
[`robustness_experiment.toml`](robustness_experiment.toml). On success the
driver audits exactly 25 logical rows and writes:

- `results/robustness_alpha1p5_1e-6/analysis/RESULTS.md`;
- `results/robustness_alpha1p5_1e-6/analysis/summary.csv`;
- `results/robustness_alpha1p5_1e-6/analysis/summary.toml`.

The official implementation consists of
[`AE11Common.jl`](AE11Common.jl), [`run_cupdcs.jl`](run_cupdcs.jl),
[`run_robustness.sh`](run_robustness.sh), and
[`analyze_robustness.jl`](analyze_robustness.jl). The older multi-solver,
on/off, strict-reference, and scale-grid scripts/results are retained only as
an exploratory archive and are excluded from the official robustness claim.
