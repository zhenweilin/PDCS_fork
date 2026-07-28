# Reproducing the Ill-Conditioned Lasso Experiments

This document reproduces the paired-column, sparse ill-conditioned Lasso experiment in `rebuttal_plan/ill_conditioned_lasso.md`.  The runner creates each instance from a seed, saves a serialized cache and a manifest with SHA-256 hashes, runs JuMP formulations, and independently checks the original Lasso KKT conditions.

## Scope and solver policy

MOSEK is a correctness reference only.  It is run automatically only in the small `smoke` and medium-size `reference` profiles.  The `pilot` and `medium` profiles omit MOSEK by default because their models are too large for a useful reference-solver run.  They instead run PDCS, SCS, and CuClarabel.  Passing `--solvers ... ,mosek` explicitly overrides this safety default, so do that only for a deliberately reduced instance.

| Profile | `m` | `n` | active support | nonzeros per column | seeds by default | MOSEK default |
|:--|--:|--:|--:|--:|:--|:--|
| `smoke` | 1,000 | 5,000 | 40 | 10 | 2026 | yes |
| `reference` | 5,000 | 25,000 | 100 | 20 | 2026 | yes |
| `pilot` | 20,000 | 100,000 | 200 | 20 | 2026, 2027 | no |
| `medium` | 50,000 | 500,000 | 500 | 20 | 2026--2035 | no |

All profiles use `K = 1, 10^2, 10^4, 10^6`, with the exact active Gram condition number recorded in `instance_manifest.csv`.

## One-time setup

Run from the repository root:

```bash
cd /home/zhenwei/PDCS_fork
export JULIA_DEPOT_PATH="$PWD/.julia-depot"
export PDCS_SKIP_GPU_PRECOMPILE=1
./.julia-bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'
./.julia-bin/julia --project=. benchmark/install_lasso_solvers.jl
```

`install_lasso_solvers.jl` installs SCS and CuClarabel.  It needs outbound access to Julia's package server and GitHub; retry it on the target machine if the local node has no DNS/network access.

For the supplied MOSEK license:

```bash
export MOSEKLM_LICENSE_FILE="$PWD/mosek_lic/mosek.lic"
./.julia-bin/julia --project=. -e 'using JuMP, MosekTools; m=Model(MosekTools.Optimizer); @variable(m, x >= 1); @objective(m, Min, x); optimize!(m); println(termination_status(m))'
```

The expected final line is `OPTIMAL`.  Do not copy the license to the result directory or commit it.

## GPU selection

On a machine managed by a scheduler, first request a GPU through that scheduler.  Within the allocation, the runner's default `--gpu auto` selects the GPU with the most free memory among GPUs whose instantaneous utilization is at most 10%:

```bash
./benchmark/rebuttal/select_idle_gpu.sh
./benchmark/run_ill_conditioned_lasso.sh --profile pilot --phase generate --gpu auto
```

The selector is a scheduling hint, not a reservation.  To bind a known allocated GPU, use `--gpu 7`.  The selected device is recorded in `environment.txt` and passed via `CUDA_VISIBLE_DEVICES`.  If `nvidia-smi` or the driver is unavailable, CPU solvers can still run; the GPU attempt is recorded as a hardware-limit result instead of being interpreted as a solver failure.

## Commands

First run the deterministic generator unit test:

```bash
./benchmark/run_ill_conditioned_lasso.sh --phase test --profile smoke
```

Run the small reference check.  This is the MOSEK correctness confirmation command and deliberately uses a stringent `1e-8` tolerance:

```bash
./benchmark/run_ill_conditioned_lasso.sh \
  --profile smoke --phase all --run-id smoke_reference_$(date -u +%Y%m%dT%H%M%SZ) \
  --solvers pdcs_cpu,mosek --tolerances 1e-8 --gpu auto
```

Run the medium-size reference confirmation before the production-scale experiment.  This remains small enough for a reference solver; it is not the 20,000-by-100,000 pilot:

```bash
./benchmark/run_ill_conditioned_lasso.sh \
  --profile reference --phase all --run-id reference_$(date -u +%Y%m%dT%H%M%SZ) \
  --solvers pdcs_cpu,mosek --tolerances 1e-8 --gpu auto
```

Run the non-reference pilot (two seeds, no MOSEK):

```bash
./benchmark/run_ill_conditioned_lasso.sh \
  --profile pilot --phase all --run-id pilot_$(date -u +%Y%m%dT%H%M%SZ) \
  --solvers pdcs_cpu,pdcs_gpu,scs,cuclarabel --tolerances 1e-3,1e-6 --gpu auto
```

Run the main ten-seed medium experiment (again no MOSEK):

```bash
./benchmark/run_ill_conditioned_lasso.sh \
  --profile medium --phase all --run-id medium_$(date -u +%Y%m%dT%H%M%SZ) \
  --solvers pdcs_cpu,pdcs_gpu,scs,cuclarabel --tolerances 1e-3,1e-6 --gpu auto
```

Use a new `--run-id` for every full run.  The generator refuses to overwrite an existing instance cache, preserving the seed-to-instance mapping.  Preview all commands without creating files with `--dry-run`.

## Outputs and acceptance checks

For a run directory `R=benchmark/results/rebuttal/ill_conditioned_lasso/<run-id>`:

- `environment.txt` records software/GPU visibility and chosen GPU.
- `source_snapshot/` records the source used for the run.
- `instances/*.jls` are the reproducible generated instances.
- `instance_manifest.csv` contains seed, exact measured condition number, KKT construction residuals, and hashes.
- `seed_level_results.csv` is the raw per-solver/per-seed output.
- `ill_conditioned_lasso_report.md` contains only completed numerical outcomes and marks an outcome `solved_verified` only when it passes the independent original-Lasso KKT and solution-error checks.

Recreate the summary after an interrupted run:

```bash
R=benchmark/results/rebuttal/ill_conditioned_lasso/<run-id>
JULIA_DEPOT_PATH="$PWD/.julia-depot" ./.julia-bin/julia --project=. \
  benchmark/analyze_ill_conditioned_lasso.jl --root "$R"
```

Do not make a performance or correctness claim from a row marked `solver_claimed_solved_but_inaccurate`, `timeout`, `input_or_conversion_failure`, or `not_attempted_hardware_limit`.
