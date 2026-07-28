# Ill-Conditioned Lasso: Initial Validation Report

## Status

The deterministic sparse paired-column generator and the JuMP SOCP formulation are implemented.  This report records the completed local smoke validation; it is not a full performance comparison.

The experiment follows the requested reference-solver policy: MOSEK is used only for the small `smoke` and medium-size `reference` correctness profiles.  It is excluded by default from the 20,000-by-100,000 pilot and 50,000-by-500,000 production-medium profiles.  Those profiles use PDCS, SCS, and CuClarabel when their packages and a functional GPU are available.

## Validation completed on 2026-07-28

The generator unit test passed **27/27** checks.  The smoke instance has `m=1000`, `n=5000`, support size 40, 10 nonzeros per column, seed 2026, and exact active condition numbers:

| requested K | measured active condition number |
|--:|--:|
| 1 | 1.0 |
| 100 | 100.00000000000018 |
| 10,000 | 10,000.000000014994 |
| 1,000,000 | 1,000,000.0001721674 |

The reverse-KKT construction residuals in the corresponding manifest are at floating-point roundoff level.  The complete cached instances, hashes, and raw log are under `benchmark/results/rebuttal/ill_conditioned_lasso/smoke_runner_20260728/`.

## Small reference check

At `K=1` and tolerance `1e-8`, MOSEK returned `OPTIMAL` and passed the independent verifier:

| solver | K | normalized KKT residual | relative solution error | objective error | result |
|:--|--:|--:|--:|--:|:--|
| MOSEK | 1 | 1.8841e-9 | 3.0145e-9 | 1.9803e-11 | solved_verified |
| PDCS CPU | 1 | 6.9159e-4 | 1.7648e-4 | 1.1718e-7 | solved_verified at `1e-3` |
| PDCS CPU | 100 | 2.4936e-4 | 2.2789e-3 | 3.7330e-7 | solved_verified at `1e-3` |

This confirms the formulation, generated data, and PDCS solution interpretation on a small independent reference instance.  It does **not** establish correctness at the larger condition numbers.

## Observed stress cases

For this smoke seed, PDCS CPU returned `OPTIMAL` but did not satisfy the independent solution-error criterion at `K=10^4` (relative solution error 0.2064) or `K=10^6` (0.4521).  MOSEK at `K=10^6` likewise returned `OPTIMAL` but did not pass the independent KKT/solution check (normalized KKT residual 9.8049e-3; relative solution error 8.2251e-3).  These rows are retained as `solver_claimed_solved_but_inaccurate`; they are not counted as successful solves.

## cuPDCS GPU check

On GPU 6 (NVIDIA H100 80GB), CUDA.jl reported `CUDA.functional=true` and the existing deprecated-`use_accelerated` GPU compatibility regression passed **6/6** in 32.5 seconds.  This confirms that cuPDCS can load and execute CUDA kernels on this machine; profiling permissions are not required for this test.

The new Lasso SOCP path was then tested directly with `pdcs_gpu`, with no profiling tools involved:

| instance | solver limit | measured wall seconds | terminal status | normalized KKT | relative solution error | conclusion |
|:--|--:|--:|:--|--:|--:|:--|
| smoke: 1,000 x 5,000, K=1 | 120 s | 169.78 s | TIME_LIMIT | 3.2527e-3 | 5.2576e-3 | not verified |
| tiny: 200 x 1,000, K=1 | 120 s | 170.07 s | TIME_LIMIT | 5.9110e-3 | 6.7202e-3 | not verified |

During the solves, GPU 6 reached roughly 56--60% compute utilization with approximately 0.75--0.85 GB allocated, confirming actual GPU execution.  Both rows retain the available incumbent diagnostics but are now correctly classified as `timeout`; a time-limited incumbent is not presented as a completed but inaccurate solve.  Thus, GPU availability is confirmed, while the current cuPDCS Lasso/JuMP formulation does not meet the requested `1e-3` verification threshold within the configured limit.  Do not include these rows as successful cuPDCS solves until that convergence issue is addressed.

## Current machine limitations

- SCS and CuClarabel were not installed locally because the node could not resolve the Julia package server/GitHub.  The installer and complete reproduction commands are provided in `rebuttal_plan/how_to_run_ill_conditioned_lasso.md`.

Therefore no solver ranking or medium-scale conclusion is made from this machine.  Run the pilot and ten-seed medium commands from the reproduction guide on the target GPU machine; MOSEK will remain excluded from those large profiles by default.
