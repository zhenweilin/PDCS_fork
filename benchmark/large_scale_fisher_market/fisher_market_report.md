# Large-scale Fisher market report

Instances are regenerated deterministically in memory and passed directly to each solver source API. No JuMP model, CBF, JLD2, NPZ, or matrix-data file is used.

## Coverage

- Smoke: 3/3 recorded, 3/3 optimal or almost optimal.
- Formal: 22/45 recorded, 16/45 optimal or almost optimal, 19 skipped by policy.
- Cross-solver numerical digest check: PASS.

## Small correctness case

| Solver | Status | Iterations | Objective | Supply rel. residual | Utility rel. residual | Exp-cone log violation | Wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| cupdcs | OPTIMAL | 6500 | -20.3955 | 4.83753e-07 | 1.11723e-08 | 9.69056e-08 | 23.059 |
| scs_gpu | OPTIMAL | 2400 | -20.3955 | 5.46424e-09 | 2.74017e-09 | 3.55415e-06 | 28.713 |
| cuclarabel | OPTIMAL | 12 | -20.3954 | 4.25116e-07 | 2.37675e-16 | 0 | 13.829 |

- Objective range: `0.000126339`.
- Relative objective range: `6.19443e-06`.

## Formal cases

| Instance | Solver | Status | Iterations | Objective | Generation (s) | Setup (s) | Solve wall (s) |
|---|---|---|---:|---:|---:|---:|---:|
| fisher-m100-n5000-r01 | cupdcs | OPTIMAL | 91000 | -360.696 | 0.26294 | 0.0067599 | 61.058 |
| fisher-m100-n5000-r01 | scs_gpu | OPTIMAL | 9250 | -360.691 | 0.31075 | 0.071899 | 9597.2 |
| fisher-m100-n5000-r01 | cuclarabel | INSUFFICIENT_PROGRESS | 5 | -202.569 | 0.32645 | 32.386 | 20.618 |
| fisher-m100-n5000-r02 | cupdcs | OPTIMAL | 223000 | -391.309 | 0.29831 | 0.0078969 | 124.18 |
| fisher-m100-n5000-r02 | scs_gpu | PENDING |  |  |  |  |  |
| fisher-m100-n5000-r02 | cuclarabel | INSUFFICIENT_PROGRESS | 5 | -221.044 | 0.30217 | 36.367 | 17.931 |
| fisher-m100-n5000-r03 | cupdcs | OPTIMAL | 82000 | -356.405 | 0.26443 | 0.00772 | 59.162 |
| fisher-m100-n5000-r03 | scs_gpu | PENDING |  |  |  |  |  |
| fisher-m100-n5000-r03 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100-n5000-r04 | cupdcs | OPTIMAL | 139000 | -356.998 | 0.26333 | 0.00422 | 78.407 |
| fisher-m100-n5000-r04 | scs_gpu | PENDING |  |  |  |  |  |
| fisher-m100-n5000-r04 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100-n5000-r05 | cupdcs | OPTIMAL | 161000 | -353.852 | 0.69938 | 0.005548 | 88.691 |
| fisher-m100-n5000-r05 | scs_gpu | PENDING |  |  |  |  |  |
| fisher-m100-n5000-r05 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100000-n1000-r01 | cupdcs | OPTIMAL | 118000 | -285068 | 2.0765 | 3.0951 | 2067.5 |
| fisher-m100000-n1000-r01 | scs_gpu | LINEAR_SYSTEM_TIMEOUT | 0 |  | 2.2429 | 120 | 1860 |
| fisher-m100000-n1000-r01 | cuclarabel | SETUP_TIMEOUT | 0 |  | 2.6364 | 18600 | 0 |
| fisher-m100000-n1000-r02 | cupdcs | OPTIMAL | 155000 | -286789 | 2.0714 | 3.157 | 2738.3 |
| fisher-m100000-n1000-r02 | scs_gpu | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100000-n1000-r02 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100000-n1000-r03 | cupdcs | OPTIMAL | 177000 | -285430 | 2.3693 | 3.0152 | 3157.8 |
| fisher-m100000-n1000-r03 | scs_gpu | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100000-n1000-r03 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100000-n1000-r04 | cupdcs | OPTIMAL | 202000 | -286121 | 3.5519 | 4.6501 | 3690.1 |
| fisher-m100000-n1000-r04 | scs_gpu | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100000-n1000-r04 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100000-n1000-r05 | cupdcs | OPTIMAL | 116000 | -286266 | 1.9907 | 3.2951 | 2083.6 |
| fisher-m100000-n1000-r05 | scs_gpu | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m100000-n1000-r05 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m250000-n1000-r01 | cupdcs | OPTIMAL | 165000 | -713108 | 5.1671 | 6.4072 | 9158 |
| fisher-m250000-n1000-r01 | scs_gpu | LINEAR_SYSTEM_TIMEOUT | 0 |  | 5.1307 | 400 | 1800 |
| fisher-m250000-n1000-r01 | cuclarabel | EXCEPTION |  |  | 5.3993 |  |  |
| fisher-m250000-n1000-r02 | cupdcs | OPTIMAL | 167000 | -715627 | 5.5321 | 6.358 | 9270.6 |
| fisher-m250000-n1000-r02 | scs_gpu | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m250000-n1000-r02 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m250000-n1000-r03 | cupdcs | OPTIMAL | 92000 | -714610 | 5.3887 | 6.3812 | 5109.6 |
| fisher-m250000-n1000-r03 | scs_gpu | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m250000-n1000-r03 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m250000-n1000-r04 | cupdcs | OPTIMAL | 167000 | -715748 | 5.4223 | 6.1221 | 9095.2 |
| fisher-m250000-n1000-r04 | scs_gpu | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m250000-n1000-r04 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m250000-n1000-r05 | cupdcs | OPTIMAL | 150000 | -714357 | 5.5979 | 6.1621 | 8305.3 |
| fisher-m250000-n1000-r05 | scs_gpu | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |
| fisher-m250000-n1000-r05 | cuclarabel | SKIPPED_AFTER_SCALE_FAILURE |  |  |  |  |  |

## Manifest

- Julia: 1.12.5
- Seeds: 2026, 2027, 2028, 2029, 2030
- Replicates per formal size: 5
- Storage policy: in_memory_only
- Modeling interface: direct_solver_api
