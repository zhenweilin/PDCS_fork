# `batch` supplied old log: GPU provenance check

## Updated conclusion after recovering the Gitee history

The supplied file
`results/formal_1h/cases/batch/oldlog.log` cannot be reproduced by any committed
historical GPU implementation recovered from `pdhg_clp`.

The decisive reason is the algorithm branch, not the CUDA version:

- the supplied log prints the non-aggressive `resolving` path;
- every recovered GPU revision defines only
  `pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving_aggressive!`;
- the non-aggressive GPU function is referenced but is never defined in the
  full Gitee history;
- therefore the log came from the CPU path or an uncommitted local GPU change.

Per the user's instruction, the new historical comparison tests GPU code only.
This supplied log is retained as provenance evidence but is excluded from GPU
performance counts. The complete GPU recovery analysis is in
`GPU_HISTORY_RECOVERY.md`.

## What the supplied log reports

| Field | Value |
|---|---:|
| Time limit | 1200 s |
| Threads | 14 |
| Status | `optimal` |
| Iterations | 4,162,000 |
| Solver time | about 1071.67 s |
| Projection time | about 1034.14 s |
| `l_inf` primal residual | `3.5754e-12` |
| `l_inf` dual residual | `1.5431e-7` |
| Relative gap | `9.9880e-7` |
| Restarts | 91 (mean 32, ergodic 58) |

The data file itself is unchanged. The two available copies have the same
SHA-256:

```text
6e8c06dc078709814ae931acf8dc68c94978a3ec24064fe0b2a92f2163c957a3
```

This rules out a changed `batch.cbf.gz` as the explanation.

## Why the initially archived `external/PDCS-main` was insufficient

`external/PDCS-main` and the public `PDCS_fork_test` history omit several
features visible in the supplied log, including the historical GPU power-method
path. The private Gitee repository at `external/pdhg_clp` contains that older
GPU development history. It was recovered using `external/gitee.txt` through
the non-interactive askpass helper without exposing the credential.

Relevant recovered commits are:

- `e61feea`: early hard-case-targeted version, but it uses `l2` termination and
  contains a later-confirmed diagonal EXP projection bug;
- `3432afc`: final old GPU solver, with EXP fixes and `l_inf` termination;
- `e095ae5` (`paper_release`): same solver source as `3432afc` except Makefile,
  plus the formal scripts.

## Why the old solved count was not comparable

Two independent differences can make old tables appear to solve more cases:

1. Early revisions declared `OPTIMAL` from `l2` residuals. In the recovered
   `e61feea` GPU test, `batch` and `batchs101006m` both self-reported
   `OPTIMAL`, but failed the current three-`l_inf` verification.
2. The paper-release `1e-6` GPU script allowed 18,000 seconds per case, while
   the current 2,100-case comparison allows 3,600 seconds.

In addition, repeated old-GPU runs are not bitwise deterministic. Floating
GPU reductions perturb adaptive-step values slightly; restart decisions then
amplify those differences. A single old raw log is therefore insufficient for
a robust same-configuration comparison.

## Current `batch` reference

At tolerance `1e-6` and a 3,600-second limit:

| Current method | Status | Max verification metric |
|---|---:|---:|
| No-Halpern Full | `TIME_LIMIT` | gap `8.8874e-6` |
| Inline-Halpern Full | `OPTIMAL` | dual residual `5.7482e-7` |
| Archived GPU `3432afc` | `OPTIMAL` at 2399.12 s | gap `9.4878e-7` |

Thus `batch` is not unsolved by every current method. The observed failure is
specific to No-Halpern Full under the one-hour budget. The committed old
aggressive GPU code does solve `batch` within one hour, but it does not
reproduce the supplied non-aggressive log's approximately 1071.67-second
trajectory: at 1200 seconds its separate preliminary run still had maximum
metric `1.7204e-5`, and the formal run solved at 2399.12 seconds.

## Reproducible GPU-only commands

Recover and compile the final old GPU version:

```bash
cd /home/zhenwei/PDCS_fork
external/old_can_solve_scripts/recover_pdhg_clp_gpu.sh \
  --credential-file external/gitee.txt \
  --commit 3432afc69f96891168c13c926d08cdaacd5b0e41 \
  --worktree external/old_can_solve_scripts/worktrees/pdhg_clp_gpu_3432afc \
  --nvcc /usr/local/cuda-12.6/bin/nvcc \
  --arch sm_90
```

Run the six GPU cases with the current one-hour budget:

```bash
external/old_can_solve_scripts/run_all.sh \
  --old-source-dir external/old_can_solve_scripts/worktrees/pdhg_clp_gpu_3432afc \
  --input-root "$HOME/PDCS_CBLIB" \
  --gpus 0,1,2,3,4,5 \
  --time-limit 3600 \
  --tolerance 1e-6 \
  --print-frequency 2000 \
  --julia-threads 14 \
  --use-aggressive true \
  --model-loader moi
```
