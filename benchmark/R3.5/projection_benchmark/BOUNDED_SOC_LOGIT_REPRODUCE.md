# Reproducing the diagonal-SOC logit root-search candidate

## Status and scope

This document describes the experimental `bounded_logit48` candidate. It is
compile-time optional and is **not** the production default until same-input,
same-GPU latency and full-solver convergence have been measured. The current
default remains the bounded-`u` safeguarded-Newton implementation.

The candidate is implemented for all four PDCS projection mappings:

- grid-wise: `few_block_proj.cu`;
- block-wise: `moderate_block_proj.cu`;
- warp-wise: `sufficient_block_proj.cu`;
- thread-wise: `massive_block_proj.cu`.

Only generic diagonal-SOC root branches use it. All other cone branches retain
their existing implementations.

## Root coordinate and safeguards

Let `c_i = d_hat_(i+1)^(-2)`. The candidate uses

```text
s = sigmoid(z),  v = sigmoid(-z) = 1 - s,
z in [-700, 700].
```

Both branch residuals are strictly increasing in `z`:

```text
t > 0: F(z) = sum_i a_i^2 [s / (1 + c_i v)]^2 - t^2,
t < 0: F(z) = sum_i a_i^2 [s / (c_i + v)]^2 - t^2.
```

`s` and `v` are computed together from `exp(-abs(z))`, avoiding
cancellation at both sigmoid endpoints. One vector traversal fuses `F`,
`F'`, and `F''`.

Each point is selected in this order:

1. bracketed Halley;
2. bracketed Newton;
3. exponent-space expansion in `z`;
4. Illinois interpolation;
5. bracket midpoint.

The exponent step doubles the magnitude of the logarithmic coordinate, so it
locates an original multiplicative scale in `O(log log range)` queries. The
strict bracket always remains valid. Convergence requires either a sufficiently
narrow `z` bracket, or both a small scaled residual and a small Newton
correction `abs(F/F')`. The correction test prevents false convergence near
a saturated sigmoid endpoint. Recovery uses the stable `(s,v)` pair.

The feature switches are:

```text
PDCS_ENABLE_BOUNDED_SOC_LOGIT_ROOT=1
PDCS_SOC_LOGIT_STEPS=48
```

They are set by the `bounded_logit48` build variant.

## Checkout and build

```bash
git fetch origin
git checkout rebuttal-task-2026-08-14
git pull --ff-only origin rebuttal-task-2026-08-14
cd /home/zhenwei/PDCS_fork

CUDA_HOME=/usr/local/cuda-12.4 ARCH=sm_90 \
  bash benchmark/R3.5/projection_benchmark/build_variants.sh bounded_logit48
```

Override `CUDA_HOME` and `ARCH` for the other machine. Generated `.ptx`
and `.so` files are intentionally ignored. For `sm_90`, the artifact is
written to:

```text
benchmark/R3.5/projection_benchmark/artifacts/bounded_logit48_sm_90/
```

## CPU correctness and root-work test

This requires no GPU. It covers both generic branches, dimensions 1 through
1024, diagonal dynamic range `1e24`, and root logits in `[-30,30]`. It
compares the actual projected point with a 256-step bracketed reference.

```bash
PDCS_Z_NEWTON_STEPS=48 \
  julia benchmark/R3.5/projection_benchmark/validate_bounded_soc_root.jl \
  20000 20260819
```

Use `PDCS_WARM_MODE=1,...,6` for cold, exact, 0.02-perturbed,
0.5-perturbed, 4.0-perturbed, and reversed warm starts:

```bash
for mode in 1 2 3 4 5 6; do
  PDCS_Z_NEWTON_STEPS=48 PDCS_WARM_MODE=$mode \
    julia benchmark/R3.5/projection_benchmark/validate_bounded_soc_root.jl \
    10000 20260819
done
```

The committed candidate produced zero projection failures in each 10,000-case
class. Counts are fused root-oracle/vector traversals, not scalar operations:

| Warm class | Plain logit Newton mean | Candidate mean | Median | p95 | Fallback |
|---|---:|---:|---:|---:|---:|
| cold | 57.6232 | 30.1997 | 25 | 90 | 12.610% |
| exact | 4.3152 | 2.8051 | 1 | 19 | 0.030% |
| 0.02 perturbation | 24.7646 | 15.9883 | 15 | 33 | 0.000% |
| 0.5 perturbation | 31.9216 | 15.3789 | 14 | 34 | 0.000% |
| 4.0 perturbation | 42.1089 | 17.8203 | 16 | 37 | 0.150% |
| reversed | 53.7115 | 45.3782 | 37 | 99 | 32.010% |

This extreme synthetic distribution is not a timing prediction. On the existing
real `table5` warm-start profile, the current bounded-`u` default uses
8.8382 function evaluations and 9.8492 vector reductions per root. GPU A/B
decides whether the candidate's additional derivative arithmetic is worthwhile.

## Same-GPU projection A/B

Build baseline and candidate without replacing solver artifacts:

```bash
bash benchmark/R3.5/projection_benchmark/build_variants.sh newton8_current
bash benchmark/R3.5/projection_benchmark/build_variants.sh bounded_logit48
```

Wait until all selected GPUs are idle. This launcher assigns exactly one process
to each of four GPUs. Run `smoke` first:

```bash
repo=$PWD
base_artifact=$repo/benchmark/R3.5/projection_benchmark/artifacts/newton8_current_sm_90
candidate_artifact=$repo/benchmark/R3.5/projection_benchmark/artifacts/bounded_logit48_sm_90
result_root=$repo/benchmark/R3.5/projection_benchmark/results/cross_machine_20260819

SAMPLES=11 PROFILE=true bash benchmark/R3.5/projection_benchmark/run_four_gpu.sh \
  newton8_current smoke "$base_artifact" "$result_root/baseline_smoke"
SAMPLES=11 PROFILE=true bash benchmark/R3.5/projection_benchmark/run_four_gpu.sh \
  bounded_logit48 smoke "$candidate_artifact" "$result_root/candidate_smoke"
```

If both smoke runs have no correctness failure, replace `smoke` with `quick`,
then `full`. Run the baseline and candidate on the same physical GPUs with no
other GPU jobs. Do not compare rows from different devices.

```bash
SAMPLES=11 PROFILE=true bash benchmark/R3.5/projection_benchmark/run_four_gpu.sh \
  newton8_current full "$base_artifact" "$result_root/baseline_full"
SAMPLES=11 PROFILE=true bash benchmark/R3.5/projection_benchmark/run_four_gpu.sh \
  bounded_logit48 full "$candidate_artifact" "$result_root/candidate_full"

python3 benchmark/R3.5/projection_benchmark/compare_variants.py \
  --baseline "$result_root"/baseline_full/shard*.csv \
  --candidate "$result_root"/candidate_full/shard*.csv \
  --baseline-label newton8_current \
  --candidate-label bounded_logit48 \
  --output-dir "$result_root/comparison_full"
```

Review `comparison_full/summary.md` and `comparison_full/comparison.csv`.
Acceptance first requires all applicable candidate rows to pass correctness,
then compares latency only for same-case/same-GPU pairs. Oracle and fallback
counters explain results but do not replace latency.

## One-GPU A/B

Create `result_root` first. Run both artifacts serially on the same idle GPU:

```bash
mkdir -p "$result_root"

CUDA_VISIBLE_DEVICES=0 PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$base_artifact" \
  julia --project=. benchmark/R3.5/projection_benchmark/run_projection_stress.jl \
  --device=0 --shard-index=0 --shard-count=1 --tier=full \
  --variant=newton8_current --heterogeneous=true --profile=true --samples=11 \
  --output="$result_root/baseline_one_gpu.csv" \
  >"$result_root/baseline_one_gpu.raw.log" 2>&1

CUDA_VISIBLE_DEVICES=0 PDCS_CUDA_PROJECTION_ARTIFACT_DIR="$candidate_artifact" \
  julia --project=. benchmark/R3.5/projection_benchmark/run_projection_stress.jl \
  --device=0 --shard-index=0 --shard-count=1 --tier=full \
  --variant=bounded_logit48 --heterogeneous=true --profile=true --samples=11 \
  --output="$result_root/candidate_one_gpu.csv" \
  >"$result_root/candidate_one_gpu.raw.log" 2>&1
```

Keep every CSV and raw log. Give reruns a new directory suffix rather than
overwriting an interrupted or failed run.

## Full-solver gate

Projection-only tests are necessary but insufficient. Before making this the
default, run the difficult real instances and the 62-case `represent_data`
campaign with the same tolerance and time limit as the current default. Compare
status, residuals, objective, PDHG iterations, projection time, and total time.
Reject a candidate that turns an optimal case into a timeout/error even if its
isolated projection kernel is faster.
