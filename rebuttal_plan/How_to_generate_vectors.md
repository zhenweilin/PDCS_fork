# How to generate projection test vectors

This document specifies the random test-case generation used by the GPU
projection benchmarks. All vectors use `Float64`. A recorded seed makes every
experiment reproducible.

## Distribution convention

In this repository, `sigma` means the **standard deviation**. Therefore,

```text
x = sigma * randn(...)
```

generates

```text
x_i ~ Normal(0, sigma²).
```

For example, `--sigma 2.0` produces coordinates with mean zero and variance
four.

## 1. Ordinary second-order cones

An ordinary SOC with full dimension `d` is stored as

```text
(t, u₁, ..., u_{d-1}).
```

For `m` cones, allocate one contiguous vector of length `m*d`. Every coordinate,
including each leading coordinate `t`, is generated independently:

```text
x_i ~ Normal(0, sigma²).
```

No feasible points, boundary points, or other hand-constructed cases are mixed
into the ordinary SOC dimension/count experiment.

Equivalent Julia code:

```julia
using Random

seed = 2026
sigma = 1.0
cone_count = 100
cone_dimension = 500

rng = MersenneTwister(seed)
x = sigma .* randn(rng, Float64, cone_count * cone_dimension)

cone(j) = @view x[(j-1)*cone_dimension+1:j*cone_dimension]
```

The main benchmark generates the vector directly on the GPU:

```julia
CUDA.seed!(seed)
randn!(x_gpu)
x_gpu .*= sigma
```

Run it with:

```bash
cd /home/zhenwei/PDCS_fork

CUDA_VISIBLE_DEVICES=7 \
benchmark/run_soc_dimension_projection.sh \
  --cuda-home /usr/local/cuda-12.6 \
  --arch sm_90 \
  --cone-count 100 \
  --dimensions 10,50,100,500,1000,2000,5000 \
  --sigma 2.0 \
  --trials 10 \
  --run-label soc_sigma_2
```

For dimension-sweep trial `r`, the main script uses:

```text
seed = base_seed + 10000 * dimension_index + r,
base_seed = 2026.
```

For a fixed dimension and trial, every GPU strategy receives the same random
vector. Random generation and host/device preparation are outside the measured
projection interval. The raw CSV records `input_sigma`.

## 2. Diagonally rescaled SOC cases

The rescaled-SOC profiler uses projection type 22. It generates the tail
coordinates and diagonal entries from independent random streams:

```text
Zx_i, Zd_i ~ Normal(0,1),
u_i = sigma_x * Zx_i,
D_i = clamp(abs(sigma_D * Zd_i), 1e-3, 1e3).
```

The first diagonal entry of each cone is fixed to one:

```text
D₁ = 1.
```

The absolute value creates a half-normal diagonal distribution. The clamp keeps
all entries positive and avoids singular or extreme rescaling.

The leading coordinate `t` is deliberately constructed from the scaled tail
norm. This profiler is intended to exercise controlled feasible, polar, and
root-finding branches; it is not the completely unstructured ordinary-SOC
benchmark.

For the `heterogeneous` case, the current implementation uses:

```text
sigma_x = sigma_D = --hetero-sigma.
```

Run a heterogeneous case with:

```bash
CUDA_VISIBLE_DEVICES=7 \
JULIA_DEPOT_PATH=/home/zhenwei/PDCS_fork/.julia-depot \
PDCS_SKIP_GPU_PRECOMPILE=1 \
./.julia-bin/julia --project=. benchmark/rescaled_soc_warp_profile.jl \
  --case heterogeneous \
  --hetero-sigma 2.0 \
  --cone-count 1024 \
  --cone-dimension 10 \
  --strategy warpWise \
  --seed 2026 \
  --trials 10
```

Use the same seed, cone count, dimension, and strategy when comparing different
`--hetero-sigma` values.

## 3. Primal diagonal exponential cones

Every exponential cone has exactly three coordinates. For `m` cones, the input
vector and diagonal vector both have length `3m`.

### Input vector

For the heterogeneous random case:

```text
x_i ~ Normal(0, sigma_x²).
```

Equivalent Julia code (projection type 27 has code `27`):

```julia
input_seed = seed + cone_count + 27
rng_x = MersenneTwister(input_seed)
x = sigma_x .* randn(rng_x, Float64, 3 * cone_count)
```

Set `sigma_x` with:

```bash
--sigma 2.0
```

### Diagonal scaling

Diagonal entries use an independent half-normal distribution:

```text
D_i = clamp(abs(sigma_D * Z_i), 1e-3, 1e3),
Z_i ~ Normal(0,1).
```

Equivalent Julia code:

```julia
diagonal_seed = seed + cone_count + 91_337
rng_D = MersenneTwister(diagonal_seed)
D = clamp.(
    abs.(sigma_D .* randn(rng_D, Float64, 3 * cone_count)),
    1e-3,
    1e3,
)
```

Set `sigma_D` independently with:

```bash
--diagonal-sigma 0.5
```

Cone `j` uses:

```julia
indices = 3j-2:3j
x_j = @view x[indices]
D_j = @view D[indices]
```

The GPU projection type is 27 (`primalDiagonal`). The CPU correctness reference
is:

```julia
PDCS_CPU.exponent_proj_diagonal!(copy(x_j), D_j)
```

Run the benchmark with:

```bash
CUDA_VISIBLE_DEVICES=7 \
benchmark/run_exp_projection.sh \
  --cuda-home /usr/local/cuda-12.6 \
  --arch sm_90 \
  --input-distribution heterogeneous \
  --sigma 2.0 \
  --diagonal-sigma 0.5 \
  --seed 2026 \
  --cone-counts 3,10,100,1000,10000 \
  --variants primalDiagonal \
  --strategies gridWise,blockWise,warpWise,threadWise \
  --trials 10 \
  --run-label exp_x2_D0p5
```

The raw and summary CSV files record:

```text
input_distribution
input_sigma
diagonal_distribution
diagonal_sigma
seed
```

## 4. Controlled sigma sweeps

Change only one parameter at a time.

Input-vector sweep:

```bash
for sigma_x in 0.1 0.5 1.0 2.0 5.0 10.0; do
  CUDA_VISIBLE_DEVICES=7 benchmark/run_exp_projection.sh \
    --cuda-home /usr/local/cuda-12.6 \
    --input-distribution heterogeneous \
    --sigma "$sigma_x" \
    --diagonal-sigma 1.0 \
    --seed 2026 \
    --run-label "exp_xsigma_${sigma_x}"
done
```

Diagonal-scaling sweep:

```bash
for sigma_D in 0.1 0.5 1.0 2.0 5.0 10.0; do
  CUDA_VISIBLE_DEVICES=7 benchmark/run_exp_projection.sh \
    --cuda-home /usr/local/cuda-12.6 \
    --input-distribution heterogeneous \
    --sigma 1.0 \
    --diagonal-sigma "$sigma_D" \
    --seed 2026 \
    --run-label "exp_Dsigma_${sigma_D}"
done
```

Keep the cone counts, seed, strategy list, and trial count identical across the
sweep. Never combine runs with different sigma values unless the sigma columns
are retained in the table or plot.

## 5. Reproducibility checklist

Record the following for every result:

- Git commit;
- GPU name and UUID;
- NVIDIA driver;
- CUDA toolkit and CUDA.jl runtime;
- Julia version;
- cone type, count, and dimension;
- input sigma;
- diagonal distribution and diagonal sigma, when applicable;
- base seed and trial seed;
- projection strategy;
- trial count;
- maximum error against the CPU reference.

Random vectors should be generated before timing. All strategies being compared
must receive the same vector for a given seed and configuration.
