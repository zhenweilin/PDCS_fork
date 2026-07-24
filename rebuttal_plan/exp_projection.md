# Exponential-cone GPU projection experiments

## Scope

These scripts test **primal diagonal exponential-cone projection** (projection code 27) independently of the SOC experiments. An exponential cone always has dimension 3, so there is no meaningful cone-dimension sweep. The experiments vary:

- number of exponential cones;
- positive diagonal scalings and number of cones;
- grid-wise, block-wise, warp-wise, and thread-wise execution;
- similarity of root-finding workloads;
- branch-class ordering within hardware warps; and
- GPU utilization, occupancy, active warps, and warp stalls.

The default publication benchmark is `primalDiagonal`. Each cone receives three deterministic positive log-normal diagonal entries. The CPU oracle is `PDCS_CPU.exponent_proj_diagonal!`. Other variants remain available for implementation diagnostics, but the reviewer experiment and profiler use only primal diagonal projection.

## Files

- `benchmark/exp_cone_projection.jl`: count/strategy/variant benchmark with sampled CPU-reference validation.
- `benchmark/summarize_exp_projection.jl`: produces an aggregate CSV and Markdown report.
- `benchmark/run_exp_projection.sh`: configures a consistent local CUDA runtime, rebuilds kernels, runs the benchmark, and summarizes it.
- `benchmark/exp_warp_profile.jl`: controlled root-finding and branch-layout experiment.
- `benchmark/profile_exp_projection.sh`: unprofiled timings plus Nsight Compute counter collection.

## Implementation consistency check

Projection type 27 passes the user-facing diagonal `D` to all four strategies.
The block-, warp-, and thread-wise kernels invert its three entries internally
because their exponential projection device routine consumes `D⁻¹`. The
grid-wise wrapper now performs the same conversion before launching
`exponent_proj_diagonal_kernel`. This is important: passing `D` directly gave
plausible output but disagreed with both the CPU implementation and the other
three GPU strategies.

The post-fix H100 smoke result is retained at
`benchmark/results/exp_projection/exp_script_smoke/primal_diagonal_after_grid_fix.csv`.
It contains 16/16 passing timed rows for 3 and 32 cones (two trials, four
strategies), with maximum absolute error `5.22e-10`.

## 1. Smoke test on a new machine

For an H100 with CUDA 12.5:

```bash
cd /home/zhenwei/PDCS_fork

CUDA_VISIBLE_DEVICES=0 \
benchmark/run_exp_projection.sh \
  --julia "$(command -v julia)" \
  --julia-depot /home/zhenwei/.julia \
  --cuda-home /usr/local/cuda-12.5 \
  --cuda-runtime local \
  --arch sm_90 \
  --smoke \
  --run-label exp_smoke
```

This runs cone counts 3, 100, and 10,000 with two trials. It performs a CUDA.jl cuBLAS preflight before compiling or benchmarking. Every timed raw CSV row contains `max_error`; correctness is never recorded only on the first row.

## 2. Full exponential-cone count sweep

```bash
CUDA_VISIBLE_DEVICES=0 \
benchmark/run_exp_projection.sh \
  --julia "$(command -v julia)" \
  --julia-depot /home/zhenwei/.julia \
  --cuda-home /usr/local/cuda-12.5 \
  --cuda-runtime local \
  --arch sm_90 \
  --cone-counts 3,10,100,1000,10000,100000,1000000 \
  --variants primalDiagonal \
  --input-distribution heterogeneous \
  --sigma 1.0 \
  --strategies gridWise,blockWise,warpWise,threadWise \
  --trials 10 \
  --max-gridwise-cones 10000 \
  --run-label exp_full
```

The grid-wise method launches projections sequentially and is deliberately skipped above 10,000 cones by default. Change `--max-gridwise-cones` only when the expected runtime is acceptable.

`--input-distribution` is the controlled random-vector parameter:

- `similar`: every cone is a small deterministic perturbation of the same
  root-finding input, with diagonal scales tightly concentrated around one;
- `heterogeneous`: every vector coordinate is independently sampled from
  `Normal(0, sigma²)`, where `sigma` is set by `--sigma`.

For both modes, diagonal entries use the same narrow log-normal law,
`exp(0.02 * Normal(0,1))`. Thus a sigma sweep changes only the input-vector
distribution. Both modes use `--seed` (default 2026), so repeated runs generate
the same vectors and scales.

Use, for example, the following controlled heterogeneity levels:

```bash
for sigma in 0.1 0.5 1.0 2.0 5.0 10.0; do
  CUDA_VISIBLE_DEVICES=0 benchmark/run_exp_projection.sh \
    --cuda-home /usr/local/cuda-12.5 \
    --input-distribution heterogeneous \
    --sigma "$sigma" \
    --seed 2026 \
    --run-label "exp_sigma_${sigma}"
done
```

## 3. Controlled divergence cases

The profiler harness defines four cases:

- `similar`: nearly identical root-search inputs;
- `heterogeneous`: broad but bounded input and diagonal-scale variation;
- `mixed_grouped`: four branch classes stored in four contiguous groups; and
- `mixed_interleaved`: the exact same cone multiset interleaved by class.

The grouped/interleaved pair contains exactly the same input vectors **and diagonal scales** and changes only their ordering. For thread-wise projection, adjacent threads in the interleaved case take different projection paths, while grouped warps contain the same class. A slowdown for interleaved data is controlled evidence of branch-divergence sensitivity. For warp-wise projection, scalar decisions are coherent within each warp, although different warps can still have workload imbalance.

Run one timing case manually:

```bash
CUDA_VISIBLE_DEVICES=0 \
JULIA_DEPOT_PATH=/home/zhenwei/.julia \
PDCS_SKIP_GPU_PRECOMPILE=1 \
julia --project=. benchmark/exp_warp_profile.jl \
  --case mixed_interleaved \
  --strategy threadWise \
  --cone-count 1048576 \
  --trials 20 \
  --output benchmark/results/exp_projection/mixed_interleaved_threadWise.csv
```

## 4. GPU utilization and Nsight Compute

First verify that performance-counter access is enabled. See `GPU_PROFILING_PERMISSION.md`. Then run:

```bash
CUDA_VISIBLE_DEVICES=0 \
benchmark/profile_exp_projection.sh \
  --gpu 0 \
  --julia "$(command -v julia)" \
  --julia-depot /home/zhenwei/.julia \
  --cuda-home /usr/local/cuda-12.5 \
  --cone-count 1048576 \
  --cases similar,heterogeneous,mixed_grouped,mixed_interleaved \
  --strategies threadWise,warpWise \
  --trials 20 \
  --output-dir benchmark/results/exp_projection/profile_h100
```

The script collects these Nsight sections:

- `SpeedOfLight`: SM and memory throughput as percentages of peak;
- `Occupancy`: theoretical and achieved occupancy;
- `SchedulerStats`: eligible and issued warps;
- `WarpStateStats`: cycles per issued instruction and warp-stall reasons; and
- `SourceCounters`: branch/source-level behavior where supported.

Use the unprofiled timing CSV files for runtime comparisons. Nsight Compute replays kernels, so profiler elapsed time is not publication-quality runtime.

For the reviewer's utilization question, report kernel-level SM throughput and achieved occupancy. Do not substitute coarse `nvidia-smi` polling for these counters. Compare similar with heterogeneous to measure convergence imbalance, and compare grouped with interleaved to measure ordering-induced divergence. A useful table is:

| Strategy | Case | Median time | SM throughput | Achieved occupancy | Eligible warps/cycle | Branch/stall metric |
|---|---|---:|---:|---:|---:|---:|

If Nsight reports `ERR_NVGPUCTRPERM`, the scripts are correct but the driver denies hardware counters. Enable access as documented in `GPU_PROFILING_PERMISSION.md` and rerun; do not invent or infer an 80--90% utilization value.

## 5. Correctness interpretation

`exp_cone_projection.jl` samples up to 256 cones uniformly from every generated collection and compares every timed primal diagonal GPU result with `PDCS_CPU.exponent_proj_diagonal!`. The diagonal entries are strictly positive. The default pass threshold is maximum absolute error `5e-8`. Inputs and scales are deterministically generated from seed 2026 and include feasible interior, feasible boundary, root-search, and general random cases. Every timed raw CSV row records its own `max_error` and `status`.
