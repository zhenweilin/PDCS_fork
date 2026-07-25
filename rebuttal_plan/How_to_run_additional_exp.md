# How to run the additional GPU experiments

This runbook executes the additional R1-2 strategy experiments and the R1-3
SOC/exponential-cone utilization experiments on a machine with NVIDIA profiler
access. It is written for a clean checkout and does not depend on results
already present in this repository.

The profiling scripts never overwrite an existing run directory. Choose a new
run label when repeating an experiment.

## Exponential-only quick start

For the primal diagonally rescaled exponential-cone experiments, execute these
sections in order: Section 2 (permissions), Section 3 (environment and build),
Section 7 (unprofiled timing), Sections 8–10 (Nsight Systems), and Section 11
(Nsight Compute).

The minimum correctness-and-timing smoke run is:

```bash
cd /home/zhenwei/PDCS_fork

export CUDA_ROOT=/usr/local/cuda-12.6
export GPU_PHYSICAL=0
export JULIA_BIN="$PWD/.julia-bin/julia"
export JULIA_DEPOT="$PWD/.julia-depot"
export GPU_ARCH=sm_90                    # use sm_80 on A100
export PATH="$CUDA_ROOT/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export JULIA_DEPOT_PATH="$JULIA_DEPOT"
export PDCS_SKIP_GPU_PRECOMPILE=1

CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
benchmark/run_exp_projection.sh \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cuda-home "$CUDA_ROOT" \
  --cuda-runtime local \
  --arch "$GPU_ARCH" \
  --input-distribution heterogeneous \
  --sigma 1 \
  --diagonal-sigma 1 \
  --variants primalDiagonal \
  --strategies gridWise,blockWise,warpWise,threadWise \
  --seed 2026 \
  --smoke \
  --run-label "exp_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
```

The command prints the locations of `raw.csv`, `summary.csv`, and `report.md`.
Every accepted row must have status `PASS`, finite output, and error no larger
than `5e-8`. Do not start profiler runs until this smoke test passes.

## 1. What each measurement is allowed to show

Use three separate measurements:

1. **Unprofiled timing** supplies publication runtime. Do not use profiler
   elapsed time.
2. **Nsight Systems** supplies a sustained CUDA timeline, GPU-metrics samples,
   launch rate, and coarse `nvidia-smi` utilization.
3. **Nsight Compute** supplies kernel-level SM/memory throughput, occupancy,
   eligible warps, branch/source counters, and warp-stall information.

GPU-busy percentage alone is not called efficiency. Interpret it with the
unprofiled runtime and Nsight Compute counters.

## 2. Required software and profiler permission

Required:

- Julia 1.10 or newer;
- an NVIDIA driver compatible with the selected CUDA toolkit;
- CUDA Toolkit 12.5 or 12.6, including `nvcc`, `ncu`, and `nsys`;
- an H100 (`sm_90`) or A100 (`sm_80`);
- permission to access NVIDIA performance counters.

Check the target machine:

```bash
nvidia-smi
/usr/local/cuda-12.6/bin/nvcc --version
/usr/local/cuda-12.6/bin/nsys --version
/usr/local/cuda-12.6/bin/ncu --version
/usr/local/cuda-12.6/bin/nsys profile --gpu-metrics-devices=help
/usr/local/cuda-12.6/bin/ncu --list-sets
```

For an H100 use:

```bash
GPU_ARCH=sm_90
```

For an A100 use:

```bash
GPU_ARCH=sm_80
```

If Nsight Compute reports `ERR_NVGPUCTRPERM`, stop the profiler subset and ask
the machine administrator to enable performance-counter access. Keep the error
log. Do not replace the missing branch or issue metrics with `nvidia-smi`.

## 3. Prepare a clean machine

```bash
cd /home/zhenwei/PDCS_fork

export CUDA_ROOT=/usr/local/cuda-12.6
export GPU_PHYSICAL=0
export JULIA_BIN=/home/zhenwei/PDCS_fork/.julia-bin/julia
export JULIA_DEPOT=/home/zhenwei/PDCS_fork/.julia-depot
export GPU_ARCH=sm_90

export PATH="$CUDA_ROOT/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export JULIA_DEPOT_PATH="$JULIA_DEPOT"
export PDCS_SKIP_GPU_PRECOMPILE=1

"$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate()'
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
```

Confirm that Julia and the shell select the intended GPU:

```bash
CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
"$JULIA_BIN" --project=. -e '
using CUDA
CUDA.functional() || error("CUDA is not functional")
println("device=", CUDA.device())
println("name=", CUDA.name(CUDA.device()))
println("uuid=", CUDA.uuid(CUDA.device()))
println("capability=", CUDA.capability(CUDA.device()))
CUDA.versioninfo()
'

nvidia-smi -i "$GPU_PHYSICAL" \
  --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free \
  --format=csv
```

Do not run other GPU workloads during timing or profiling.

## 4. Validate profiler commands without collecting data

The dry run checks paths, cases, strategies, and final command construction. It
does not require counter permission and creates no result directory.

SOC:

```bash
benchmark/rebuttal/profile_nsys.sh \
  --kind soc \
  --case similar \
  --strategy threadWise \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --duration 30 \
  --run-id dry_soc \
  --dry-run
```

Exponential cone:

```bash
benchmark/rebuttal/profile_nsys.sh \
  --kind exp \
  --case heterogeneous \
  --strategy threadWise \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --sigma 1.0 \
  --diagonal-sigma 1.0 \
  --duration 30 \
  --run-id dry_exp \
  --dry-run
```

## 5. R1-2 SOC count-dimension strategy map

The required primary grid is:

```text
counts     = 3,10,100,1000,10000,100000,1000000
dimensions = 10,32,100,500,2000,10000,50000
sigma      = 2
trials     = 10
```

Run each cone count in a separate directory. The first run rebuilds native
kernels; later runs reuse those exact artifacts:

```bash
RESULT_ROOT="benchmark/results/rebuttal/$(date -u +%Y%m%dT%H%M%SZ)_strategy_map"
first=1

for count in 3 10 100 1000 10000 100000 1000000; do
  extra=()
  if (( first == 0 )); then
    extra+=(--no-build)
  fi

  CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
  benchmark/run_soc_dimension_projection.sh \
    --julia "$JULIA_BIN" \
    --julia-depot "$JULIA_DEPOT" \
    --cuda-home "$CUDA_ROOT" \
    --cuda-runtime local \
    --arch "$GPU_ARCH" \
    --output-dir "$RESULT_ROOT" \
    --run-label "count_${count}" \
    --cone-count "$count" \
    --dimensions 10,32,100,500,2000,10000,50000 \
    --sigma 2.0 \
    --trials 10 \
    --strategies gridWise,blockWise,warpWise,threadWise \
    "${extra[@]}"

  first=0
done
```

For very large cone counts, grid-wise can require millions of sequential
library calls. Record it as skipped rather than silently removing it:

```bash
--strategies blockWise,warpWise,threadWise \
--skip-strategies gridWise
```

Use this option for the counts where grid-wise is known to exceed the
15-second strategy limit. Do not enter an invented runtime for a skipped or
timed-out strategy.

The off-grid validation matrix is:

```text
counts     = 30,300,3000,30000,300000
dimensions = 20,64,256,1000,5000,25000
```

Repeat the same loop with those counts and dimensions, using a different
`RESULT_ROOT`.

## 6. Unprofiled SOC divergence timings

These are the publication timing observations:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_soc_timing"
mkdir -p "benchmark/results/rebuttal/$RUN_ID"

CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
PDCS_SKIP_GPU_PRECOMPILE=1 \
"$JULIA_BIN" --project=. benchmark/rescaled_soc_warp_profile.jl \
  --case uniform,similar,heterogeneous,mixed_grouped,mixed_interleaved \
  --strategy threadWise,warpWise,blockWise \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --hetero-sigma 2.0 \
  --seed 2026 \
  --trials 10 \
  --check \
  --output "benchmark/results/rebuttal/$RUN_ID/cold.csv"

CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
PDCS_SKIP_GPU_PRECOMPILE=1 \
"$JULIA_BIN" --project=. benchmark/rescaled_soc_warp_profile.jl \
  --case uniform,similar,heterogeneous,mixed_grouped,mixed_interleaved \
  --strategy threadWise,warpWise,blockWise \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --hetero-sigma 2.0 \
  --seed 2026 \
  --trials 10 \
  --warm-start \
  --check \
  --output "benchmark/results/rebuttal/$RUN_ID/warm.csv"
```

Repeat timing seeds 2026 through 2035 in distinct output files when preparing
the final paired statistical analysis.

## 7. Unprofiled exponential-cone experiments

Input heterogeneity sweep, holding diagonal sigma fixed:

```bash
for sigma_x in 0.1 0.5 1 2 5 10; do
  CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
  benchmark/run_exp_projection.sh \
    --julia "$JULIA_BIN" \
    --julia-depot "$JULIA_DEPOT" \
    --cuda-home "$CUDA_ROOT" \
    --cuda-runtime local \
    --arch "$GPU_ARCH" \
    --input-distribution heterogeneous \
    --sigma "$sigma_x" \
    --diagonal-sigma 1 \
    --seed 2026 \
    --cone-counts 3,10,100,1000,10000,100000,1000000 \
    --variants primalDiagonal \
    --strategies gridWise,blockWise,warpWise,threadWise \
    --trials 10 \
    --run-label "exp_xsigma_${sigma_x}"
done
```

Diagonal heterogeneity sweep, holding input sigma fixed:

```bash
for sigma_D in 0.1 0.5 1 2 5 10; do
  CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
  benchmark/run_exp_projection.sh \
    --julia "$JULIA_BIN" \
    --julia-depot "$JULIA_DEPOT" \
    --cuda-home "$CUDA_ROOT" \
    --cuda-runtime local \
    --arch "$GPU_ARCH" \
    --input-distribution heterogeneous \
    --sigma 1 \
    --diagonal-sigma "$sigma_D" \
    --seed 2026 \
    --cone-counts 3,10,100,1000,10000,100000,1000000 \
    --variants primalDiagonal \
    --strategies gridWise,blockWise,warpWise,threadWise \
    --trials 10 \
    --run-label "exp_Dsigma_${sigma_D}"
done
```

Every heterogeneous coordinate uses `Normal(0,sigma_x²)`. Every diagonal entry
uses `clamp(abs(Normal(0,sigma_D²)),1e-3,1e3)`. The two random streams are
independent.

### Run the complete controlled exponential matrix

`benchmark/rebuttal/exp_root_profile.jl` is the batch driver for the independent
input/diagonal sigma sweeps and the controlled similar, heterogeneous, grouped,
and interleaved layouts. It starts a fresh Julia process for every
configuration, records every subprocess in `manifest.csv`, and never combines
input and diagonal sigma changes in one sensitivity point.

First run a small validation matrix:

```bash
EXP_SMOKE="benchmark/results/rebuttal/$(date -u +%Y%m%dT%H%M%SZ)_exp_smoke"

CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
PDCS_SKIP_GPU_PRECOMPILE=1 \
"$JULIA_BIN" --project=. benchmark/rebuttal/exp_root_profile.jl \
  --cases similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved \
  --strategies threadWise,warpWise,blockWise \
  --seeds 2026 \
  --sigma-x 0.5,1,2 \
  --sigma-d 0.5,1,2 \
  --cone-count 1024 \
  --trials 2 \
  --output "$EXP_SMOKE"
```

Inspect the smoke run before starting the full experiment:

```bash
column -s, -t < "$EXP_SMOKE/manifest.csv" | less -S
grep -R 'FAIL' "$EXP_SMOKE" || true
```

The full unprofiled timing matrix is:

```bash
EXP_FULL="benchmark/results/rebuttal/$(date -u +%Y%m%dT%H%M%SZ)_exp_full"

CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
PDCS_SKIP_GPU_PRECOMPILE=1 \
"$JULIA_BIN" --project=. benchmark/rebuttal/exp_root_profile.jl \
  --cases similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved \
  --strategies threadWise,warpWise,blockWise \
  --seeds 2026,2027,2028,2029,2030,2031,2032,2033,2034,2035 \
  --sigma-x 0.1,0.5,1,2,5,10 \
  --sigma-d 0.1,0.5,1,2,5,10 \
  --cone-count 1048576 \
  --trials 1 \
  --output "$EXP_FULL"
```

This expands into:

- an input sweep `(sigma_x, sigma_D) = (0.1,1), …, (10,1)`;
- a diagonal sweep `(sigma_x, sigma_D) = (1,0.1), …, (1,10)`;
- five controlled layouts, including a deterministic random permutation;
- three GPU strategies;
- ten paired seeds.

The duplicated `(1,1)` point is automatically removed. Each leaf CSV contains
`input_sigma`, `diagonal_distribution`, `diagonal_sigma`, CPU-reference error,
and status. `manifest.csv` records subprocess failures even if a leaf CSV was
not produced.

Create a combined index:

```bash
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
"$JULIA_BIN" --project=. benchmark/rebuttal/summarize_root_profiles.jl \
  --root "$EXP_FULL" \
  --output "$EXP_FULL/root_profile_summary.csv"
```

For publication runtime, use these unprofiled CSV files. Do not take elapsed
times from Nsight.

### Profile the exponential matrix with Nsight Systems

Run the primary sustained matrix for the three profiler seeds:

```bash
EXP_NSYS="benchmark/results/rebuttal/$(date -u +%Y%m%dT%H%M%SZ)_exp_nsys"

benchmark/rebuttal/run_nsys_matrix.sh \
  --kinds exp \
  --exp-cases similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved \
  --strategies threadWise,warpWise,blockWise \
  --seeds 2026,2027,2028 \
  --gpu "$GPU_PHYSICAL" \
  --gpu-metrics required \
  --duration 30 \
  --cone-count 1048576 \
  --sigma 1 \
  --diagonal-sigma 1 \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --output-root "$EXP_NSYS"
```

Summarize GPU-busy samples and completed launch rates:

```bash
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
"$JULIA_BIN" --project=. benchmark/rebuttal/summarize_nsys.jl \
  --root "$EXP_NSYS" \
  --output "$EXP_NSYS/nsys_summary.csv"
```

To profile the extreme sensitivity points, run separate matrices while holding
the other sigma fixed:

```bash
for sigma_x in 0.1 1 10; do
  benchmark/rebuttal/run_nsys_matrix.sh \
    --kinds exp \
    --exp-cases heterogeneous \
    --strategies threadWise,warpWise,blockWise \
    --seeds 2026,2027,2028 \
    --gpu "$GPU_PHYSICAL" \
    --duration 30 \
    --cone-count 1048576 \
    --sigma "$sigma_x" \
    --diagonal-sigma 1 \
    --cuda-home "$CUDA_ROOT" \
    --julia "$JULIA_BIN" \
    --julia-depot "$JULIA_DEPOT" \
    --output-root "${EXP_NSYS}_xsigma_${sigma_x}"
done

for sigma_D in 0.1 1 10; do
  benchmark/rebuttal/run_nsys_matrix.sh \
    --kinds exp \
    --exp-cases heterogeneous \
    --strategies threadWise,warpWise,blockWise \
    --seeds 2026,2027,2028 \
    --gpu "$GPU_PHYSICAL" \
    --duration 30 \
    --cone-count 1048576 \
    --sigma 1 \
    --diagonal-sigma "$sigma_D" \
    --cuda-home "$CUDA_ROOT" \
    --julia "$JULIA_BIN" \
    --julia-depot "$JULIA_DEPOT" \
    --output-root "${EXP_NSYS}_Dsigma_${sigma_D}"
done
```

### Profile exponential kernels with Nsight Compute

Use one warmed kernel per process:

```bash
EXP_NCU="benchmark/results/rebuttal/$(date -u +%Y%m%dT%H%M%SZ)_exp_ncu"

for case_name in similar heterogeneous mixed_grouped mixed_random mixed_interleaved; do
  for strategy in threadWise warpWise blockWise; do
    for seed in 2026 2027 2028; do
      benchmark/rebuttal/profile_ncu.sh \
        --kind exp \
        --case "$case_name" \
        --strategy "$strategy" \
        --seed "$seed" \
        --gpu "$GPU_PHYSICAL" \
        --cone-count 1048576 \
        --sigma 1 \
        --diagonal-sigma 1 \
        --cuda-home "$CUDA_ROOT" \
        --julia "$JULIA_BIN" \
        --julia-depot "$JULIA_DEPOT" \
        --output-root "$EXP_NCU" \
        --run-id "exp_${case_name}_${strategy}_seed${seed}"
    done
  done
done
```

Before the full profiler loops, replace `1048576` with `1024` and add
`--dry-run` to verify every command. A successful profile directory must
contain `profile.ncu-rep`, `profile_raw.csv`, `exit_status.txt`, and no
`PROFILE_INCOMPLETE.txt`.

## 8. One Nsight Systems profile

Run a 30-second sustained SOC case:

```bash
PROFILE_ROOT="benchmark/results/rebuttal/$(date -u +%Y%m%dT%H%M%SZ)_nsys"

benchmark/rebuttal/profile_nsys.sh \
  --kind soc \
  --case similar \
  --strategy threadWise \
  --seed 2026 \
  --gpu "$GPU_PHYSICAL" \
  --gpu-metrics required \
  --gpu-metrics-frequency 10000 \
  --duration 30 \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --hetero-sigma 2.0 \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --output-root "$PROFILE_ROOT" \
  --run-id soc_similar_threadWise_cold_seed2026
```

Add `--warm-start` for the warm SOC run.

Run a sustained exponential-cone case:

```bash
benchmark/rebuttal/profile_nsys.sh \
  --kind exp \
  --case heterogeneous \
  --strategy threadWise \
  --seed 2026 \
  --gpu "$GPU_PHYSICAL" \
  --gpu-metrics required \
  --duration 30 \
  --cone-count 1048576 \
  --sigma 1.0 \
  --diagonal-sigma 1.0 \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --output-root "$PROFILE_ROOT" \
  --run-id exp_heterogeneous_threadWise_seed2026
```

Each run directory contains:

```text
environment.txt
application.log
nsys.log
exit_status.txt
nvidia_smi.csv
nvidia_smi.log
trace.nsys-rep
trace.sqlite
stats_*.csv
```

The application log records completed launches and elapsed seconds. The
`nvidia_smi.csv` sampling process starts before Nsight and is always stopped by
the script's cleanup trap.

## 9. Full Nsight Systems matrix

The full matrix is intentionally large:

- SOC: 4 cases × 3 strategies × 3 seeds × cold/warm = 72 traces;
- exponential: 4 cases × 3 strategies × 3 seeds = 36 traces.

At 30 seconds per trace, application time alone is 54 minutes; Julia startup
and trace export add overhead.

```bash
MATRIX_ROOT="benchmark/results/rebuttal/$(date -u +%Y%m%dT%H%M%SZ)_nsys_matrix"

benchmark/rebuttal/run_nsys_matrix.sh \
  --kinds soc,exp \
  --soc-cases similar,heterogeneous,mixed_grouped,mixed_interleaved \
  --exp-cases similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved \
  --strategies threadWise,warpWise,blockWise \
  --seeds 2026,2027,2028 \
  --warm-modes cold,warm \
  --gpu "$GPU_PHYSICAL" \
  --gpu-metrics required \
  --duration 30 \
  --cone-count 1048576 \
  --soc-dimension 10 \
  --hetero-sigma 2.0 \
  --sigma 1.0 \
  --diagonal-sigma 1.0 \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --output-root "$MATRIX_ROOT"
```

If storage is limited, first run the primary subset:

```bash
benchmark/rebuttal/run_nsys_matrix.sh \
  --kinds soc,exp \
  --soc-cases similar,heterogeneous,mixed_grouped,mixed_interleaved \
  --exp-cases similar,heterogeneous,mixed_grouped,mixed_random,mixed_interleaved \
  --strategies threadWise \
  --seeds 2026,2027,2028 \
  --warm-modes cold,warm \
  --gpu "$GPU_PHYSICAL" \
  --duration 30 \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --output-root "$MATRIX_ROOT"
```

## 10. Summarize sustained utilization

```bash
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
"$JULIA_BIN" --project=. benchmark/rebuttal/summarize_nsys.jl \
  --root "$MATRIX_ROOT" \
  --output "$MATRIX_ROOT/nsys_summary.csv"
```

The summary contains:

- mean, median, 10th percentile, 90th percentile, and peak GPU busy;
- sample count;
- completed launches;
- elapsed workload-loop time;
- launches per second;
- profiler exit status;
- presence of the `.nsys-rep`;
- whether a GPU-metrics permission failure was detected.

Open a full trace with:

```bash
nsys-ui "$MATRIX_ROOT/soc_similar_threadWise_cold_seed2026/trace.nsys-rep"
```

## 11. Nsight Compute profiles

Profile one warmed kernel per process. Use seeds 2026, 2027, and 2028:

```bash
NCU_ROOT="benchmark/results/rebuttal/$(date -u +%Y%m%dT%H%M%SZ)_ncu"

for seed in 2026 2027 2028; do
  benchmark/rebuttal/profile_ncu.sh \
    --kind soc \
    --case similar \
    --strategy threadWise \
    --seed "$seed" \
    --gpu "$GPU_PHYSICAL" \
    --cone-count 1048576 \
    --cone-dimension 10 \
    --cuda-home "$CUDA_ROOT" \
    --julia "$JULIA_BIN" \
    --julia-depot "$JULIA_DEPOT" \
    --output-root "$NCU_ROOT" \
    --run-id "soc_similar_threadWise_seed${seed}"
done
```

The collector requests:

```text
SpeedOfLight
Occupancy
SchedulerStats
WarpStateStats
SourceCounters
```

It preserves both `profile.ncu-rep` and `profile_raw.csv`. Repeat the command
for heterogeneous, grouped, and interleaved cases and for the other strategies.
Do not interpret Nsight replay passes as independent trials.

## 12. H100 and A100 replication

Use a separate result root for each architecture:

```bash
benchmark/results/rebuttal/<run-id>/h100-sm90/
benchmark/results/rebuttal/<run-id>/a100-sm80/
```

Rebuild with the native architecture before each hardware run:

```bash
# H100
make -C src/pdcs_gpu/cuda rebuild-gpu CUDA_HOME="$CUDA_ROOT" ARCH=sm_90

# A100
make -C src/pdcs_gpu/cuda rebuild-gpu CUDA_HOME="$CUDA_ROOT" ARCH=sm_80
```

Never use an `sm_90` build as the reported A100 measurement. Keep absolute
times separate; compare normalized strategy ratios and qualitative effects.

## 13. Failure handling

- Existing run directory: select a new `--run-id`; do not delete the old run.
- `ERR_NVGPUCTRPERM`: preserve `PROFILE_INCOMPLETE.txt` and rerun on an
  authorized host.
- Missing `trace.nsys-rep`: inspect `exit_status.txt`, `nsys.log`, and
  `application.log`.
- CUDA not functional: compare the Julia CUDA runtime, toolkit path, driver,
  `CUDA_VISIBLE_DEVICES`, and GPU UUID.
- Out of memory: reduce the case only for a diagnostic smoke test. Record the
  primary configuration as memory-excluded rather than silently substituting
  a smaller publication workload.
- Incorrect projection: reject its timing row and inspect the CPU-reference
  error before profiling.

## 14. Reporting checklist

For every manuscript claim retain:

- the unprofiled raw timing CSV;
- the exact seed and case parameters;
- CPU-reference error/status;
- environment and GPU UUID;
- Nsight Systems report and utilization summary;
- Nsight Compute report and raw metrics;
- git commit;
- whether the run is H100 or A100.

Grouped/interleaved comparisons are causal divergence evidence only when they
contain exactly the same cone multiset and differ solely by permutation.
Thread-wise ordering is the primary intra-warp comparison. Describe warp-wise
ordering differences as cross-warp scheduling/workload balance unless
source-level counters demonstrate otherwise.

## 15. Validation performed on the preparation machine

The following checks were completed before handoff:

- `bash -n` passed for `run_exp_projection.sh`, `profile_nsys.sh`,
  `run_nsys_matrix.sh`, and `profile_ncu.sh`;
- Julia parsing passed for `exp_cone_projection.jl`, `exp_warp_profile.jl`,
  `exp_root_profile.jl`, `summarize_nsys.jl`, and
  `summarize_root_profiles.jl`;
- SOC and exponential one-case Nsight Systems commands passed dry-run argument
  expansion;
- the exponential Nsight Systems matrix expanded all five layouts and all
  three strategies correctly in dry-run mode;
- SOC and exponential Nsight Compute commands passed dry-run argument
  expansion with the correct native kernel filters;
- the modified exponential profiler and Nsight summary Julia files passed
  parser checks;
- the summary generator produced a valid empty-result CSV schema;
- `git diff --check` passed.

On 2026-07-25, a real H100 smoke run of
`exp_cone_projection.jl` completed for 3, 100, and 1024 primal diagonal
exponential cones, all four strategies, and two trials per configuration.
All 24 timed rows were `PASS`; the largest sampled CPU/GPU absolute error was
`1.66821297e-08`, below the `5e-8` rejection threshold. The raw CSV also passed
through `summarize_exp_projection.jl`, producing 12 `COMPLETE` summary rows and
the Markdown report.

An actual Nsight capture cannot be validated on this preparation machine:
Nsight Systems fails immediately with `open: Operation not permitted`, and
Nsight Compute hardware counters are restricted. This is an environment
permission failure, not a command-construction fallback. Run Sections 4, 8,
and 11 on the authorized target machine before launching the full matrix.
