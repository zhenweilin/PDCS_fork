# Reproducing the SOC projection experiments for R1-2

This guide reproduces the current GPU results on another Linux/NVIDIA machine.
Run every command from the repository root. Two complementary experiments are
included:

1. Fix the cone count at 3 or 100 and vary the individual cone dimension.
2. Fix total dimension at `1.2e9`, vary the cone count, and therefore vary the
   individual dimension inversely as in Figure 3.

Both experiments compare four implementations:

| Paper name | Internal name | Work assigned per cone |
|:---|:---|:---|
| Grid-wise | `few` | One grid/cuBLAS sequence |
| Block-wise | `moderate` | One CUDA block |
| Warp-wise | `sufficient` | One CUDA warp |
| Thread-wise | `massive` | One CUDA thread |

The master report is `rebuttal_plan/cone_projectioin_results.md`. The spelling
of `projectioin` is intentional because it is the requested output path.

## 1. Scripts and generated artifacts

- `benchmark/run_soc_dimension_projection.sh`: environment discovery, native
  compilation, benchmark execution, summary generation, and optional plotting.
- `benchmark/soc_dimension_projection.jl`: random-vector generation,
  correctness checking, and synchronized timing.
- `benchmark/summarize_soc_dimension.jl`: per-run summary CSV and figure.
- `benchmark/compare_soc_dimension_cases.jl`: combines the 3- and 100-cone
  dimension sweeps.
- `benchmark/summarize_soc_paper_count_sweep.jl`: combines the fixed-total
  count sweep and generates its four-curve figure.
- `benchmark/assemble_soc_rebuttal_report.jl`: assembles both component reports
  without overwriting either one.
- `rebuttal_plan/cone_dimension_results.md`: fixed-count component report.
- `rebuttal_plan/cone_count_results.md`: fixed-total component report.
- `rebuttal_plan/cone_projectioin_results.md`: master report containing both.
- `rebuttal_plan/figures/soc_count_fixed_total.{tex,pdf,png}`: Figure 3-style
  fixed-total figure.

Every benchmark run writes into
`benchmark/results/soc_dimension/<run-label>/`:

```text
raw.csv
summary.csv
environment.txt
benchmark.log
summary.log
report.md
```

Individual runs never write the master report. Only the final assembly command
does so.

## 2. Target-machine requirements

Required:

- Linux and an NVIDIA GPU;
- a working NVIDIA driver (`nvidia-smi` succeeds);
- CUDA toolkit 12.6 or 13.2 with `nvcc` and cuBLAS;
- Julia compatible with this project;
- GNU Make and a CUDA-compatible host compiler.

For PDF/PNG figures, also install LaTeX with PGFPlots and Poppler's
`pdftoppm`.

The fixed-total experiment allocates vectors containing `1.2e9` Float64
entries. Use a GPU with at least approximately 40 GB free memory; an 80 GB H100
was used for the current results. The 100–120 million-cone endpoints also need
several gigabytes of host memory for block metadata.

`nvidia-smi` and `nvcc --version` may show different CUDA versions. This is
normal: `nvidia-smi` describes driver capability, while `--cuda-home` selects
the toolkit used to compile PDCS's native kernels. The raw CSV records the
driver, build toolkit, CUDA.jl runtime, Julia version, GPU UUID, compute
capability, and Git commit separately.

## 3. Configure the machine

Choose the GPU and toolkit. These examples use GPU index 7 and CUDA 12.6;
change them for the target machine:

```bash
export CUDA_VISIBLE_DEVICES=7
export CUDA_HOME=/usr/local/cuda-12.6
export JULIA_DEPOT_PATH="$PWD/.julia-depot"
```

For CUDA 13.2 instead:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
```

The runner locates Julia in this order:

1. `--julia PATH`;
2. `PDCS_JULIA`;
3. `./.julia-bin/julia`;
4. `julia` from `PATH`.

Check the environment without changing files:

```bash
benchmark/run_soc_dimension_projection.sh \
  --cuda-home "$CUDA_HOME" \
  --dry-run
```

Verify the driver, compiler, and Julia CUDA runtime:

```bash
nvidia-smi
"$CUDA_HOME/bin/nvcc" --version

JULIA_DEPOT_PATH="$JULIA_DEPOT_PATH" \
  ./.julia-bin/julia --project=. -e '
    using CUDA
    println("CUDA functional: ", CUDA.functional())
    CUDA.versioninfo()
  '
```

`CUDA functional: true` is required.

## 4. Compile for the target GPU

The first runner invocation detects the selected GPU's compute capability and
rebuilds all PTX kernels plus `libfew_block_proj.so`. Start with a small smoke
test:

```bash
benchmark/run_soc_dimension_projection.sh \
  --cuda-home "$CUDA_HOME" \
  --smoke \
  --run-label smoke_target_gpu
```

After this succeeds, subsequent commands use `--no-build`. Do not use
`--no-build` after changing the GPU architecture or CUDA toolkit.

To override detection explicitly:

```bash
benchmark/run_soc_dimension_projection.sh \
  --cuda-home "$CUDA_HOME" \
  --arch sm_90 \
  --smoke \
  --run-label smoke_explicit_arch
```

Check native linkage when necessary:

```bash
ldd src/pdcs_gpu/cuda/libfew_block_proj.so | grep -E 'cublas|not found'
```

There must be no `not found` entry.

## 5. Experiment A: fix 3 and 100 cones

This experiment uses the common full SOC dimensions

```text
10, 50, 100, 500, 1000, 2000, 5000, 10000, 25000, 50000
```

and ten independent trials per dimension and strategy.

Run the 3-cone case:

```bash
benchmark/run_soc_dimension_projection.sh \
  --cuda-home "$CUDA_HOME" \
  --no-build --no-plot \
  --cone-count 3 \
  --dimensions 10,50,100,500,1000,2000,5000,10000,25000,50000 \
  --trials 10 \
  --run-label gpu7_3cones_four_strategies_rerun
```

Run the matching 100-cone case on the same GPU:

```bash
benchmark/run_soc_dimension_projection.sh \
  --cuda-home "$CUDA_HOME" \
  --no-build \
  --cone-count 100 \
  --dimensions 10,50,100,500,1000,2000,5000,10000,25000,50000 \
  --trials 10 \
  --run-label gpu7_100cones_four_strategies
```

The expected raw files are:

```text
benchmark/results/soc_dimension/gpu7_3cones_four_strategies_rerun/raw.csv
benchmark/results/soc_dimension/gpu7_100cones_four_strategies/raw.csv
```

Each raw CSV must have 401 lines: one header plus
`10 dimensions × 4 strategies × 10 trials = 400` `PASS` rows.

Generate the component report:

```bash
JULIA_DEPOT_PATH="$JULIA_DEPOT_PATH" \
./.julia-bin/julia --project=. benchmark/compare_soc_dimension_cases.jl \
  --raw-3 benchmark/results/soc_dimension/gpu7_3cones_four_strategies_rerun/raw.csv \
  --summary-3 benchmark/results/soc_dimension/gpu7_3cones_four_strategies_rerun/summary.csv \
  --raw-100 benchmark/results/soc_dimension/gpu7_100cones_four_strategies/raw.csv \
  --summary-100 benchmark/results/soc_dimension/gpu7_100cones_four_strategies/summary.csv \
  --output rebuttal_plan/cone_dimension_results.md
```

Current H100 result to compare—not to copy—is:

- 3 cones: warp-wise wins at dimensions 10–50, block-wise at 100–10,000,
  and grid-wise at 25,000–50,000.
- 100 cones: warp-wise wins at dimensions 10–50 and block-wise at
  100–50,000.

The exact crossover on another GPU may differ. Report the new measurements.

## 6. Experiment B: fixed total dimension of 1.2e9

Use these exact `(cone count, full dimension per cone)` pairs:

```text
(10,        120000000)
(100,        12000000)
(1000,        1200000)
(10000,        120000)
(100000,        12000)
(1000000,        1200)
(10000000,        120)
(100000000,        12)
(120000000,        10)
```

Every pair has exactly `1.2e9` scalar dimensions.

### 6.1 Counts from 10 through 1,000,000

Run all four strategies. The loop below creates the exact run labels expected
by the count-summary script:

```bash
for spec in \
  10:120000000 \
  100:12000000 \
  1000:1200000 \
  10000:120000 \
  100000:12000 \
  1000000:1200
do
  count=${spec%%:*}
  dimension=${spec##*:}

  benchmark/run_soc_dimension_projection.sh \
    --cuda-home "$CUDA_HOME" \
    --no-build --no-plot \
    --cone-count "$count" \
    --dimensions "$dimension" \
    --trials 10 \
    --run-label "gpu7_paper_total1p2e9_count${count}"
done
```

The 15-second rule is applied after a projection call returns. At 10 cones,
thread-wise is expected to return one `TIMEOUT` row. At one million cones,
grid-wise is expected to return one `TIMEOUT` row. Results on another GPU may
differ.

### 6.2 Counts above 1,000,000

Do not launch grid-wise here. It would issue 10–120 million sequential cuBLAS
projection calls before the current in-process timeout could inspect elapsed
time. Record it honestly as `SKIPPED_TIMEOUT_RISK` and run block-, warp-, and
thread-wise:

```bash
for spec in \
  10000000:120 \
  100000000:12 \
  120000000:10
do
  count=${spec%%:*}
  dimension=${spec##*:}

  benchmark/run_soc_dimension_projection.sh \
    --cuda-home "$CUDA_HOME" \
    --no-build --no-plot \
    --cone-count "$count" \
    --dimensions "$dimension" \
    --trials 10 \
    --strategies moderate,sufficient,massive \
    --skip-strategies few \
    --run-label "gpu7_paper_total1p2e9_count${count}"
done
```

`SKIPPED_TIMEOUT_RISK` is not a measured timeout and must never be plotted as
a runtime.

### 6.3 Generate the fixed-total report and figure

```bash
JULIA_DEPOT_PATH="$JULIA_DEPOT_PATH" \
./.julia-bin/julia --project=. benchmark/summarize_soc_paper_count_sweep.jl \
  --root benchmark/results/soc_dimension \
  --output rebuttal_plan/cone_count_results.md \
  --figure-tex rebuttal_plan/figures/soc_count_fixed_total.tex

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory rebuttal_plan/figures \
  rebuttal_plan/figures/soc_count_fixed_total.tex

pdftoppm -png -r 300 -singlefile \
  rebuttal_plan/figures/soc_count_fixed_total.pdf \
  rebuttal_plan/figures/soc_count_fixed_total
```

Current H100 result to compare—not to copy—is:

- grid-wise wins at 10–100 cones;
- warp-wise wins at 1,000–10 million cones;
- thread-wise wins at 100–120 million cones;
- transitions occur between 100–1,000 and 10–100 million cones.

## 7. Assemble the master report

After both component reports exist, assemble them without overwriting either:

```bash
JULIA_DEPOT_PATH="$JULIA_DEPOT_PATH" \
./.julia-bin/julia --project=. benchmark/assemble_soc_rebuttal_report.jl \
  --dimension-report rebuttal_plan/cone_dimension_results.md \
  --count-report rebuttal_plan/cone_count_results.md \
  --output rebuttal_plan/cone_projectioin_results.md
```

Confirm that the master contains both sections:

```bash
grep -n '^# SOC dimension projection results' \
  rebuttal_plan/cone_projectioin_results.md
grep -n '^# Figure 3 reproduction' \
  rebuttal_plan/cone_projectioin_results.md
```

## 8. Timing, randomness, and correctness semantics

Every timed vector is generated directly on the GPU with independent Float64
coordinates

```text
x_i = sigma × Z_i,  Z_i ~ Normal(0, 1),
```

so `x_i ~ Normal(0, sigma²)`, where `sigma` is the standard deviation selected
with `--sigma` (default `1.0`). No feasible, boundary, or other hand-constructed
SOC cases are mixed into this benchmark. The deterministic seed is

```text
2026 + 10000 × dimension_index + trial
```

For a fixed cone count, dimension, and trial, all four strategies receive the
same random vector. Different trials receive different vectors. This makes the
comparison paired and repeatable within a software environment.

The timed interval includes synchronized projection execution. It excludes:

- allocation;
- random-number generation;
- input preparation;
- kernel warm-up and compilation;
- correctness copies and CPU reference computation.

Only trial 1 of each dimension/strategy performs the explicit closed-form CPU
correctness check, using representative first, middle, and last cones.
Therefore:

- trial 1 has a numeric `max_error`;
- trials 2–10 have an empty `max_error`;
- `PASS` on trials 2–10 means successful timed execution, not a separately
  computed CPU-reference error.

The first trial's correctness work is also outside its timed interval.

## 9. CSV schemas and statuses

Raw CSV columns:

```text
run_id,timestamp_utc,gpu_name,gpu_uuid,compute_capability,
driver_version,cuda_toolkit,cuda_runtime,julia_version,git_commit,
cone_count,cone_dimension,total_dimension,strategy,trial,seed,
runtime_ms,max_error,status,note
```

Summary CSV columns:

```text
cone_count,cone_dimension,strategy,completed_trials,mean_ms,std_ms,
median_ms,min_ms,max_ms,fastest_strategy,status
```

Statuses:

- `PASS`: measured projection completed and the applicable correctness check
  passed;
- `FAIL`: checked projection exceeded numerical tolerance;
- `TIMEOUT`: a projection returned after exceeding 15 seconds;
- `ERROR`: runtime failure;
- `SKIPPED_MEMORY`: conservative memory check rejected the case;
- `SKIPPED_TIMEOUT_RISK`: deliberately not launched because sequential launch
  count prevents safe in-process timeout handling.

## 10. Validation checklist

Validate Experiment A:

```bash
for run in \
  gpu7_3cones_four_strategies_rerun \
  gpu7_100cones_four_strategies
do
  raw="benchmark/results/soc_dimension/$run/raw.csv"
  test "$(wc -l < "$raw")" -eq 401
  awk -F, 'NR>1 {gsub(/"/,"",$19); n[$19]++}
             END {for (s in n) print FILENAME, s, n[s]}' "$raw"
done
```

Expected: `PASS 400` for each file.

Validate Experiment B:

```bash
for count in 10 100 1000 10000 100000 1000000 \
             10000000 100000000 120000000
do
  raw="benchmark/results/soc_dimension/gpu7_paper_total1p2e9_count${count}/raw.csv"
  awk -F, 'NR>1 {gsub(/"/,"",$19); n[$19]++}
             END {for (s in n) print FILENAME, s, n[s]}' "$raw"
done
```

For the current H100 results, aggregated expected statuses are:

```text
PASS 310
TIMEOUT 2
SKIPPED_TIMEOUT_RISK 3
```

Also verify:

```bash
test -s rebuttal_plan/cone_dimension_results.md
test -s rebuttal_plan/cone_count_results.md
test -s rebuttal_plan/cone_projectioin_results.md
test -s rebuttal_plan/figures/soc_count_fixed_total.pdf
test -s rebuttal_plan/figures/soc_count_fixed_total.png
```

Before citing results, compare GPU UUIDs in every `raw.csv`. Both experiments
must use the intended device. Review every `environment.txt`, `benchmark.log`,
and `summary.log` for unexpected warnings.

## 11. Common problems

### `nvidia-smi` works but CUDA.jl is not functional

The driver being visible does not guarantee that Julia has a usable CUDA
runtime. Run the CUDA.jl check in Section 3 and inspect its error.

### Invalid device function

The PTX/shared library was probably compiled for another architecture. Rerun
the smoke command without `--no-build`, or pass the correct `--arch sm_XX`.

### cuBLAS cannot be loaded

Check `ldd` as shown in Section 4 and ensure the chosen toolkit's libraries are
visible to the dynamic loader.

### A large case reports `SKIPPED_MEMORY`

Read its raw CSV `note` column for estimated required and available bytes.
Close other GPU processes or use a larger-memory GPU. Do not silently remove
the point and call the result a complete reproduction.

### Plotting tools are unavailable

Benchmarking and CSV/Markdown summaries still work. Install PGFPlots and
Poppler later, then rerun only the commands in Section 6.3.

### Another machine gives different crossovers

The thresholds are empirical and GPU-dependent. Confirm identical experiment
definitions and correctness first, then report the new machine's measured
crossover rather than replacing it with the current H100 result.
