# How to reproduce the SOC warp-divergence experiments

This runbook implements the paired \(2\times2\) design in
`rebuttal_plan/warp_divergence.md`. Run it from a clean checkout on the target
GPU machine. It does not use the older independent-sigma divergence results.

The primary workload is the diagonally rescaled SOC projection (type 22) with
1,048,576 cones of dimension 10. Every seed uses the same cone multiset in
grouped and interleaved order and forces both `threadWise` and `warpWise`.

## 1. Scripts

| File | Purpose |
|:--|:--|
| `benchmark/rescaled_soc_divergence_2x2.jl` | Generate/cache cases, run diagnostic manipulation gates, check correctness, time, profile one launch, or sustain a workload |
| `benchmark/rebuttal/soc_divergence_cases.jl` | Deterministic Gaussian direction, centered log-diagonal, branch, parametric, and exact permutation generators |
| `benchmark/run_soc_divergence_2x2.sh` | Full generation and unprofiled timing for seeds 2026–2045 |
| `benchmark/analyze_soc_divergence_2x2.jl` | Compute \(A\), \(R_T\), \(R_W\), \(\Theta\), paired intervals, bootstrap intervals, and 5% decisions |
| `benchmark/profile_soc_divergence_2x2_ncu.sh` | Profile the four paired cells with Nsight Compute and an NVTX-selected launch |
| `benchmark/run_soc_divergence_2x2_utilization.sh` | Collect 30-second `nvidia-smi dmon` traces |
| `benchmark/summarize_soc_divergence_utilization.jl` | Summarize sustained SM-busy samples |
| `benchmark/run_soc_parametric_similarity.sh` | Run all four parametric-similar perturbation levels and seeds |
| `benchmark/analyze_soc_parametric_similarity.jl` | Summarize parametric runtime, root work, spread, and thread/warp ratios |
| `benchmark/profile_soc_parametric_ncu.sh` | Profile the parametric levels with Nsight Compute |
| `benchmark/run_soc_parametric_utilization.sh` | Collect sustained utilization for every parametric level |
| `test/test_soc_divergence_2x2_cases.jl` | Determinism, centered-log invariant, and exact-layout tests |

The diagnostic build writes separate `*_profile.ptx` files. It never replaces
the production PTX. Publication timing uses only the uninstrumented production
kernels.

## 2. Machine requirements

- Julia 1.10 or newer;
- CUDA Toolkit 12.5 or 12.6 with `nvcc`, `ncu`, and `nvidia-smi`;
- an H100 (`sm_90`) for the primary result, or A100 (`sm_80`) for a separate
  replication;
- at least 80 GB GPU memory for the full candidate pool and paired workload;
- Nsight Compute performance-counter permission for the NCU subset;
- `zstd` is recommended for compressing raw per-cone CSV files.

Check the machine before running anything:

```bash
nvidia-smi -L
nvidia-smi
/usr/local/cuda-12.6/bin/nvcc --version
/usr/local/cuda-12.6/bin/ncu --version
/usr/local/cuda-12.6/bin/ncu --list-sets
```

If `ncu` reports `ERR_NVGPUCTRPERM`, the timing and dmon experiments can still
run, but the NCU result is unavailable until an administrator enables
performance-counter access. Do not run the profiler with `sudo` as a routine
workaround.

## 3. Configure the clean checkout

Adjust the physical GPU index and architecture:

```bash
cd /home/zhenwei/PDCS_fork

export CUDA_ROOT=/usr/local/cuda-12.6
export GPU_PHYSICAL=0
export GPU_ARCH=sm_90                 # H100
# export GPU_ARCH=sm_80               # A100 replication

export JULIA_BIN="$PWD/.julia-bin/julia"
export JULIA_DEPOT="$PWD/.julia-depot"
export PATH="$CUDA_ROOT/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export JULIA_DEPOT_PATH="$JULIA_DEPOT"
export PDCS_SKIP_GPU_PRECOMPILE=1
```

If the local Julia binary is absent, install the repository's documented
Julia version first, then set `JULIA_BIN` to that executable.

Instantiate the project:

```bash
"$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate()'
```

The project includes `NVTX.jl` because the NCU driver selects exactly the
`PDCS_PROJECTION` range rather than relying on a fragile launch-skip count.

Verify Julia sees the same physical GPU:

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

The UUIDs must identify the same GPU. Stop unrelated GPU workloads before
timing.

## 4. Build production and diagnostic kernels

Build the normal solver kernels:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
```

Build the separate counter-enabled diagnostic PTX:

```bash
make -C src/pdcs_gpu/cuda rebuild-profile \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
```

The outputs are intentionally different:

```text
Production timing:
  massive_block_proj.ptx
  sufficient_block_proj.ptx

Diagnostic counters:
  massive_block_proj_profile.ptx
  sufficient_block_proj_profile.ptx
```

Never use `*_profile.ptx` runtime as a manuscript number.

## 5. Run deterministic CPU tests

```bash
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
"$JULIA_BIN" --project=. test/test_soc_divergence_2x2_cases.jl
```

Expected result:

```text
SOC divergence generators are deterministic: 6/6 pass
exact grouped/interleaved layouts: 6/6 pass
parametric-similar generator: 8/8 pass
```

Do not continue if these tests fail.

## 6. Check every command without launching the GPU experiment

```bash
benchmark/run_soc_divergence_2x2.sh \
  --smoke \
  --dry-run \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --arch "$GPU_ARCH" \
  --run-id command_check
```

This must print separate generation and timing commands for both `iteration`
and `branch`. Also check the complete parametric matrix:

```bash
benchmark/run_soc_parametric_similarity.sh \
  --run-dir /tmp/pdcs_parametric_command_check \
  --smoke \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --dry-run
```

Dry-run mode creates no result directory.

## 7. Real-GPU smoke test

Use a unique run ID:

```bash
SMOKE_ID="$(date -u +%Y%m%dT%H%M%SZ)_soc_divergence_smoke"

benchmark/run_soc_divergence_2x2.sh \
  --smoke \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --run-id "$SMOKE_ID"
```

Smoke mode uses 1,024 cones and seed 2026 but preserves the exact layout
formulas, both strategies, five warm-ups, ten counterbalanced rounds, CPU
sampling, diagnostic/production agreement, and manipulation gates.

Set:

```bash
export PDCS_RUN_DIR="$PWD/benchmark/results/rebuttal/soc_divergence_2x2/$SMOKE_ID"
```

Check:

```bash
find "$PDCS_RUN_DIR" -maxdepth 3 -type f | sort
grep -R ',FAIL' "$PDCS_RUN_DIR" || true
column -s, -t < "$PDCS_RUN_DIR/iteration/seed2026/manipulation_checks.csv"
column -s, -t < "$PDCS_RUN_DIR/branch/seed2026/manipulation_checks.csv"
column -s, -t < "$PDCS_RUN_DIR/iteration/seed2026/correctness.csv"
column -s, -t < "$PDCS_RUN_DIR/branch/seed2026/correctness.csv"
```

Required gates:

- iteration manipulation status is `PASS`;
- branch manipulation status is `PASS`;
- correctness status is `PASS`;
- diagnostic/production error is at most `5e-8`;
- sampled CPU/GPU scaled error is at most 1;
- no timing cell is absent.

The pilot selection intentionally stops if none of the four predeclared
family/grid stages establishes the required iteration contrast. Do not weaken
the gate or select a family from runtime results.

## 8. Full unprofiled 20-seed experiment

Use a new result directory:

```bash
FULL_ID="$(date -u +%Y%m%dT%H%M%SZ)_soc_divergence_full"

benchmark/run_soc_divergence_2x2.sh \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seeds 2026:2045 \
  --experiments iteration,branch \
  --run-id "$FULL_ID"

export PDCS_RUN_DIR="$PWD/benchmark/results/rebuttal/soc_divergence_2x2/$FULL_ID"
```

For every experiment and seed, the runner first generates and validates an
immutable `case_cache.jls`, then starts a separate timing process using that
cache. Consequently, diagnostic kernels are not launched during the
publication timing process.

The runner refuses to overwrite an existing run directory. If interrupted,
keep the partial directory and start a new run ID; do not delete or merge
partial seed blocks into the confirmatory result.

## 9. Inspect and analyze timing

The runner automatically invokes the analyzer after all 20 seeds finish.
It fails if a seed, correctness row, or one of the four paired cells is
missing.

To repeat analysis without rerunning the GPU:

```bash
"$JULIA_BIN" --project=. benchmark/analyze_soc_divergence_2x2.jl \
  --root "$PDCS_RUN_DIR" \
  --seeds 2026:2045 \
  --bootstrap 10000 \
  --output "$PDCS_RUN_DIR"
```

Read:

```text
$PDCS_RUN_DIR/seed_effects.csv
$PDCS_RUN_DIR/effect_estimates.csv
$PDCS_RUN_DIR/publication_tables.md
```

`effect_estimates.csv` reports:

- `A`: thread-wise/warp-wise under interleaving;
- `A_grouped`: the grouped absolute reference;
- `R_T`: thread-wise interleaved/grouped;
- `R_W`: warp-wise ordering control;
- `Theta`: `R_T/R_W`;
- paired log-scale confidence intervals;
- 10,000 seed-block bootstrap intervals;
- the predeclared one-sided 5% decision.

## 10. Nsight Compute

NCU is supplementary and must run only after unprofiled timing completes.
Rebuild optimized production PTX with source line information:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH" LINEINFO=-lineinfo
```

Dry-run all 12 primary profile commands (3 seeds × 4 cells):

```bash
benchmark/profile_soc_divergence_2x2_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seeds 2026,2027,2028 \
  --experiments iteration \
  --dry-run
```

Then collect:

```bash
benchmark/profile_soc_divergence_2x2_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seeds 2026,2027,2028 \
  --experiments iteration
```

The wrapper:

- saves `ncu/available_metrics.txt`;
- filters the native `massive_block_proj` or `sufficient_block_proj` kernel;
- selects the single `PDCS_PROJECTION` NVTX range;
- requests SpeedOfLight, Occupancy, SchedulerStats, WarpStateStats, and
  SourceCounters;
- preserves `.ncu-rep`, imported CSV, application log, NCU log, and exit
  status;
- writes `PROFILE_INCOMPLETE.txt` when permission is denied.

To profile the secondary branch experiment, use:

```bash
benchmark/profile_soc_divergence_2x2_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --seeds 2026,2027,2028 \
  --experiments branch
```

Do not report NCU replay duration as projection runtime.

## 11. Sustained GPU utilization

Collect at least 30 seconds for all four primary cells:

```bash
benchmark/run_soc_divergence_2x2_utilization.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seed 2026 \
  --duration 30 \
  --experiments iteration,branch
```

The monitor starts before Julia and is stopped by a cleanup trap even if Julia
fails. Each projection restores the immutable device input and zeroes root
state. The application log separately reports completed launches,
restore/reset time, and projection time.

Summarize:

```bash
"$JULIA_BIN" --project=. \
  benchmark/summarize_soc_divergence_utilization.jl \
  --root "$PDCS_RUN_DIR" \
  --output "$PDCS_RUN_DIR/utilization_summary.csv"
```

The dmon interval includes device restoration between kernels. Disclose this
when comparing the SM-busy distribution with the reviewer's suggested
80–90% range. Use NCU for projection-kernel-only throughput and occupancy.

## 12. Parametric-similar descriptive experiment

The generator implements
\(\delta_u=\delta_D\in\{0,10^{-4},10^{-3},10^{-2}\}\).
Run the full four-level, twenty-seed matrix:

```bash
benchmark/run_soc_parametric_similarity.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seeds 2026:2045 \
  --deltas 0,0.0001,0.001,0.01
```

This creates one diagnostic case cache and one separate production timing
process for every `(delta, seed)` pair. It then writes:

```text
$PDCS_RUN_DIR/parametric/parametric_summary.csv
```

Profile seeds 2026–2028 at every perturbation level:

```bash
benchmark/profile_soc_parametric_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seeds 2026,2027,2028 \
  --deltas 0,0.0001,0.001,0.01
```

Collect sustained utilization for both strategies at all four levels:

```bash
benchmark/run_soc_parametric_utilization.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seed 2026 \
  --duration 30 \
  --deltas 0,0.0001,0.001,0.01

"$JULIA_BIN" --project=. \
  benchmark/summarize_soc_divergence_utilization.jl \
  --root "$PDCS_RUN_DIR" \
  --output "$PDCS_RUN_DIR/utilization_summary.csv"
```

This experiment is descriptive and must not replace the paired iteration
experiment.

## 13. Complete all-experiments launch order

After the smoke gates pass, the complete reproduction is:

```bash
FULL_ID="${FULL_ID:-$(date -u +%Y%m%dT%H%M%SZ)_soc_divergence_full}"

# 1. Primary same-branch and secondary branch-divergence timing, 20 seeds.
benchmark/run_soc_divergence_2x2.sh \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT" \
  --seeds 2026:2045 --experiments iteration,branch --run-id "$FULL_ID"

export PDCS_RUN_DIR="$PWD/benchmark/results/rebuttal/soc_divergence_2x2/$FULL_ID"

# 2. Applied parametric-similar timing, four deltas × 20 seeds.
benchmark/run_soc_parametric_similarity.sh \
  --run-dir "$PDCS_RUN_DIR" --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" --seeds 2026:2045

# 3. Primary and secondary NCU profiles, seeds 2026–2028.
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH" LINEINFO=-lineinfo

benchmark/profile_soc_divergence_2x2_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" --seeds 2026,2027,2028 \
  --experiments iteration,branch

# 4. Parametric NCU profiles, all deltas and seeds 2026–2028.
benchmark/profile_soc_parametric_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT"

# 5. Sustained dmon workloads for iteration and branch.
benchmark/run_soc_divergence_2x2_utilization.sh \
  --run-dir "$PDCS_RUN_DIR" --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" --duration 30 \
  --experiments iteration,branch

# 6. Sustained dmon workloads for all parametric levels.
benchmark/run_soc_parametric_utilization.sh \
  --run-dir "$PDCS_RUN_DIR" --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" --duration 30

# 7. Final utilization table.
"$JULIA_BIN" --project=. benchmark/summarize_soc_divergence_utilization.jl \
  --root "$PDCS_RUN_DIR" \
  --output "$PDCS_RUN_DIR/utilization_summary.csv"
```

Steps 3 and 4 require NCU counter permission. Steps 1, 2, 5, 6, and 7 do not.

## 14. Result directory

The important artifacts are:

```text
environment.txt
iteration/seed*/case_manifest.csv
iteration/seed*/case_cache.jls
iteration/seed*/permutations.csv[.zst]
iteration/seed*/root_work_raw.csv[.zst]
iteration/seed*/manipulation_checks.csv
iteration/seed*/correctness.csv
iteration/seed*/timings_raw.csv
iteration/seed*/timings_seed_summary.csv
branch/seed*/...
parametric/delta_*/seed*/...
parametric/parametric_summary.csv
seed_effects.csv
effect_estimates.csv
publication_tables.md
ncu/*
nvidia_smi/*
utilization_summary.csv
```

Keep the entire directory immutable. Do not copy a favorable seed from another
run into an incomplete run.

## 15. Validation status of the scripts

On the preparation machine:

- all three shell scripts passed `bash -n`;
- all Julia scripts passed parser checks;
- deterministic generator/layout tests passed 20/20 assertions;
- diagnostic PTX rebuilt successfully with CUDA 12.6 for `sm_90`;
- the full smoke runner expanded both experiments correctly in dry-run mode;
- NCU expanded all four paired cells with native kernel filters and the NVTX
  range;
- dmon expanded all four paired cells and retained its cleanup trap;
- the analysis script completed a synthetic 20-seed paired dataset and
  generated all effect estimates.
- the parametric runner expanded both smoke deltas into separate diagnostic
  generation and production timing processes;
- the parametric NCU wrapper expanded both strategies for every requested
  `(delta, seed)` case;
- the parametric utilization wrapper expanded both strategies for every
  requested delta;
- the parametric analyzer completed a synthetic four-level-compatible,
  20-seed directory schema;
- every `benchmark/*.jl` or `benchmark/*.sh` path referenced by this runbook
  exists, all new Julia files parse, and all relevant shell files pass
  `bash -n`.

The preparation machine currently cannot communicate with the NVIDIA driver:
even `nvidia-smi -L` fails. Therefore the real-GPU smoke test, manipulation
gate, full timings, NCU collection, and dmon collection must be executed on
the reproduction machine. Do not treat dry-run validation as experimental
data.
