# How to Run the R1-3 Parametric-Similarity Experiments

This runbook executes the required experiment in
`rebuttal_plan/warp_divergence_3.md`. It is designed for reproduction on a
separate H100 machine. The new experiment is independent of the older
grouped/interleaved 2-by-2 harness and does not require Nsight Compute
privilege.

The inferential unit is one workload seed. Use exactly:

```text
2026,2027,2028,2029,2030,2031,2032,2033,2034,2035
```

The pilot seed is 2025 and is not included in inference.

## 1. Implemented files

| File | Purpose |
|:--|:--|
| `benchmark/rescaled_soc_parametric_similar.jl` | Dedicated generator, diagnostic, timing, correctness, and duration driver |
| `benchmark/run_soc_parametric_similarity_v3.sh` | Complete phase runner |
| `benchmark/run_soc_parametric_similarity_utilization.sh` | READY/START/DONE-aligned endpoint utilization wrapper |
| `benchmark/analyze_soc_parametric_similar.jl` | Ten-seed paired analysis and publication tables |
| `benchmark/rebuttal/soc_divergence_cases.jl` | Coupled controlled parametric generator |
| `test/rebuttal/soc_parametric_similarity_test.jl` | Determinism and mathematical generator tests |
| `benchmark/rebuttal/diagnostic_root_profile.jl` | Existing diagnostic PTX interface reused by the new driver |
| `benchmark/rebuttal/snapshot_source.sh` | Dirty-worktree snapshot and source hashes |

The experiment uses:

```text
projection        = type-22 diagonally rescaled SOC
cone count        = 1,048,576
cone dimension    = 10
delta             = 0, 1e-4, 1e-3, 1e-2
strategies        = threadWise, warpWise
warm-ups          = 5 per timing cell
measured launches = 10 per timing cell
duration          = 35 seconds per utilization cell
precision         = Float64
tolerances        = abs_tol=1e-12, rel_tol=1e-12
```

## 2. Copy the repository

Copy the complete source checkout without old result directories:

```bash
rsync -a \
  --exclude '.git/' \
  --exclude '.julia-depot/' \
  --exclude '.CondaPkg/' \
  --exclude 'benchmark/results/' \
  /home/zhenwei/PDCS_fork/ \
  USER@H100_HOST:/home/USER/PDCS_fork/
```

The command does not delete files on the destination. Prefer a fresh
destination directory.

## 3. Configure the target machine

On the H100 machine:

```bash
cd /home/USER/PDCS_fork

export REPO_ROOT="$PWD"
export JULIA_BIN="$REPO_ROOT/.julia-bin/julia"
export JULIA_DEPOT="$REPO_ROOT/.julia-depot"
export CUDA_ROOT=/usr/local/cuda-12.5
export GPU_PHYSICAL=0
export GPU_ARCH=sm_90

export CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export JULIA_DEPOT_PATH="$JULIA_DEPOT"
export PDCS_SKIP_GPU_PRECOMPILE=1
export PATH="$CUDA_ROOT/bin:$PATH"
```

Change `CUDA_ROOT` to the installed toolkit directory. Do not use the
driver-reported maximum CUDA version as the toolkit path.

Verify:

```bash
"$JULIA_BIN" --version
"$CUDA_ROOT/bin/nvcc" --version
nvidia-smi -i "$GPU_PHYSICAL" \
  --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free,power.limit,compute_mode \
  --format=csv

env CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
    JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" --project=. -e '
using CUDA
CUDA.functional() || error("CUDA is not functional")
println("device = ", CUDA.name(CUDA.device()))
println("uuid = ", CUDA.uuid(CUDA.device()))
println("capability = ", CUDA.capability(CUDA.device()))
println("runtime = ", CUDA.runtime_version())
'
```

The Julia and `nvidia-smi` UUIDs must refer to the same physical GPU.

## 4. Install and prewarm dependencies

Perform dependency installation before any experiment:

```bash
env JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" --project=. -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
using CUDA, PDCS
CUDA.functional() || error("CUDA is not functional")
'
```

This experiment does not run under NCU, so it does not force
`JULIA_CONDAPKG_BACKEND=Null`. If the target machine deliberately uses the
Null backend, configure a compatible system Python before loading PDCS.
Dependency installation must finish before timing or utilization collection.

## 5. Select an immutable run ID

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_soc_parametric_similarity"
export RUN_ROOT="$REPO_ROOT/benchmark/results/rebuttal/soc_parametric_similarity/$RUN_ID"
```

Reuse this exact `RUN_ID` for every phase. Use a new ID after a failed or
invalidated formal run; do not merge favorable cells from different IDs.

## 6. Inspect all commands without running the GPU

```bash
benchmark/run_soc_parametric_similarity_v3.sh \
  --phase all --run-id "$RUN_ID" --smoke --dry-run \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

The smoke dry run expands eleven top-level commands:

```text
snapshot + build + test + pilot + generate + timing
+ 4 endpoint utilization cells + analysis
```

The full utilization phase expands:

```text
2 deltas × 2 strategies × 10 seeds = 40 cells
```

## 7. Run the CPU generator tests

```bash
env JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" --project=. \
    test/rebuttal/soc_parametric_similarity_test.jl
```

The tests verify:

- deterministic regeneration and seed changes;
- reuse of primitive perturbations across all four `delta` values;
- bitwise-identical cones at `delta=0`;
- centered log diagonals;
- RMS log-diagonal perturbation equal to `delta`;
- monotone angular and diagonal perturbation magnitudes;
- finite values and diagonal bounds.

Expected: 31 tests pass (27 generator checks and 4 Williams-order checks).

## 8. Build production and diagnostic kernels

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu rebuild-profile \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
```

Production timing calls only the ordinary `threadWise` and `warpWise`
projection entry points. Diagnostic PTX is used only during generation and
path/work validation.

Never use diagnostic runtime as publication timing.

## 9. Run the seed-2025 pilot

The pilot verifies the fixed mathematical construction

```text
t_i = 0.20 * norm(diag(d_i) * u_i)
```

maps to production path code 2 for every cone:

```bash
benchmark/run_soc_parametric_similarity_v3.sh \
  --phase pilot --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

Read:

```text
$RUN_ROOT/pilot/seed_2025/generator/
$RUN_ROOT/pilot/seed_2025/diagnostic/termination_summary.csv
$RUN_ROOT/pilot/seed_2025/diagnostic/warp_root_work.csv
```

Every `distinct_path_min` and `distinct_path_max` must equal 1, every
`modal_path_fraction_min` must equal 1, `max_iter_count` must equal zero, and
`nonfinite_count` must equal zero. The pilot must not be used to tune the
coefficient for a favorable runtime outcome.

## 10. Run a complete smoke experiment

Use a separate smoke ID:

```bash
export SMOKE_ID="${RUN_ID}_smoke"

benchmark/run_soc_parametric_similarity_v3.sh \
  --phase all --run-id "$SMOKE_ID" --smoke \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

Smoke settings are 1,024 cones, seed 2026, and five-second duration cells.
They validate the workflow only and are not publication data.

Check:

```bash
find "$REPO_ROOT/benchmark/results/rebuttal/soc_parametric_similarity/$SMOKE_ID" \
  -maxdepth 4 -type f | sort

grep -R ',FAIL' \
  "$REPO_ROOT/benchmark/results/rebuttal/soc_parametric_similarity/$SMOKE_ID" \
  || echo "No recorded FAIL rows"
```

## 11. Formal execution order

Run phases sequentially on one otherwise idle GPU:

```bash
for phase in snapshot build test pilot generate timing utilization analysis; do
  benchmark/run_soc_parametric_similarity_v3.sh \
    --phase "$phase" --run-id "$RUN_ID" \
    --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
    --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
done
```

Do not run timing and utilization concurrently.

### 11.1 Generation and diagnostics

The `generate` phase creates four coupled mathematical cases for every seed,
runs thread-wise and warp-wise diagnostic kernels, applies the path and
termination gates, checks diagnostic/production invariance, and saves one
case cache per seed.

Important outputs:

```text
$RUN_ROOT/seeds/seed_<seed>/case_cache.jls
$RUN_ROOT/seeds/seed_<seed>/generator/input_hashes.csv
$RUN_ROOT/seeds/seed_<seed>/generator/seed_level_similarity.csv
$RUN_ROOT/seeds/seed_<seed>/generator/rejection_counts.csv
$RUN_ROOT/seeds/seed_<seed>/diagnostic/per_cone/
$RUN_ROOT/seeds/seed_<seed>/diagnostic/per_cone_summary.csv
$RUN_ROOT/seeds/seed_<seed>/diagnostic/warp_root_work.csv
$RUN_ROOT/seeds/seed_<seed>/diagnostic/termination_summary.csv
```

Per-cone CSV output is enabled for the formal workflow. When `zstd` is
available, the runner replaces each successfully compressed `.csv` with its
lossless `.csv.zst` representation immediately after that seed finishes.
The pilot omits per-cone files because only its path gate is required. If disk
space is limited during a smoke test only, use `--no-per-cone`. Do not use that
option for the formal experiment because the plan requires retained per-cone
diagnostic records.

### 11.2 Eight-cell timing

Within each seed, the timing driver holds the same four mathematical cases for
both strategies, warms every cell five times, then executes ten rounds. Each
round contains all eight `strategy × delta` cells in a cyclic balanced order
shifted by the seed index.

Read:

```text
$RUN_ROOT/seeds/seed_<seed>/timing/launch_level.csv
$RUN_ROOT/seeds/seed_<seed>/timing/seed_level.csv
$RUN_ROOT/seeds/seed_<seed>/correctness/cpu_gpu.csv
```

Each seed must contain:

```text
80 measured timing rows
40 warm-up rows
8 seed-level timing rows
4 PASS correctness rows
```

`launch_level.csv` records the exact round, position, GPU UUID, clocks, power,
temperature, and gate status. No launch is discarded as a statistical
outlier.

### 11.3 Endpoint utilization

The utilization phase runs:

```text
delta    = 0, 0.01
strategy = threadWise, warpWise
seed     = 2026, ..., 2035
```

The wrapper:

1. starts and warms the Julia process;
2. waits for its `READY` record;
3. starts one-second `nvidia-smi` sampling;
4. sends `START`;
5. stops monitoring after `DONE`;
6. preserves application logs, raw samples, error logs, and exit status.

Raw files are under:

```text
$RUN_ROOT/utilization/raw/
```

No numerical utilization value is filtered. The analyzer discards the
predeclared first five samples and retains the next thirty.

## 12. Analysis

The final phase runs:

```bash
env JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" --project=. \
    benchmark/analyze_soc_parametric_similar.jl \
    --root "$RUN_ROOT" \
    --seeds 2026,2027,2028,2029,2030,2031,2032,2033,2034,2035 \
    --bootstrap 10000
```

Read:

```text
$RUN_ROOT/analysis/seed_level_ratios.csv
$RUN_ROOT/analysis/effect_estimates.csv
$RUN_ROOT/analysis/bootstrap_draws.csv
$RUN_ROOT/analysis/root_work_publication.csv
$RUN_ROOT/analysis/warp_root_work_publication.csv
$RUN_ROOT/analysis/publication_tables.md
$RUN_ROOT/utilization/aligned_seed_level.csv
```

The analyzer:

- requires exactly ten seed-level observations;
- requires ten measured launches in every timing cell;
- rejects any non-PASS correctness cell;
- computes ratios within seed before aggregation;
- reports geometric mean ratios;
- uses 10,000 coupled seed-block bootstrap draws;
- keeps endpoint timing ratios descriptive;
- treats one-second utilization samples as repeated measurements, not
  independent statistical observations.

## 13. Acceptance checks

Run:

```bash
test -s "$RUN_ROOT/manifest.json"
test -s "$RUN_ROOT/environment.txt"
test -d "$RUN_ROOT/source_snapshot"

find "$RUN_ROOT/seeds" -name seed_level.csv | wc -l
find "$RUN_ROOT/seeds" -name cpu_gpu.csv | wc -l
find "$RUN_ROOT/utilization/raw" -name '*.gpu.csv' | wc -l
find "$RUN_ROOT/utilization/raw" -name '*.exit_status.txt' \
  -exec sh -c 'test "$(cat "$1")" = 0' _ {} \;

grep -R ',FAIL' "$RUN_ROOT/seeds" && \
  echo "FAIL rows found" || echo "No FAIL rows"
```

Expected formal counts:

```text
seed_level.csv       = 10
cpu_gpu.csv          = 10
*.gpu.csv            = 40
*.exit_status.txt    = 40, all containing 0
```

Also verify:

- 10/10 seeds pass, not “20/20 seeds”;
- all path codes match the intended positive-root path;
- zero `MAX_ITER` and nonfinite records;
- four deltas and two strategies exist for every seed;
- at least 30 publication-window utilization samples exist per cell;
- GPU UUID is constant across timing and monitoring;
- the time ledger relative error is at most 1%.

## 14. Storage policy

The case caches and per-cone diagnostic files are the largest intermediates.
Keep them until generation, diagnostic invariance, timing, utilization, and
analysis have all passed. The formal run must retain raw per-cone records,
raw launch timings, raw utilization samples, source snapshots, and hashes.

Do not delete or overwrite a partial run. If additional space is required,
copy a completed immutable run to archival storage and record checksums before
removing a local copy.

## 15. Failure handling

- A pilot path failure means the type-22 mathematical mapping must be fixed
  before inferential seeds are run.
- Any correctness or path failure invalidates the complete seed.
- Exit 137 indicates an external SIGKILL; preserve logs and inspect host/GPU
  memory pressure.
- A GPU UUID change or competing workload invalidates the affected seed.
- Do not replace only a slow or unfavorable cell.
- Use a new run ID for a clean rerun.
- Do not pool repeatability runs as additional independent seeds.

## 16. Interpretation limits

The final response may conclude only for the tested type-22, dimension-10,
\(2^{20}\)-cone, FP64 workload on the recorded H100 and software environment.

Call `nvidia-smi utilization.gpu` **GPU utilization**, not SM busy or active
lane efficiency. High utilization is supporting evidence that the device
remained active; it does not prove that divergence was absent or that every
lane performed useful work.

The work-count efficiency in `warp_root_work.csv` is a
**work-count-based modeled lane-efficiency proxy**, not a hardware active-mask
measurement.
