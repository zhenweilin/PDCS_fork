# How to Run the Additional Warp-Divergence Experiments

This runbook implements the executable workflow defined in
`rebuttal_plan/warp_divergence_2.md`. It is intended for a separate H100
machine and starts from environment verification rather than assuming the
preparation machine's CUDA installation.

The inferential seeds are fixed to:

```text
2026,2027,2028,2029,2030,2031,2032,2033,2034,2035
```

Do not add seeds 2036--2045, and do not treat ten kernel launches as ten
independent observations.

This document mirrors the operational structure of
`how_to_run_warp_divergence.md` and extends it for the second experiment plan:

| Original runbook topic | This runbook |
|:--|:--|
| Machine requirements | Sections 2–5 |
| Configure environment and verify GPU | Sections 2–4 |
| Production, diagnostic, and line-info builds | Sections 9 and 21 |
| Deterministic tests | Section 21 |
| CondaPkg preparation | Sections 4 and 22 |
| Smoke test | Section 8 |
| Full ten-seed timing | Section 10 |
| Re-analysis | Section 17 |
| Nsight Compute | Sections 5 and 13 |
| Sustained utilization | Section 11 |
| Complete launch order | Section 23 |
| Fixed bugs | Section 24 |
| Machine differences | Section 25 |

Unlike the original document, this runbook never instructs the user to remove
a CondaPkg lock or create a temporary Julia wrapper. The checked-in scripts
pass `JULIA_CONDAPKG_BACKEND=Null` and `-O1` directly.

## 1. Scripts and current scope

| Script | Purpose |
|:--|:--|
| `benchmark/run_warp_divergence_2.sh` | Top-level phase runner |
| `benchmark/rebuttal/snapshot_source.sh` | Preserve commit, dirty diff, untracked sources, and hashes |
| `benchmark/run_soc_divergence_2x2.sh` | Ten-seed baseline and dimension timing |
| `benchmark/run_soc_parametric_similarity.sh` | Four-level cold parametric SOC timing |
| `benchmark/rebuttal/run_active_window_utilization.sh` | READY/START/DONE-aligned utilization collection |
| `benchmark/profile_soc_divergence_2x2_ncu.sh` | One-launch branch/root NCU profiling |
| `benchmark/profile_soc_parametric_ncu.sh` | Parametric NCU profiling |
| `benchmark/exp_cone_projection.jl` | Independent exponential input/diagonal sigma sweeps |
| `benchmark/summarize_soc_divergence_utilization.jl` | Trim the five-second stabilization interval and summarize the next 30 samples |
| `benchmark/analyze_soc_divergence_2x2.jl` | Paired ten-seed SOC analysis |

The phase runner deliberately refuses to invent application commands. The
checked-in `benchmark/rebuttal/application_manifest.csv` still contains
placeholders. Experiment H becomes executable only after replacing them with
the manuscript's real Fisher, portfolio, and Lasso/SOC solve commands.

The new plan also requests diagnostic active-mask sampling, block-balanced
warp controls, warm five-state parametric sequences, and exponential
iteration-quartile layouts. These require new diagnostic kernel/data-model
implementations, not merely shell commands. They must not be represented as
completed by substituting modeled work counts or old exponential timings.

## 2. Copy the repository to the experiment machine

Preserve the dirty worktree because the experiment sources may not all be in a
commit:

```bash
rsync -a \
  --exclude '.git/' \
  --exclude '.julia-depot/' \
  --exclude '.CondaPkg/' \
  --exclude 'benchmark/results/' \
  /home/zhenwei/PDCS_fork/ \
  USER@H100_HOST:/home/USER/PDCS_fork/
```

On the H100 machine:

```bash
cd /home/USER/PDCS_fork
pwd
git status --short
```

The command does not delete files already present on the destination. Use a
fresh destination directory when possible, and do not copy old result
directories into the new immutable run.

## 3. Select Julia, CUDA, and the physical GPU

For the environment specified by `warp_divergence_2.md`:

```bash
export REPO_ROOT="$PWD"
export JULIA_BIN="$REPO_ROOT/.julia-bin/julia"
export JULIA_DEPOT="$REPO_ROOT/.julia-depot"
export CUDA_ROOT=/usr/local/cuda-12.4
export GPU_PHYSICAL=0
export GPU_ARCH=sm_90

export CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export JULIA_DEPOT_PATH="$JULIA_DEPOT"
export JULIA_CONDAPKG_BACKEND=Null
export PDCS_SKIP_GPU_PRECOMPILE=1
export PATH="$CUDA_ROOT/bin:$PATH"
```

If the machine has CUDA 12.5 or 12.6 instead, change only `CUDA_ROOT`. Record
the actual toolkit; do not describe the driver's maximum CUDA version as the
installed toolkit.

Verify:

```bash
"$JULIA_BIN" --version
"$CUDA_ROOT/bin/nvcc" --version
nvidia-smi -i "$GPU_PHYSICAL" \
  --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.free,power.limit,compute_mode \
  --format=csv

env JULIA_CONDAPKG_BACKEND=Null \
    JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" -O1 --project=. -e '
using CUDA
println("CUDA.jl = ", Base.pkgversion(CUDA))
println("device = ", CUDA.name(CUDA.device()))
println("uuid = ", CUDA.uuid(CUDA.device()))
println("capability = ", CUDA.capability(CUDA.device()))
println("runtime = ", CUDA.runtime_version())
println("functional = ", CUDA.functional())
'
```

The Julia UUID and `nvidia-smi` UUID must identify the same GPU.

## 4. Install project dependencies

```bash
env JULIA_CONDAPKG_BACKEND=Null \
    JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" --project=. -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
'
```

`JULIA_CONDAPKG_BACKEND=Null` prevents an NCU-launched Julia process from
starting `pixi`. Do not delete `.CondaPkg/lock` as part of the experiment.

## 5. Nsight Compute permission preflight

```bash
"$CUDA_ROOT/bin/ncu" --version
"$CUDA_ROOT/bin/ncu" --query-metrics > /tmp/pdcs_ncu_metrics.txt
test -s /tmp/pdcs_ncu_metrics.txt
```

Counter permission cannot be installed locally by an unprivileged user.
Administrators must enable NVIDIA performance counters or grant the machine's
approved profiling access. Verify with a small NCU smoke run before launching
the full profile matrix. Preserve `ERR_NVGPUCTRPERM` logs if permission is
still unavailable.

## 6. Choose one immutable run ID

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_warp_divergence_2"
export RUN_ROOT="$REPO_ROOT/benchmark/results/rebuttal/additional_experiments_2/$RUN_ID"
```

Every phase below must reuse this exact `RUN_ID`. The scripts refuse to
overwrite individual profiler/utilization artifacts.

## 7. Validate command expansion without running the GPU

```bash
benchmark/run_warp_divergence_2.sh \
  --phase baseline --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT" --dry-run

benchmark/run_warp_divergence_2.sh \
  --phase utilization --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT" --dry-run
```

The full utilization dry run must expand 80 commands:

```text
2 experiments × 2 layouts × 2 strategies × 10 seeds = 80
```

## 8. Smoke workflow

Use a separate smoke ID:

```bash
export SMOKE_ID="${RUN_ID}_smoke"

for phase in snapshot build baseline utilization parametric dimensions exp analysis; do
  benchmark/run_warp_divergence_2.sh \
    --phase "$phase" --run-id "$SMOKE_ID" --smoke \
    --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
    --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
done
```

The smoke workflow uses 1,024 cones, one seed, and five-second utilization.
It tests control flow and correctness only; it is not publication data.

Inspect:

```bash
find "$REPO_ROOT/benchmark/results/rebuttal/additional_experiments_2/$SMOKE_ID" \
  -type f -maxdepth 8 | sort
```

Do not reuse the smoke run directory for the full experiment.

## 9. Freeze source and build binaries

```bash
benchmark/run_warp_divergence_2.sh \
  --phase snapshot --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"

benchmark/run_warp_divergence_2.sh \
  --phase build --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

The snapshot contains:

```text
source_snapshot/git_commit.txt
source_snapshot/git_status.txt
source_snapshot/git_diff_binary.patch
source_snapshot/git_diff_cached_binary.patch
source_snapshot/untracked_sources/
source_snapshot/source_hashes.sha256
source_snapshot/binary_hashes.sha256
```

The production and diagnostic PTX are separate. Publication timing uses only
the production PTX.

## 10. Baseline SOC timing

Run Section 9's build phase first. The master runner passes `--no-build` to
the child timing runner so the same kernels are not recompiled unnecessarily.

```bash
benchmark/run_warp_divergence_2.sh \
  --phase baseline --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

This runs iteration and branch experiments for seeds 2026--2035, with five
warmups and ten CUDA-event measurements per cell. It keeps all case caches
because the new plan profiles and monitors all ten seeds.

Before continuing:

```bash
find "$RUN_ROOT/baseline/paired" -name correctness.csv -print | wc -l
find "$RUN_ROOT/baseline/paired" -name case_cache.jls -print | wc -l
```

Both counts must be 20. Every correctness row must be `PASS`.

## 11. Active-window utilization

```bash
benchmark/run_warp_divergence_2.sh \
  --phase utilization --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

For each cell, Julia initializes and warms the kernel, prints `READY`, and
blocks. The wrapper then starts one-second monitoring, sends `START`, and
stops monitoring after `DONE`. Each application log contains:

```text
READY
FIRST_PROJECTION_START
UTILIZATION_COMPLETE
LAST_PROJECTION_STOP
DONE
```

Each seed directory also receives a duration ledger containing wall,
restore/reset, projection, and unexplained time.

The summarizer discards exactly the first five one-second stabilization
samples and retains at most the following 30 samples:

```bash
"$JULIA_BIN" --project=. \
  benchmark/summarize_soc_divergence_utilization.jl \
  --root "$RUN_ROOT/utilization/raw" \
  --ledger-root "$RUN_ROOT/baseline/paired" \
  --output "$RUN_ROOT/utilization/utilization_summary.csv"
```

Do not remove zero-utilization samples based on their value.

## 12. Cold parametric-similar SOC panel

```bash
benchmark/run_warp_divergence_2.sh \
  --phase parametric --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

This runs:

```text
delta = 0, 0.0001, 0.001, 0.01
seed  = 2026, ..., 2035
```

Read:

```text
$RUN_ROOT/soc_parametric/parametric/parametric_summary.csv
```

This phase is the cold-start panel. Do not label it as the five-state warm
sequence requested separately in Experiment C.

## 13. Branch Nsight Compute profiles

Run only after NCU permission succeeds:

```bash
benchmark/run_warp_divergence_2.sh \
  --phase soc-profile --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

`soc-profile` collects iteration and branch cells for all ten seeds. Use
`--phase branch-profile` only when intentionally collecting the branch subset.

Collect the four parametric levels separately:

```bash
benchmark/run_warp_divergence_2.sh \
  --phase parametric-profile --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

The wrapper uses:

- exact kernel names `massive_block_proj` and `sufficient_block_proj`;
- `--launch-skip 5` to bypass the five warm-up projections and capture the
  explicit post-warm-up `profile-one` launch;
- `--launch-count 1`;
- no invalid `--force-overwrite=false`;
- no NVTX include filter;
- no `--target-processes all`;
- `JULIA_CONDAPKG_BACKEND=Null`; and
- Julia `-O1`.

NCU replay duration is not publication timing.

## 14. SOC dimension panel

This phase reuses the production binaries from Section 9 and does not rebuild
them once per dimension.

```bash
benchmark/run_warp_divergence_2.sh \
  --phase dimensions --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

This fixes the cone count at 262,144 and runs dimensions 10, 32, and 100 for
both iteration and branch panels, using all ten seeds.

Before each dimension, verify at least 20% HBM remains free:

```bash
nvidia-smi -i "$GPU_PHYSICAL" \
  --query-gpu=memory.total,memory.free --format=csv,noheader
```

## 15. Exponential independent heterogeneity sweeps

```bash
benchmark/run_warp_divergence_2.sh \
  --phase exp --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

The phase creates separate files for:

```text
input sigma    = 0.1, 0.5, 1, 2, 5, 10; diagonal sigma = 1
diagonal sigma = 0.1, 0.5, 1, 2, 5, 10; input sigma = 1
strategies     = threadWise, warpWise, blockWise
seeds          = 2026, ..., 2035
type           = primal diagonal exponential cone, 27
```

SOC and exponential timings must remain in separate tables.

After NCU permission succeeds, collect the implemented exponential profile
matrix:

```bash
benchmark/run_warp_divergence_2.sh \
  --phase exp-profile --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

This profiles `similar`, `heterogeneous`, `mixed_grouped`, and
`mixed_interleaved` cases for thread-wise and warp-wise kernels at every seed.

## 16. Application phase

First replace every placeholder command:

```bash
sed -n '1,20p' benchmark/rebuttal/application_manifest.csv
```

Every command must run one declared manuscript instance and write the path in
`$PDCS_APPLICATION_TRACE_OUTPUT`. Then run:

```bash
benchmark/run_warp_divergence_2.sh \
  --phase applications --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

The phase refuses placeholder commands instead of fabricating application
evidence.

## 17. Analysis

```bash
benchmark/run_warp_divergence_2.sh \
  --phase analysis --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

Read:

```text
$RUN_ROOT/seed_effects.csv
$RUN_ROOT/effect_estimates.csv
$RUN_ROOT/publication_tables.md
$RUN_ROOT/utilization/utilization_summary.csv
```

Verify exactly ten seed-level rows per inferential effect before quoting any
confidence interval.

## 18. Storage policy

The new plan requests profiling and utilization for all ten seeds, so the
master baseline and parametric phases retain their caches. After every
dependent NCU, utilization, and validation phase is complete, caches are
reproducible intermediates; measurements, manifests, hashes, summaries, logs,
`.ncu-rep`, and raw monitoring data are the required evidence.

Never delete or modify an existing result directory while a run is in
progress. If storage cleanup is needed, copy the immutable completed run to
archival storage first and record a checksum inventory.

## 19. Failure handling

- A failed correctness gate invalidates that seed; do not time around it.
- Preserve partial directories and use a new run ID for a clean rerun.
- Preserve `ERR_NVGPUCTRPERM` rather than substituting replay duration.
- Exit 137 is an external SIGKILL. Record host OOM logs and rerun the exact
  standalone Julia command printed by the wrapper.
- Do not delete `.CondaPkg/lock`; use `JULIA_CONDAPKG_BACKEND=Null`.
- Do not merge favorable seeds from different run IDs.

## 20. Final completeness audit

The following parts of `warp_divergence_2.md` require dedicated source work
beyond the executable phases above and must remain marked incomplete until
their acceptance tests exist and pass:

- five-state warm-root parametric SOC sequence;
- block-balanced warp-wise ordering controls;
- sampled active-mask diagnostic guarded by
  `PDCS_PROFILE_ACTIVE_LANES`;
- exponential iteration-quartile, branch-ordering, warm-sequence, NCU, and
  aligned-utilization panels;
- application-level NVTX/diagnostic projection instrumentation using real
  manuscript solvers; and
- integrated publication Tables 2--6.

Do not claim the exhaustive plan is complete merely because
`run_warp_divergence_2.sh --phase all` finishes. The phase runner completes
the currently implemented and scientifically matched subset and leaves
external or unimplemented evidence explicit.

## 21. Production, diagnostic, line-info builds and deterministic tests

These are the direct equivalents of Sections 3 and 4 in the original
runbook.

Production timing build:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
```

Separate diagnostic build:

```bash
make -C src/pdcs_gpu/cuda rebuild-profile \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
```

Line-info production build before NCU source correlation:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH" LINEINFO=-lineinfo
```

Run CPU/deterministic tests before using the GPU:

```bash
for test_file in \
  test/test_soc_divergence_2x2_cases.jl \
  test/test_rebuttal_case_generation.jl \
  test/test_rebuttal_permutations.jl \
  test/test_rebuttal_timing.jl \
  test/rebuttal/utilization_alignment_test.jl
do
  env JULIA_CONDAPKG_BACKEND=Null \
      JULIA_DEPOT_PATH="$JULIA_DEPOT" \
      "$JULIA_BIN" --project=. "$test_file"
done
```

On the GPU machine, additionally run:

```bash
env CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
    JULIA_CONDAPKG_BACKEND=Null \
    JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    PDCS_SKIP_GPU_PRECOMPILE=1 \
    "$JULIA_BIN" -O1 --project=. \
    test/test_soc_diagnostic_hardcases_gpu.jl
```

Never use `*_profile.ptx` for publication timing.

## 22. CondaPkg handling

This replaces Section 5 of the original runbook.

```bash
export JULIA_CONDAPKG_BACKEND=Null

env CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
    JULIA_CONDAPKG_BACKEND=Null \
    JULIA_DEPOT_PATH="$JULIA_DEPOT" \
    "$JULIA_BIN" -O1 --project=. -e '
using CUDA
CUDA.functional() || error("CUDA is not functional")
'
```

Do not run `pixi install` under NCU, and do not delete `.CondaPkg/lock`.

## 23. Complete launch order

The safe full order mirrors Section 10 of the original runbook. Run phases
sequentially on a single GPU:

```bash
for phase in snapshot build baseline utilization parametric dimensions exp analysis; do
  benchmark/run_warp_divergence_2.sh \
    --phase "$phase" --run-id "$RUN_ID" \
    --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
    --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
done
```

After NCU permission is verified:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH" LINEINFO=-lineinfo

benchmark/run_warp_divergence_2.sh \
  --phase soc-profile --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"

benchmark/run_warp_divergence_2.sh \
  --phase parametric-profile --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"

benchmark/run_warp_divergence_2.sh \
  --phase exp-profile --run-id "$RUN_ID" \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT"
```

Run `--phase applications` only after replacing the manifest placeholders.
Do not run timing, NCU, and utilization concurrently.

## 24. Reproduction bugs already fixed

| Bug | Current behavior |
|:--|:--|
| Invalid `--force-overwrite=false` in NCU | Removed |
| NVTX include matched no kernels | NCU uses exact native kernel name |
| `--target-processes all` attached to `pixi` | Removed |
| `NVTX.range_push(String)` failed | Uses a matching `NVTX.Domain` for push/pop |
| `CUDA.Event` missing | Uses `CUDA.CuEvent` |
| Warp-wise SOC returned 0.8 instead of 0.4 | Recovery preserves the evaluated root `xi` |
| Profile counters changed projection output | Diagnostics use local observational variables |
| Duration wrapper included Julia startup | READY/START/DONE monitoring handshake |
| LLVM/JIT failure under duration wrapper | Scripts pass Julia `-O1` directly |
| CondaPkg startup under profilers | Scripts set `JULIA_CONDAPKG_BACKEND=Null` |

See `DEBUG.md` for the detailed evidence and verification status.

## 25. Machine differences

| Item | Plan reference machine | Reproduction-machine action |
|:--|:--|:--|
| CUDA Toolkit | 12.4 at `/usr/local/cuda` | Set `CUDA_ROOT` to the actual 12.4–12.6 toolkit |
| GPU | H100, `sm_90` | Use `sm_90`; use `sm_80` only for a separately labeled A100 replication |
| Julia | 1.10.4 | Set `JULIA_BIN`; do not assume `.julia-bin/julia` |
| Depot | Repository-local | Set `JULIA_DEPOT_PATH` explicitly |
| Seeds | 2026–2035 | Exactly ten seeds; never restore the superseded 20-seed default |
| NCU privilege | Machine-dependent | Run permission preflight and preserve failures |
| CondaPkg | May invoke `pixi` | Set `JULIA_CONDAPKG_BACKEND=Null` |
| Application commands | Not checked in | Populate and freeze the real manuscript manifest before Experiment H |
