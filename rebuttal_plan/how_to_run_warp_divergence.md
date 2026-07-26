# How to Reproduce the SOC Warp-Divergence Experiments

This runbook documents the **actual working procedure** used to reproduce the
paired \(2\times2\) warp-divergence experiment on the target GPU machine. It
supersedes the original runbook and incorporates all bug fixes discovered
during the 2026-07-26 reproduction run.

**Primary result:** `rebuttal_plan/warp_divergence_results.md`  
**Bug log:** `DEBUG.md` (Sections 9–14)  
**Run artifacts:** `benchmark/results/rebuttal/soc_divergence_2x2/<run-id>/`

---

## 1. Machine Requirements

- Julia 1.10.4 (or 1.10.x)
- CUDA Toolkit 12.4–12.6 with `nvcc`, `ncu`, and `nvidia-smi`
- NVIDIA H100 80GB HBM3 (`sm_90`) or A100 (`sm_80`) GPU
- At least 80 GB GPU memory for full candidate pool (4M cones) and workload
- `zstd` for compressing raw CSV files (optional)
- ~20 GB free disk space for 10-seed experiment + NCU profiling artifacts

### 1.1 Environment Check

```bash
nvidia-smi -L
nvidia-smi
nvcc --version
ncu --version
julia --version
```

If `ncu` reports `ERR_NVGPUCTRPERM`, NCU profiling is unavailable but timing
and dmon experiments still work.

---

## 2. Configure Environment

Adjust paths to match your machine. The values below reflect the H100
reproduction machine used on 2026-07-26.

```bash
cd /home/zhenwei/PDCS_fork

# --- ADAPT THESE ---
export CUDA_ROOT=/usr/local/cuda          # CUDA toolkit path
export GPU_PHYSICAL=0                     # nvidia-smi GPU index
export GPU_ARCH=sm_90                     # H100; use sm_80 for A100
export JULIA_BIN=/home/zhenwei/.juliaup/bin/julia
export JULIA_DEPOT=/home/zhenwei/PDCS_fork/.julia-depot
# -------------------

export PATH="$CUDA_ROOT/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export JULIA_DEPOT_PATH="$JULIA_DEPOT"
export PDCS_SKIP_GPU_PRECOMPILE=1

# CRITICAL: Prevents CondaPkg network hangs (DEBUG.md Section 14)
export JULIA_CONDAPKG_OFFLINE=true
```

### 2.1 Verify GPU Visibility

```bash
CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
"$JULIA_BIN" --project=. -e '
using CUDA
CUDA.functional() || error("CUDA not functional")
println("device=", CUDA.device())
println("name=", CUDA.name(CUDA.device()))
println("uuid=", CUDA.uuid(CUDA.device()))
'

nvidia-smi -i "$GPU_PHYSICAL" \
  --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total \
  --format=csv
```

The UUIDs reported by Julia and `nvidia-smi` must match.

---

## 3. Build Kernels

### 3.1 Production Kernels (for timing)

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
```

### 3.2 Diagnostic Kernels (for root-work counters, used by `generate`)

```bash
make -C src/pdcs_gpu/cuda rebuild-profile \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH"
```

### 3.3 Lineinfo PTX (for Nsight Compute source-level profiling)

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH" LINEINFO=-lineinfo
```

Output files:
```
Production:  massive_block_proj.ptx   sufficient_block_proj.ptx
Diagnostic:  massive_block_proj_profile.ptx  sufficient_block_proj_profile.ptx
```

**Never use `*_profile.ptx` for publication timing.**

---

## 4. Run Deterministic CPU Tests

```bash
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
"$JULIA_BIN" --project=. test/test_soc_divergence_2x2_cases.jl
```

Expected: all 20/20 assertions pass.
```
SOC divergence generators are deterministic: 6/6 pass
exact grouped/interleaved layouts: 6/6 pass
parametric-similar generator: 8/8 pass
```

---

## 5. Pre-Warm CondaPkg (CRITICAL)

CondaPkg must be pre-installed before any NCU profiling. Run once:

```bash
CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
JULIA_CONDAPKG_OFFLINE=true \
"$JULIA_BIN" --project=. -e '
using CUDA
CUDA.functional() || error("CUDA not functional")
'
```

If this hangs on "CondaPkg Installing packages", delete the stale lock file:
```bash
rm -f /home/zhenwei/PDCS_fork/.CondaPkg/lock
```
Then set `JULIA_CONDAPKG_OFFLINE=false` temporarily for the first install, and
re-set to `true` afterwards.

---

## 6. Smoke Test (1,024 cones, seed 2026)

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

Verify gates:
```bash
RUN_DIR="$PWD/benchmark/results/rebuttal/soc_divergence_2x2/$SMOKE_ID"
grep -R ',FAIL' "$RUN_DIR" || echo "No failures"
column -s, -t < "$RUN_DIR/iteration/seed2026/manipulation_checks.csv"
column -s, -t < "$RUN_DIR/iteration/seed2026/correctness.csv"
column -s, -t < "$RUN_DIR/branch/seed2026/manipulation_checks.csv"
column -s, -t < "$RUN_DIR/branch/seed2026/correctness.csv"
```

All must show `PASS`.

---

## 7. Full 10-Seed Timing Experiment

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
  --seeds 2026:2035 \
  --experiments iteration,branch \
  --run-id "$FULL_ID"

export PDCS_RUN_DIR="$PWD/benchmark/results/rebuttal/soc_divergence_2x2/$FULL_ID"
```

This runs 10 seeds × 2 experiments = 20 generation+timing pairs. Each seed:
1. Generates a 4M-cone candidate pool and selects 1,048,576 cones
2. Runs diagnostic pass to compute root-work counters
3. Constructs grouped and interleaved layouts
4. Reports manipulation checks
5. Runs 5 warm-up + 10 counterbalanced measured launches per cell
6. Checks correctness (CPU/GPU, cross-strategy agreement)

The runner automatically invokes the analyzer. Results are in:
- `$PDCS_RUN_DIR/seed_effects.csv`
- `$PDCS_RUN_DIR/effect_estimates.csv`
- `$PDCS_RUN_DIR/publication_tables.md`

### 7.1 Re-Analyze Without Re-Running

```bash
"$JULIA_BIN" --project=. benchmark/analyze_soc_divergence_2x2.jl \
  --root "$PDCS_RUN_DIR" \
  --seeds 2026:2035 \
  --bootstrap 10000 \
  --output "$PDCS_RUN_DIR"
```

---

## 8. Nsight Compute Profiling

### 8.1 Prerequisites

- Production PTX rebuilt with `-lineinfo` (Step 3.3)
- `JULIA_CONDAPKG_OFFLINE=true` set (Step 5 — **mandatory**, see DEBUG.md §14)
- Iteration caches exist for target seeds (kept for seeds 2026–2028 by default;
  re-generate others as needed, see §8.4)

### 8.2 Profile 10 Seeds (Iteration Only)

```bash
# Warm up CondaPkg first
CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
JULIA_DEPOT_PATH="$JULIA_DEPOT" \
JULIA_CONDAPKG_OFFLINE=true \
"$JULIA_BIN" --project=. -e 'using CUDA; CUDA.functional()'

# Run NCU profiling (use fixed script)
benchmark/profile_soc_divergence_2x2_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seeds 2026,2027,2028,2029,2030,2031,2032,2033,2034,2035 \
  --experiments iteration
```

Each cell takes ~30–40 seconds. 10 seeds × 4 cells = ~20–25 minutes total.
Output: 40 `.ncu-rep` + 40 `.csv` files in `$PDCS_RUN_DIR/ncu/`.

### 8.3 Profile Branch Experiment (Optional, Same Pattern)

```bash
benchmark/profile_soc_divergence_2x2_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" \
  --seeds 2026:2035 \
  --experiments branch
```

### 8.4 Re-Generate Caches for Additional Seeds

The default runner keeps caches only for seeds 2026–2028. To add seeds:

```bash
for seed in 2029 2030 2031 2032 2033 2034 2035; do
  seed_dir="$PDCS_RUN_DIR/iteration/seed$seed"
  mkdir -p "$seed_dir"
  CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL" \
  JULIA_DEPOT_PATH="$JULIA_DEPOT" \
  JULIA_CONDAPKG_OFFLINE=true \
  "$JULIA_BIN" --project=. \
    benchmark/rescaled_soc_divergence_2x2.jl \
    --experiment iteration --mode generate --seed "$seed" \
    --cone-count 1048576 --cone-dimension 10 --candidate-factor 4 \
    --output-dir "$seed_dir" \
    > "$seed_dir/generate.log" 2>&1
done
```

Each seed generates a ~403 MB `case_cache.jls`. Ensure sufficient disk space.

### 8.5 Extract NCU Metrics

```bash
python3 << 'PYEOF'
import csv, os, statistics

run_dir = os.environ['PDCS_RUN_DIR']
seeds = list(range(2026, 2036))
# ... (see warp_divergence_results.md §9 for full extraction script)
PYEOF
```

---

## 9. Sustained GPU Utilization

### 9.1 Create Julia -O1 Wrapper (LLVM 15 Workaround)

Julia 1.10.4's LLVM 15 SLP vectorizer crashes during GPU kernel JIT compilation
in `--mode duration`. Workaround: run Julia at optimization level `-O1`.

```bash
cat > /tmp/julia_O1_wrapper.sh << 'EOF'
#!/bin/bash
exec /home/zhenwei/.juliaup/bin/julia -O1 "$@"
EOF
chmod +x /tmp/julia_O1_wrapper.sh
```

(Adjust the Julia path to match your machine.)

### 9.2 Run Utilization Collection (Seed 2026, 30 Seconds Each)

```bash
benchmark/run_soc_divergence_2x2_utilization.sh \
  --run-dir "$PDCS_RUN_DIR" \
  --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" \
  --julia /tmp/julia_O1_wrapper.sh \
  --julia-depot "$JULIA_DEPOT" \
  --cone-count 1048576 \
  --cone-dimension 10 \
  --seed 2026 \
  --duration 30 \
  --experiments iteration
```

**IMPORTANT:** Do NOT run other GPU workloads concurrently. The utilization
script uses `nvidia-smi dmon` and will be killed by any `pkill -f rescaled_soc`
issued from another terminal.

Output: 4 `.dmon.txt` files in `$PDCS_RUN_DIR/nvidia_smi/`.

### 9.3 Extend to More Seeds

Change `--seed` for each seed. Requires cached data for that seed
(see §8.4 for cache regeneration).

---

## 10. Complete All-Experiments Launch Order

After smoke gates pass and CondaPkg is pre-warmed:

```bash
FULL_ID="${FULL_ID:-$(date -u +%Y%m%dT%H%M%SZ)_soc_divergence_full}"

# 1. Timing: 10 seeds × iteration + branch
benchmark/run_soc_divergence_2x2.sh \
  --gpu "$GPU_PHYSICAL" --cuda-home "$CUDA_ROOT" --arch "$GPU_ARCH" \
  --julia "$JULIA_BIN" --julia-depot "$JULIA_DEPOT" \
  --seeds 2026:2035 --experiments iteration,branch --run-id "$FULL_ID"

export PDCS_RUN_DIR="$PWD/benchmark/results/rebuttal/soc_divergence_2x2/$FULL_ID"

# 2. NCU profiling: iteration, 10 seeds
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME="$CUDA_ROOT" ARCH="$GPU_ARCH" LINEINFO=-lineinfo

benchmark/profile_soc_divergence_2x2_ncu.sh \
  --run-dir "$PDCS_RUN_DIR" --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" --julia "$JULIA_BIN" \
  --julia-depot "$JULIA_DEPOT" --seeds 2026:2035 --experiments iteration

# 3. Sustained utilization: seed 2026, iteration
benchmark/run_soc_divergence_2x2_utilization.sh \
  --run-dir "$PDCS_RUN_DIR" --gpu "$GPU_PHYSICAL" \
  --cuda-home "$CUDA_ROOT" --julia /tmp/julia_O1_wrapper.sh \
  --julia-depot "$JULIA_DEPOT" --seed 2026 --duration 30 --experiments iteration

# 4. Final analysis (already run by step 1; re-run if needed)
"$JULIA_BIN" --project=. benchmark/analyze_soc_divergence_2x2.jl \
  --root "$PDCS_RUN_DIR" --seeds 2026:2035 --bootstrap 10000 \
  --output "$PDCS_RUN_DIR"
```

Steps 2–3 can run in any order but must NOT run concurrently (single GPU).

---

## 11. Key Bugs Fixed During Reproduction

See `DEBUG.md` for full details.

| # | Bug | Fix | Files |
|:--|:---|:---|:---|
| 12.1 | `--force-overwrite=false` invalid NCU flag | Remove flag | Both `profile_*_ncu.sh` |
| 12.2 | `--nvtx-include` regex doesn't match | Remove `--nvtx` flags | Both `profile_*_ncu.sh` |
| 12.3 | `--target-processes all` tracks CondaPkg | Remove flag | Both `profile_*_ncu.sh` |
| 12.4 | `NVTX.range_push(String)` API mismatch | Use `NVTX.Domain(...)` | `rescaled_soc_divergence_2x2.jl` |
| 12.5 | Stale CondaPkg lock after SIGKILL | `rm -f .CondaPkg/lock` | Workaround |
| 14 | CondaPkg pixi hangs NCU forever | `JULIA_CONDAPKG_OFFLINE=true` | Both `profile_*_ncu.sh` |
| 5 | LLVM 15 SLP vectorizer SIGKILL | `julia -O1` for duration mode | `/tmp/julia_O1_wrapper.sh` |

---

## 12. Differences from Original Machine

| Original assumption | Actual | Resolution |
|:---|:---|:---|
| CUDA 12.6 at `/usr/local/cuda-12.6` | CUDA 12.4 at `/usr/local/cuda` | Use `--cuda-home /usr/local/cuda` |
| Julia at `$PWD/.julia-bin/julia` | Julia at `~/.juliaup/bin/julia` | Use `--julia` flag |
| `.julia-depot` at `$PWD/.julia-depot` | Same location | No change |
| 20 seeds (2026:2045) | 10 seeds (2026:2035) | Per plan §6.1 |
| `--nvtx-include "PDCS_PROJECTION/"` | NVTX filter doesn't match | Removed (use `--kernel-name`) |
| NVTX `range_push(String)` | NVTX.jl v1.0.3 needs `Domain` | Create `NVTX.Domain(...)` |
