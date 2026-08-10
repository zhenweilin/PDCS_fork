# Recover and test the historical `pdhg_clp` GPU solver

This directory contains the isolated environment, recovery/build scripts,
launchers, full raw logs, JSON records, summaries, and analysis for the six
difficult exponential-cone cases. No CPU solver is used in the GPU-history
experiments.

The six cases are `batch`, `batchs101006m`, `batchs121208m`,
`batchs151208m`, `batchs201210m`, and `enpro56`. They are the union of the
failed EXP cases from the current No-Halpern and Inline-Halpern 2,100-instance
comparison.

## 1. Recover the private Gitee history and compile the kernels

The credential file is read only through `GIT_ASKPASS`; its contents are not
placed in a command line, remote URL, Git configuration, or log. Keep it
private:

```bash
cd /home/zhenwei/PDCS_fork
chmod 600 external/gitee.txt
```

Recover the final historical GPU source used by the paper-era branch and build
for an H100 with CUDA 12.6:

```bash
external/old_can_solve_scripts/recover_pdhg_clp_gpu.sh \
  --credential-file external/gitee.txt \
  --commit 3432afc69f96891168c13c926d08cdaacd5b0e41 \
  --worktree external/old_can_solve_scripts/worktrees/pdhg_clp_gpu_3432afc \
  --nvcc /usr/local/cuda-12.6/bin/nvcc \
  --arch sm_90
```

If `external/pdhg_clp` is absent, the script clones
`https://gitee.com/zhenweilin/pdhg_clp.git` first; if it already exists, it
fetches all branches and tags. Both paths use `GIT_ASKPASS`, so the username and
password never appear in the remote URL or command line. Use `--remote-url` and
`--repo` only if the repository location differs on another machine.

Commit `3432afc` has an upstream two-line compilation typo: two loops still
refer to `total_thread` after the variable was renamed `total_threads`. The
recovery script applies the recorded patch
`patches/pdhg_clp_3432afc_compile_total_threads.patch` only in the detached
worktree. The original repository and current PDCS source remain unchanged.

For a different GPU, change `--arch` to its compute capability. For example,
use `sm_80` for A100.

For provenance only, the old Gitee `origin/gpu_branch` tip can be recovered as:

```bash
external/old_can_solve_scripts/recover_pdhg_clp_gpu.sh \
  --credential-file external/gitee.txt \
  --commit 80cb4a85d87e1bee19f2bbb19468fa2682f15da7 \
  --worktree external/old_can_solve_scripts/worktrees/pdhg_clp_gpu_branch_80cb4a8 \
  --nvcc /usr/local/cuda-12.6/bin/nvcc \
  --arch sm_90
```

That branch is an early side branch, not a newer version of `3432afc`. It uses
the old `l2` stopping rule and contains a subsequently fixed diagonal EXP
projection expression, so it must not be used as a correctness baseline or
restored into the current solver.

## 2. Install the isolated Julia environment

The environment uses CUDA.jl 5.11.3 so that it can run with the CUDA 13.2
driver on this machine while loading the archived solver source directly.

```bash
cd /home/zhenwei/PDCS_fork
JULIA_DEPOT_PATH="$PWD/.julia-depot:" \
  .julia-bin/julia-1.10.4/bin/julia --startup-file=no \
  --project=external/old_can_solve_scripts \
  external/old_can_solve_scripts/install.jl
```

## 3. Smoke test

The historical formal scripts used the `MOI.Utilities.Model` +
`MOI.read_from_file` path. `--model-loader moi` reproduces that path.

```bash
external/old_can_solve_scripts/run_all.sh \
  --old-source-dir external/old_can_solve_scripts/worktrees/pdhg_clp_gpu_3432afc \
  --input-root "$HOME/PDCS_CBLIB" \
  --gpus 0 \
  --time-limit 10 \
  --tolerance 1e-6 \
  --print-frequency 2000 \
  --julia-threads 14 \
  --use-aggressive true \
  --model-loader moi \
  --output-dir external/old_can_solve_scripts/results/smoke_3432afc
```

## 4. Formal six-case GPU run

Use six idle GPUs to run all cases concurrently. The present 1-hour comparison
uses 3,600 seconds:

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
  --model-loader moi \
  --output-dir external/old_can_solve_scripts/results/pdhg_clp_3432afc_gpu_hardcases_3600s
```

The archived paper-release script actually allowed `3600 * 5 = 18,000`
seconds at tolerance `1e-6`. To reproduce that historical budget, change
`--time-limit 3600` to `--time-limit 18000`.

`run_all.sh` records the source commit/status and hashes the Julia/CUDA source,
PTX files, and shared library. A case is counted as verified only if the solver
reports `OPTIMAL` and all three printed `l_inf` primal residual, `l_inf` dual
residual, and relative gap values are at most the requested tolerance. Each new
case JSON/raw log also records the input SHA-256 and archived source commit, so
the data and source identity remain auditable if the directory is copied.

## 5. Result layout

```text
results/<run>/
  environment.txt
  old_source_commit.txt
  old_source_status.txt
  old_source_sha256.txt
  summary.csv
  report.md
  cases/<case>/launcher.log
  cases/<case>/solver.raw.log
  cases/<case>/result.json
```

Run the summarizer again without changing raw logs:

```bash
python3 external/old_can_solve_scripts/summarize.py \
  external/old_can_solve_scripts/results/<run>
```

The recovery findings and experiment comparison are recorded in
`GPU_HISTORY_RECOVERY.md`. The supplied standalone `batch` log is discussed
separately in `BATCH_OLDLOG_ANALYSIS.md` because that log follows the old
non-aggressive/CPU-style path and is not evidence for a GPU implementation.
