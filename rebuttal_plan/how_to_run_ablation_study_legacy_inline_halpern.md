# How to reproduce the legacy inline-Halpern ablation study

This file reproduces the five-component ablation experiment on the 63 CBF
instances in `benchmark/represent_data`.

## Formal settings

```text
configurations = full, no_scaling, no_adaptive_step, no_restart,
                 no_reflection, no_halpern
time limit     = 600 seconds per instance/configuration
tolerance      = 1e-6
order seed     = 20260728
formal tasks   = 63 × 6 = 378
```

All configurations for an instance run serially on the same selected GPU.
Configuration order is deterministically randomized within each instance.
Every attempt has a separate `solver.raw.log`; old attempts are not deleted or
overwritten.

## 1. Enter the repository

```bash
cd /home/zhenwei/PDCS_fork
```

The formal run uses the repository-local Julia:

```bash
./.julia-bin/julia-1.12.6/bin/julia --version
```

Install the versions recorded by `Manifest.toml` if the machine has not been
prepared:

```bash
./.julia-bin/julia-1.12.6/bin/julia --project=. -e '
using Pkg
Pkg.instantiate()
'
```

MOSEK is not used by this experiment. If an unrelated MOSEK build step fails
after the PDCS dependencies have been installed, it does not affect the
ablation runner.

## 2. Build production GPU kernels

For an H100 with the system CUDA 13.2 toolkit:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME=/usr/local/cuda-13.2 \
  ARCH=sm_90
```

For a CUDA 12.6 installation:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME=/usr/local/cuda-12.6 \
  ARCH=sm_90
```

Do not use `rebuild-profile` or any `*_profile.ptx` file for this timing
experiment.

## 3. Run correctness gates

Check the independent reflection/Halpern combinations:

```bash
PDCS_SKIP_GPU_PRECOMPILE=1 \
JULIA_PKG_PRECOMPILE_AUTO=0 \
CUDA_VISIBLE_DEVICES=7 \
./.julia-bin/julia-1.12.6/bin/julia \
  --startup-file=no \
  --compiled-modules=existing \
  -O1 \
  --project=. \
  test/test_ablation_update_gpu.jl
```

Expected final marker:

```text
ABLATION_UPDATE_GPU_PASS
```

The dataset inspector must report 63 instances and 378 tasks:

```bash
RUN_DIR=benchmark/results/rebuttal/ablation/preflight

python3 benchmark/ablation/inspect_represent_data.py \
  --input-dir benchmark/represent_data \
  --output "${RUN_DIR}/manifest.csv" \
  --run-order "${RUN_DIR}/run_order.csv" \
  --order-seed 20260728 \
  --expected-count 63
```

Expected inventory:

```text
SIZE_COUNTS {'small': 53, 'medium': 6, 'large': 4}
CONE_MIX_COUNTS {'soc_without_exp': 40, 'soc_exp': 17,
                 'exp_without_soc': 6}
RUN_ORDER_COMPLETE tasks=378
```

## 4. Select an idle GPU

```bash
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
```

Use one GPU with low utilization and enough free memory. The formal run on this
machine used physical GPU 7. The wrapper refuses a selected GPU at startup if
it already uses at least 2048 MiB or has utilization above 10%.

## 5. Launch or resume all experiments

```bash
cd /home/zhenwei/PDCS_fork

bash benchmark/ablation/run_ablation.sh \
  --input-dir benchmark/represent_data \
  --output-dir benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2 \
  --run-id ablation_600s_20260728_r2 \
  --gpu 7 \
  --julia ./.julia-bin/julia-1.12.6/bin/julia \
  --time-limit 600 \
  --tolerance 1e-6 \
  --order-seed 20260728 \
  --resume true
```

The same command is safe for resume:

- a case with a valid case-level `DONE` file is skipped;
- an interrupted attempt is retained;
- a retry is written to a new `attempt_XXX` directory;
- raw logs from earlier attempts remain unchanged.

Do not reuse the r1 directory for publication results.
`ablation_600s_20260728_r1` is a retained partial diagnostic run made before
the termination-candidate export fix.

## 6. Monitor progress

Count formal result records:

```bash
find benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/cases \
  -path '*/attempt_*/result.json' -type f | wc -l
```

Count raw logs:

```bash
find benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/cases \
  -path '*/attempt_*/solver.raw.log' -type f | wc -l
```

Inspect the selected GPU:

```bash
nvidia-smi -i 7
```

Inspect one instance driver:

```bash
tail -f \
  benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/\
instance_100_0_1_w.driver.log
```

Inspect one complete raw solver log:

```bash
less \
  benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/\
cases/100_0_1_w/full/attempt_001/solver.raw.log
```

## 7. Regenerate result tables

The batch wrapper runs the analyzer automatically. It can also be run
manually:

```bash
python3 benchmark/ablation/analyze_ablation.py \
  --run-dir \
    benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2 \
  --tolerance 1e-6 \
  --timeout-value 600 \
  --sgm-shift 10 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260728 \
  --report \
    benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/report.md
```

The report must say `COMPLETE (378/378 formal records)` before it is used in
the paper.

## 8. Output layout and raw logs

```text
benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/
├── environment.txt
├── manifest.csv
├── run_order.csv
├── raw_results.csv
├── summary_overall.csv
├── summary_by_size.csv
├── summary_by_cone_mix.csv
├── paired_effects.csv
├── report.md
├── warmup.raw.log
├── instance_<instance>.driver.log
└── cases/
    └── <instance>/
        └── <configuration>/
            ├── DONE
            └── attempt_001/
                ├── DONE
                ├── result.json
                └── solver.raw.log
```

`solver.raw.log` is the complete solver output for that attempt. A solve is
counted as verified only when the exported point satisfies all three common
checks:

```text
l_inf relative primal residual <= 1e-6
l_inf relative dual residual   <= 1e-6
relative duality gap           <= 1e-6
```

Native `OPTIMAL` without these three checks is not counted as solved.
