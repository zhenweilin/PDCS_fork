# How to reproduce the current cuPDCS core ablation study

## 1. Current experiment definition

The current one-at-a-time study has six configurations:

```text
full
no_scaling
no_adaptive_step
no_adaptive_primal_weight
no_restart
no_reflection
```

All six configurations use:

```text
use_halpern = false
```

Thus Halpern is not a factor in this experiment. Its candidate-only comparison
is reproduced separately by
`rebuttal_plan/how_to_run_halpern_candidate_ablation.md`.

Common settings:

```text
instances      = 63
time limit     = 600 seconds per instance/configuration
tolerance      = 1e-6
composed tasks = 63 × 6 = 378
```

The old `ablation_600s_20260728_r2` run used the removed inline-Halpern main
sequence and must not be used for current figures or conclusions. Its
reproduction guide is archived as
`how_to_run_ablation_study_legacy_inline_halpern.md`.

## 2. Enter the repository

```bash
cd /home/zhenwei/PDCS_fork
./.julia-bin/julia-1.12.6/bin/julia --version
```

If necessary:

```bash
./.julia-bin/julia-1.12.6/bin/julia --project=. -e '
using Pkg
Pkg.instantiate()
'
```

## 3. Build production GPU kernels

For H100 and CUDA 13.2:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME=/usr/local/cuda-13.2 \
  ARCH=sm_90
```

For CUDA 12.6:

```bash
make -C src/pdcs_gpu/cuda rebuild-gpu \
  CUDA_HOME=/usr/local/cuda-12.6 \
  ARCH=sm_90
```

Do not use profile PTX files for formal timing.

## 4. Verify the six switch definitions

```bash
PDCS_SKIP_GPU_PRECOMPILE=1 \
JULIA_PKG_PRECOMPILE_AUTO=0 \
./.julia-bin/julia-1.12.6/bin/julia \
  --startup-file=no \
  --compiled-modules=existing \
  -O1 \
  --project=. \
  test/test_one_at_a_time_ablation_flags.jl
```

Expected:

```text
ONE_AT_A_TIME_ABLATION_FLAGS_PASS
```

The test verifies that every configuration changes exactly one core switch and
that all six configurations have `use_halpern=false`.

## 5. Reuse two completed progressive configurations

Do not rerun `full` or `no_adaptive_step`. They already exist in:

```text
benchmark/results/rebuttal/progressive_ablation/
progressive_six_stage_600s_all_idle_20260730
```

Mappings:

```text
full
  <- pdhg_restart_scaling_reflection_adaptive

no_adaptive_step
  <- pdhg_restart_scaling_reflection_adaptive_primal_weight
```

Both source configurations contain 63 records and have Halpern disabled.

## 6. Run the four missing configurations

First inspect idle GPUs:

```bash
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

Then launch:

```bash
bash benchmark/progressive_ablation/run_progressive_ablation.sh \
  --experiment one_at_a_time_core_missing4 \
  --configs no_scaling,no_adaptive_primal_weight,no_restart,no_reflection \
  --run-id one_at_a_time_core_missing4_600s_all_idle_20260731 \
  --output-dir \
    benchmark/results/rebuttal/ablation/one_at_a_time_core_missing4_600s_all_idle_20260731 \
  --gpus auto \
  --julia ./.julia-bin/julia-1.12.6/bin/julia \
  --time-limit 600 \
  --tolerance 1e-6 \
  --order-seed 20260731 \
  --resume true
```

The same command safely resumes an interrupted run. Existing attempts and raw
logs are never overwritten.

Expected formal records:

```text
63 × 4 = 252
```

Monitor:

```bash
MISSING_RUN=benchmark/results/rebuttal/ablation/\
one_at_a_time_core_missing4_600s_all_idle_20260731

find "$MISSING_RUN/cases" \
  -mindepth 3 -maxdepth 3 -name DONE -type f | wc -l
```

The final count must be 252.

## 7. Compose the six-configuration result

```bash
python3 benchmark/ablation/compose_core_one_at_a_time.py \
  --missing-run \
    benchmark/results/rebuttal/ablation/one_at_a_time_core_missing4_600s_all_idle_20260731 \
  --output-dir \
    benchmark/results/rebuttal/ablation/one_at_a_time_core_600s_20260731 \
  --tolerance 1e-6 \
  --time-limit 600 \
  --bootstrap-seed 20260731
```

The composition script:

- verifies identical instance hashes in both source batches;
- verifies the complete expected flag vector for all 378 selected records;
- rejects any record with Halpern enabled;
- creates relative symbolic links instead of copying logs;
- records every mapping in `source_map.csv`;
- runs the standard analyzer.

Check:

```bash
FINAL_RUN=benchmark/results/rebuttal/ablation/\
one_at_a_time_core_600s_20260731

grep -F 'COMPLETE** (378/378 formal records)' "$FINAL_RUN/report.md"
column -s, -t "$FINAL_RUN/summary_overall.csv"
```

## 8. Generate figures

```bash
python3 benchmark/ablation/plot_one_at_a_time_ablation.py \
  --run-dir \
    benchmark/results/rebuttal/ablation/one_at_a_time_core_600s_20260731 \
  --output-dir rebuttal_plan/figures \
  --prefix one_at_a_time_ablation \
  --dpi 300
```

Outputs:

```text
rebuttal_plan/figures/one_at_a_time_ablation_solved.{tex,pdf,png}
rebuttal_plan/figures/one_at_a_time_ablation_sgm.{tex,pdf,png}
rebuttal_plan/figures/one_at_a_time_ablation_paired_effects.{tex,pdf,png}
```

## 9. Result provenance

The final composed directory contains:

```text
manifest.csv
source_map.csv
raw_results.csv
summary_overall.csv
summary_by_size.csv
summary_by_cone_mix.csv
paired_effects.csv
report.md
cases/
```

`raw_results.csv` points to the original nonempty `solver.raw.log` and
`result.json` files. No source result is deleted or duplicated.

The solve-count comparison is directly valid across the identical 63-instance
suite. Cross-batch paired wall-time ratios should be treated as secondary
because four configurations and the reused full baseline were run in separate
controlled batches. The progressive experiment supplies the strict
within-batch paired timing evidence.
