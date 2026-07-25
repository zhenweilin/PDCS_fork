# Comprehensive rebuttal experiments

This directory implements the executable experiment package specified in
`rebuttal_plan/additional_experiments.md`.

## Entry points

- `reproduce_all.sh`: smoke or full unprofiled reproduction.
- `run_h100.sh`, `run_a100.sh`: architecture-specific wrappers.
- `soc_strategy_map.jl`: paired count/dimension strategy grid.
- `soc_root_profile.jl`, `exp_root_profile.jl`: controlled timing matrices.
- `profile_nsys.sh`, `run_nsys_matrix.sh`: sustained utilization traces.
- `profile_ncu.sh`: one warmed kernel hardware profile.
- `application_trace.jl`: executes an application command manifest and requires
  each solver command to write the documented trace CSV.

Run a smoke test:

```bash
benchmark/rebuttal/reproduce_all.sh \
  --gpu 0 --cuda-home /usr/local/cuda-12.6 --arch sm_90 \
  --run-id h100_smoke
```

Run the full unprofiled package:

```bash
benchmark/rebuttal/run_h100.sh \
  --gpu 0 --cuda-home /usr/local/cuda-12.6 \
  --run-id h100_full --full
```

Add `--profiles` only on a machine with profiler permission.

## Diagnostic-counter status contract

The profile PTX files have a fixed `RootProfileRecord` ABI and are built
separately from production PTX. Counter fields initialized to `-1` mean “not
collected” and must never be summarized as zero iterations. The present
production projection functions do not expose all internal root-search events;
therefore manuscript iteration-distribution claims require a run in which the
diagnostic consumer confirms that all required fields are nonnegative. Nsight
branch/source counters do not substitute for missing per-cone records.

## Application manifest

This repository does not contain the manuscript Fisher, Lasso, and MPO
instance generators. Replace the deliberately failing commands in
`application_manifest.csv` with the exact commands from the manuscript
artifact. Each command must write the CSV path supplied in
`PDCS_APPLICATION_TRACE_OUTPUT`. A missing trace is recorded and is not treated
as a successful experiment.
