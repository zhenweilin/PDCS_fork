# cuPDCS benchmark for represent_data

This directory contains the complete H100 campaign driver for the 62 selected
CBF instances under `../../represent_data`. Only cuPDCS is run. Models are read
with the official `JuMP.read_from_file` API; JumpRW is not used.

The source folder currently contains 63 CBF files. `isil01.cbf.gz`, the case
previously classified separately as `hardest_prob`, is excluded from this
campaign. The generated case list must therefore contain exactly 62 entries or
submission stops.

Scientific settings are fixed by both the submission and compute scripts:

- one Slurm-visible NVIDIA H100 per job;
- Julia 1.10.4;
- absolute and relative tolerance `1e-6`;
- per-instance solver time limit 3,600 seconds;
- strict native grid-wise CUDA projection with its ABI/runtime self-test;
- automatic Int32/Int64 sparse-index selection;
- production `sm_90` CUDA artifacts rebuilt from the current source.

Submit the entire dataset as one Slurm job:

```bash
bash benchmark/represent_data/submit_h100.sh
```

Check the job and result count:

```bash
bash benchmark/represent_data/check_status.sh
```

Results are written incrementally below `results/current/cupdcs`. Every case
has its own `result.toml` and solver log. The log records CUDA memory status
before and after that solve. `results/current/campaign_summary.toml` is written
after all 62 cases have been visited. Existing results are skipped, so an
intentional resubmission resumes the campaign:

```bash
CBF_ALLOW_RESUBMIT=1 bash benchmark/represent_data/submit_h100.sh
```

The default Julia executable, project, and depot are the already prepared
Julia 1.10.4 cuPDCS environment used by the Fisher benchmark. They can be
overridden with `CBF_JULIA_BIN`, `CBF_CUPDCS_PROJECT`, and
`CBF_JULIA_DEPOT` when moving the scripts to another machine.

