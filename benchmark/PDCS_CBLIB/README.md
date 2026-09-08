# cuPDCS benchmark for PDCS_CBLIB

This directory contains the complete H100 campaign driver for the 2,100 CBF
instances under `../../PDCS_CBLIB`. Only cuPDCS is run. Models are read with
the official `JuMP.read_from_file` API; JumpRW is not used.

The campaign deliberately excludes any file below a `hardest_prob` directory.
The current input tree contains exactly 2,100 included CBF files. A count
mismatch is fatal, so a silently incomplete or unexpectedly changed dataset
cannot be submitted.

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
bash benchmark/PDCS_CBLIB/submit_h100.sh
```

Check the job and result count:

```bash
bash benchmark/PDCS_CBLIB/check_status.sh
```

Results are written incrementally below `results/current/cupdcs`. Every case
has its own `result.toml` and solver log. The log records CUDA memory status
before and after that solve. `results/current/campaign_summary.toml` is written
after all 2,100 cases have been visited. Existing results are skipped, so an
intentional resubmission resumes the campaign:

```bash
CBF_ALLOW_RESUBMIT=1 bash benchmark/PDCS_CBLIB/submit_h100.sh
```

The default Julia executable, project, and depot are the already prepared
Julia 1.10.4 cuPDCS environment used by the Fisher benchmark. They can be
overridden with `CBF_JULIA_BIN`, `CBF_CUPDCS_PROJECT`, and
`CBF_JULIA_DEPOT` when moving the scripts to another machine.

