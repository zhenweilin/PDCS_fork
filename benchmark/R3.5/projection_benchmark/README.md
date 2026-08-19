# PDCS projection stress benchmark

This directory is the isolated workspace for selecting and validating the GPU
projection implementation used by PDCS. The benchmark covers the cone families
in the requested experiments: SOC, primal exponential, and dual exponential.
It intentionally contains no RSOC cases.

The accepted run is
`results/20260815_final_newton2_v3_branch_complete_full`: 881/881
valid-reference rows pass, 16 deliberately extreme reference-invalid
diagnostics are separated, three pathological forced-grid layouts are
`NOT_APPLICABLE`, and there are zero runtime errors, non-finite candidate
outputs, or correctness-gate violations. Its strict branch gate additionally
passes all 216/216 family/branch/scale/hierarchy Cartesian cells.

## What is tested

`case_matrix.jl` generates three tiers:

| Tier | Cases | Purpose |
|---|---:|---|
| `smoke` | 12 | fast API, correctness, profiling-ABI and artifact check |
| `quick` | 612 | optimization loop and hierarchy crossover fitting |
| `full` | 900 | final stress/regression and complete branch gate |

The full matrix includes:

- closed-form, boundary, polar, decreasing-root, increasing-root and nearly
  degenerate inputs;
- cold, prior-root, perturbed-prior-root and deliberately bad warm starts;
- SOC dimensions around 8/9, 32/33, 149/150, 1999/2000, dimension 201, and
  large dimensions up to 105589;
- cone-count thresholds around 8, 64, 256, 1000 and 60000;
- identity, scalar and diagonal scaling, condition numbers through `1e12`, and
  input amplitudes from `1e-12` through `1e12`;
- bit-identical same-input comparisons of grid-, block-, warp- and thread-wise
  mappings where each mapping is meaningful;
- same-input grid/block/warp/thread comparisons on solver-shaped cone
  inventories from the difficult represent_data cases (structured cones
  retain exact counts/dimensions; simple cones are combined into the two
  leading blocks used by the GPU solver):
  `ravem`, `gams01`, `batch`, `batchs101006m`, `varun`,
  `integrated_KleinesBeispiel_02_6`, `qssp180`, `cx02-100`,
  `db-plane-strain-prism`, and `joint_FC_12`.

Same-input hierarchy variants are assigned as a group to one GPU by a stable
hash of `input_key`, so grid/block/warp/thread timings are never compared across
different cards. Each timing row also checks finiteness, agreement with the general thread-wise
mapping when that reference is feasible, cone feasibility, idempotence, and
nonexpansiveness. Diagnostic PTX is run separately from timing and records
oracle calls, interval expansions, bisections, Newton attempts/accepts,
warm-start attempts/accepts, vector reductions, maximum-iteration events, and
nonfinite outputs.

The full correctness gate uses the requested sample count everywhere except
for grid-wise solver-shaped layouts. Those deliberately losing host/cuBLAS
comparators use the first candidate call as their single timing sample and a
second call for idempotence. They retain the independent thread reference,
agreement, finiteness, feasibility, and idempotence gates; the redundant
perturbed-input and profile repeats are explicitly marked absent in the
`validation_scope` column. Every synthetic grid case retains the full timing,
perturbation, and profiling protocol. Stable timing for the very slow real
rows comes from the dedicated same-input hierarchy A/B; this prevents a
known-bad mapping from occupying one GPU for hours. Three solver-shaped rows
that combine more than 10,000 structured cones with a dimension above 10,000
(`qssp180`, `db-plane-strain-prism`, and `joint_FC_12`) are recorded as
`NOT_APPLICABLE` for forced grid execution. This is an explicit selector
constraint, not a correctness pass; their block/warp/thread rows remain in the
matrix, and grid correctness is covered by the synthetic dimension/count
surface.

## Reproducible artifacts and execution

Build all root-search variants without overwriting solver artifacts:

```bash
benchmark/R3.5/projection_benchmark/build_variants.sh all
```

The variants form a cumulative A/B ladder: pure bisection, safeguarded Newton,
fused root oracle, fused initial cone tests, hybrid log coordinates, and the
experimental EXP reciprocal pair. Every directory contains production and
diagnostic PTX plus the grid-wise shared library.

Run one process on each of four GPUs, with one task per card:

```bash
benchmark/R3.5/projection_benchmark/run_four_gpu.sh final_newton2_v3 full
```

Run the dedicated same-input four-level A/B on the ten difficult
solver-shaped layouts:

```bash
benchmark/R3.5/projection_benchmark/run_represent_hierarchy_ab.sh
```

Merge shards and generate the strategy table:

```bash
python3 benchmark/R3.5/projection_benchmark/summarize_results.py \
  'benchmark/R3.5/projection_benchmark/results/20260815_final_newton2_v3_branch_complete_full/shard*.csv' \
  --output-dir benchmark/R3.5/projection_benchmark/results/20260815_final_newton2_v3_branch_complete_full/summary

python3 benchmark/R3.5/projection_benchmark/verify_branch_coverage.py \
  'benchmark/R3.5/projection_benchmark/results/20260815_final_newton2_v3_branch_complete_full/shard*.csv' \
  --output-dir benchmark/R3.5/projection_benchmark/results/20260815_final_newton2_v3_branch_complete_full/branch_coverage

python3 benchmark/R3.5/projection_benchmark/compare_selectors.py \
  'benchmark/R3.5/projection_benchmark/results/20260815_final_newton2_v3_branch_complete_full/shard*.csv' \
  --output-dir benchmark/R3.5/projection_benchmark/results/20260815_final_newton2_v3_branch_complete_full/selector_comparison
```

Pair dispatch-on/off measurements only when they came from the same GPU and
the same artifact:

```bash
python3 benchmark/R3.5/projection_benchmark/summarize_dispatch.py \
  'benchmark/R3.5/projection_benchmark/results/dispatch_ab/**/*.csv' \
  --output-dir benchmark/R3.5/projection_benchmark/results/dispatch_ab/summary
```

For any two compiled artifacts, `compare_variants.py` similarly requires
same-case, same-GPU pairs and reports both latency and root-work counters.
`compare_full_stack.py` additionally reconstructs the base-commit selector and
directly compares its measured old-artifact row with the accepted selector's
measured final-artifact row. The accepted full-stack result is `2.241x` by
geometric mean over 119 same-device, same-input groups.

## Acceptance loop

1. All candidate outputs must pass the correctness gate; a faster invalid row
   is discarded.
2. Same-input crossover rows determine candidate hierarchy boundaries.
3. Root-search variants are compared on latency and diagnostic work, since
   fewer bisections can still be slower when each derivative oracle costs more.
4. Candidate thresholds are applied to `select_projection_strategy` and the
   heterogeneous dispatch planner.
5. The full 900-case matrix is rerun. Regressions around a boundary reject or
   narrow the rule.
6. The selected rule is run through the difficult real PDCS cases and finally
   the 62-instance represent_data campaign. Only a rule that preserves solver
   convergence and improves end-to-end projection time becomes the default.

Result directories are append-only. Failed or interrupted runs remain present
and are identified by their own metadata/status rather than being deleted.
