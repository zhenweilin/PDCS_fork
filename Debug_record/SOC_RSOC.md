# Debug Record: JuMP/MOI Incorrectly Bridged SOC to RSOC

Date: 2026-07-29

## Conclusion

The Lasso model contains the following constraints:

```julia
@constraint(model, t == (x[1] + x[2]) / sqrt(2))
@constraint(model, u == (x[1] - x[2]) / sqrt(2))
@constraint(model, [t; u; x[3:3+m]] in SecondOrderCone())
```

This is a standard second-order cone (SOC), not a rotated second-order cone
(RSOC). The reason cuPDCS previously failed to solve this case was not the Lasso
formulation or the SOC projection itself. The root cause was the
MathOptInterface (MOI) capability declaration in PDCS.

Both the CPU and GPU PDCS optimizers advertised support for
`VectorAffineFunction{Float64}`-in-`RotatedSecondOrderCone`. However, the
current RSOC projection execution path still contains `not implemented`
placeholders, so RSOC must not be exposed to JuMP/MOI as a supported native
capability.

## How the Error Occurred

When using `Model(PDCS_GPU.Optimizer)` directly, JuMP initially represents the
constraint above as `VectorOfVariables`-in-`SecondOrderCone`. PDCS does not
declare direct support for this function type; it only declares support for
affine-vector cone constraints.

Because PDCS incorrectly advertised support for both VAF-in-SOC and
VAF-in-RSOC, the MOI bridge system selected `SOCtoRSOCBridge`. It transformed
the user's SOC constraint into an RSOC constraint before passing it to PDCS.
The solver then entered the incomplete RSOC execution path, causing
stagnation, oscillation, or divergence.

The old workflow succeeded because the model was first built with another
solver and then written to and read from CBF. After the CBF round trip, the cone
constraint was already represented as
`VectorAffineFunction`-in-`SecondOrderCone`, so PDCS entered the SOC path
directly. This behavior initially hid the incorrect capability declaration.

## Fix

`MOI.RotatedSecondOrderCone` was removed from the
`MOI.supports_constraint` declarations in both wrappers:

- `src/pdcs_gpu/MOI_wrapper/MOI_wrapper.jl`
- `src/pdcs_cpu/MOI_wrapper/MOI_wrapper.jl`

Support for `MOI.SecondOrderCone` remains enabled. The underlying historical
RSOC data structures were not removed; this fix only withdraws the inaccurate
public capability declaration to keep the change narrowly scoped.

After the fix:

1. A normal JuMP SOC constraint is converted by the function-conversion bridge
   into VAF-in-SOC, which PDCS genuinely supports. It is no longer transformed
   into RSOC.
2. If a user explicitly constructs an RSOC constraint, PDCS no longer claims
   to handle it natively. With bridges enabled, JuMP/MOI may convert it to a
   supported SOC representation.
3. `test/test_moi_cone_support.jl` was added to verify permanently that both
   CPU and GPU optimizers report SOC capability as `true` and native RSOC
   capability as `false`.

## Regression Tests

Test conditions:

- GPU1 only.
- `verbose = 2`.
- Time limit: 600 seconds.
- Problems generated entirely in memory, without writing CBF or other problem
  data files.
- The ablation experiment running on GPU7 was not touched.

### Capability Test

Running `test/test_moi_cone_support.jl` passed all four checks:

| Optimizer | VAF-in-SOC | VAF-in-RSOC |
|---|---:|---:|
| `PDCS_CPU.Optimizer` | `true` | `false` |
| `PDCS_GPU.Optimizer` | `true` | `false` |

### Old Lasso Case on the Current Version

The current working tree at commit
`9935b8f999dc2679aab22b608fa95e4aa49457e1` was tested using the generation
rules from the old script:

- Seed: 1
- `m = 300`
- `n = 1000`
- Density: `1e-3`
- `nnz(A) = 281`
- `lambda = 0.09362283511485615`
- The SOC slice from the old script was preserved: `z[3:(3+m)]`

The model summary clearly reported:

```text
Number of soc cone constraints: 1
Number of rsoc cone constraints: 0
```

Final result:

| Metric | Result |
|---|---:|
| Termination status | `OPTIMAL` |
| Iterations | 4100 |
| Solver time | 5.9027 s |
| Wall time, including first-time compilation and initialization | 29.9637 s |
| Relative primal residual (L-inf) | `3.7432e-9` |
| Relative dual residual (L-inf) | `2.8843e-8` |
| Relative gap | `1.4834e-7` |
| Conic objective | `0.05245150300670077` |
| Reconstructed Lasso objective | `0.05245151127507635` |

This confirms that the failure of the current version on this Lasso problem
has been identified and fixed. After correcting the capability declaration,
the original JuMP `SecondOrderCone()` formulation converges through the SOC
path without a CBF round trip or manual construction of a
`VectorAffineFunction`.

### Repository Benchmark Driver

The original JuMP SOC formulation in
`benchmark/large_scale_lasso/solve_table5_instance.jl` was also tested using
`pilot-r01` from the fixed pilot manifest:

- `m = 100`
- `n = 1000`
- Density: `0.01`
- `nnz(A) = 1027`
- Seed: `1456099608458829617`

This model also reported `SOC=1, RSOC=0` and produced:

| Metric | Result |
|---|---:|
| Termination status | `OPTIMAL` |
| Primal status | `FEASIBLE_POINT` |
| Iterations | 4100 |
| Solver time | 6.0219 s |
| Relative primal residual (L-inf) | `7.3194e-8` |
| Relative dual residual (L-inf) | `1.3035e-7` |
| Relative gap | `3.4143e-7` |
| Objective | `0.13532836471005838` |

## Branch Comparison

To isolate the solver algorithm from the effect of the MOI bridge, the same
old Lasso case was run on four historical branches using an explicitly forced
VAF-in-SOC representation. The first three branches solved the problem
successfully, confirming that the SOC projection and the Lasso formulation
were not the problem:

| Branch | Commit | Result | Iterations | Solver time | Relative primal residual (L-inf) | Relative dual residual (L-inf) | Relative gap |
|---|---|---|---:|---:|---:|---:|---:|
| `paper_release` | `e095ae5` | `OPTIMAL` | 300 | 6.257 s | `1.8757e-10` | `2.5069e-10` | `2.9138e-9` |
| `gpu` | `3432afc` | `OPTIMAL` | 300 | 5.954 s | `1.8011e-10` | `2.3845e-10` | `2.8745e-9` |
| `gpu_branch` | `80cb4a8` | `OPTIMAL` | 300 | 5.065 s | `2.7524e-10` | `2.0662e-10` | `1.1631e-9` |
| `add_warm_start` | `e983d99` | Diverged; diagnostic run stopped early | 350000 | About 158 s | About `1.477e17` | About `9.689e14` | About `1.0` |

The SOC path in `add_warm_start` belongs to an earlier algorithm version and
has an independent numerical-divergence issue. It does not change the MOI
bridge root cause described here.

As a counterexample, before the fix, passing the original JuMP model directly
to these optimizers produced a model summary with `SOC=0, RSOC=1`. The solver
then entered the incorrect RSOC path and oscillated, stalled, or diverged.
