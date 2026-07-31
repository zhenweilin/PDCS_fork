# cuPDCS legacy inline-Halpern ablation: archived results

## 1. Completion status

The formal ablation experiment is **COMPLETE**:

- instances: 63 CBF problems from `benchmark/represent_data`;
- configurations per instance: 6;
- formal records: 378/378;
- tolerance: `1e-6`;
- time limit: 600 seconds (10 minutes) per instance/configuration;
- execution: serial execution on GPU 7;
- selected records with infrastructure/runtime errors: 0;
- selected records with a nonempty raw solver log: 378/378.

The complete machine-generated report is:

```text
benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/report.md
```

All selected raw-log paths and result paths are indexed in:

```text
benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/raw_results.csv
```

Every historical attempt was retained. No failed or superseded raw log was
deleted.

## 2. Configurations

`full` enables all five components. Each other configuration disables exactly
one component while holding the other options fixed:

| Configuration | Disabled component |
|---|---|
| `full` | none |
| `no_scaling` | diagonal rescaling |
| `no_adaptive_step` | adaptive update of the common step parameter `eta` |
| `no_restart` | every restart trigger |
| `no_reflection` | reflected/extrapolated primal point |
| `no_halpern` | Halpern averaging |

For `no_adaptive_step`, the fixed common step is selected from the operator norm:

```text
eta = 0.9 / ||G_hat||_2
tau = eta / omega
sigma = eta * omega
```

If the power method reaches its iteration limit, the conservative factor `0.8`
is used instead of `0.9`. This ablation disables adaptive `eta`, not the
primal/dual weight-balancing option.

### 2.1 Scope limitation of the adaptive ablation

cuPDCS contains two distinct adaptive mechanisms:

1. adaptation of the common step parameter `eta`, controlled by
   `use_adaptive_step`;
2. adaptation of the primal/dual step-size weight `omega`, controlled by
   `use_adaptive_step_size_weight`.

The completed `no_adaptive_step` experiment sets:

```text
use_adaptive_step = false
use_adaptive_step_size_weight = true
```

whereas Full sets both options to `true`. Consequently, this experiment isolates
the effect of adaptive `eta` **conditional on adaptive `omega` remaining
enabled**. It does not isolate the effect of adaptive primal/dual weighting and
does not measure the joint effect of disabling both mechanisms.

There is no `no_adaptive_weight` configuration and no 2-by-2 experiment over
the two adaptive switches in this data set. Every occurrence of “adaptive
step” in the interpretation below therefore means only the adaptive update of
`eta`, not the combined adaptive-step and adaptive-weight system.

### 2.2 Completed follow-up separating adaptive weight and adaptive eta

A separate six-stage cumulative experiment now isolates the two adaptive
mechanisms without changing the public solver API. Its weight-only stage maps
to the existing `use_adaptive_step_size_weight` option and keeps
`use_adaptive_step=false`.

On the same 63-instance suite, the controlled sequence is:

```text
PDHG
  -> + restart
  -> + scaling
  -> + reflection
  -> + adaptive primal weight
  -> + adaptive eta
```

The verified solve counts are `37, 37, 44, 46, 55, 56`. Relative to the
reflection stage, adaptive primal weighting alone has a paired wall-time ratio
of `0.4125` with 95% CI `[0.2904, 0.5715]` and adds nine solves. Adding adaptive
`eta` afterwards has a paired wall-time ratio of `1.2677` with 95% CI
`[1.0438, 1.4872]`, reduces paired iterations by 19.6%, and adds one large
instance.

The full follow-up report is:

```text
rebuttal_plan/progressive_ablation_six_stage_results.md
```

## 3. Evaluation rule

A configuration is counted as verified solved only if all three independently
reported measures satisfy `1e-6`:

1. relative primal infeasibility;
2. relative dual infeasibility;
3. relative primal-dual objective gap.

SGM(10) is the shifted geometric mean of wall-clock time. An unsolved or
time-limited record receives the full 600-second penalty. Lower SGM(10) is
better.

The paired runtime and iteration ratios are:

```text
ratio = ablated configuration / full configuration
```

They use only instances solved by both configurations. A ratio larger than one
means that removing the component made performance worse. The SGM statistic,
which penalizes every failure, should be read together with the jointly-solved
ratio to avoid survivorship bias.

## 4. Overall results

| Configuration | Verified solved | SGM(10), seconds | GM iterations |
|---|---:|---:|---:|
| `full` | 53/63 | 30.90 | 27,623 |
| `no_scaling` | 43/63 | 64.75 | 38,741 |
| `no_adaptive_step` | 53/63 | 30.38 | 40,460 |
| `no_restart` | 0/63 | 600.00 | NA |
| `no_reflection` | 53/63 | 34.50 | 32,742 |
| `no_halpern` | 56/63 | 24.34 | 15,334 |

## 5. Paired effects relative to Full

| Ablation | Jointly solved | Runtime ratio | 95% bootstrap CI | Iteration ratio | Full only | Ablation only |
|---|---:|---:|---:|---:|---:|---:|
| `no_scaling` | 42 | 1.481 | [1.168, 1.892] | 1.760 | 11 | 1 |
| `no_adaptive_step` | 51 | 0.789 | [0.692, 0.930] | 1.291 | 2 | 2 |
| `no_restart` | 0 | NA | NA | NA | 53 | 0 |
| `no_reflection` | 52 | 1.174 | [1.071, 1.314] | 1.207 | 1 | 1 |
| `no_halpern` | 53 | 0.640 | [0.559, 0.736] | 0.449 | 0 | 3 |

## 6. Results by problem size

The suite contains 53 small, 6 medium, and 4 large instances.

| Configuration | Small solved / SGM | Medium solved / SGM | Large solved / SGM |
|---|---:|---:|---:|
| `full` | 47/53 / 21.82 | 4/6 / 72.84 | 2/4 / 384.48 |
| `no_scaling` | 39/53 / 47.95 | 2/6 / 246.32 | 2/4 / 333.83 |
| `no_adaptive_step` | 47/53 / 21.45 | 5/6 / 52.51 | 1/4 / 565.96 |
| `no_restart` | 0/53 / 600.00 | 0/6 / 600.00 | 0/4 / 600.00 |
| `no_reflection` | 46/53 / 24.87 | 5/6 / 75.38 | 2/4 / 413.00 |
| `no_halpern` | 49/53 / 17.36 | 5/6 / 52.33 | 2/4 / 276.32 |

Each cell reports `verified solved / SGM(10) seconds`.

## 7. Results by cone mix

| Configuration | EXP, no SOC | SOC + EXP | SOC, no EXP |
|---|---:|---:|---:|
| `full` | 5/6 | 12/17 | 36/40 |
| `no_scaling` | 4/6 | 10/17 | 29/40 |
| `no_adaptive_step` | 5/6 | 13/17 | 35/40 |
| `no_restart` | 0/6 | 0/17 | 0/40 |
| `no_reflection` | 5/6 | 13/17 | 35/40 |
| `no_halpern` | 5/6 | 14/17 | 37/40 |

## 8. Interpretation of each component

### 8.1 Diagonal rescaling: strong empirical benefit

Disabling rescaling reduces the verified solve count from 53 to 43 and more
than doubles penalized SGM time from 30.90 to 64.75 seconds. Full solves 11
instances that `no_scaling` does not solve, whereas the reverse occurs for only
one instance. On the 42 jointly solved cases, removing scaling increases
runtime by 48.1% and iterations by 76.0%; the runtime confidence interval
excludes one.

The results strongly support diagonal rescaling as both a robustness and an
efficiency component.

### 8.2 Adaptive eta: fewer iterations, but no overall wall-time gain

Full and `no_adaptive_step` both solve 53/63 instances. On jointly solved
instances, fixed-step execution uses 29.1% more iterations but takes only 78.9%
of Full's wall time. The adaptive `eta` procedure therefore improves progress
per accepted iteration, but its line-search/trial overhead is not recovered in
overall time on this suite. Adaptive primal/dual weight balancing is enabled in
both configurations and is not responsible for the measured contrast.

The size breakdown is informative: Full solves two of four large instances,
whereas fixed step solves one. Thus adaptive stepping may improve robustness on
the largest cases even though fixed stepping is faster on the suite overall.
This one-at-a-time experiment does not support a claim that adaptive stepping
universally reduces wall-clock time. By itself it also cannot determine whether
adaptive `omega` alone is beneficial or how much of the full adaptive system's
effect comes from `eta` versus `omega`. The completed six-stage follow-up in
Section 2.2 supplies that separate contrast.

### 8.3 Restart: essential under the tested time limit

All 63 `no_restart` runs reach the 600-second limit without satisfying all three
`1e-6` verification criteria. Full solves 53 of these instances. This is the
largest ablation effect and shows that restart is essential for the tested
solver configuration.

### 8.4 Reflection: modest but statistically supported efficiency benefit

Full and `no_reflection` both solve 53/63 instances. On the 52 jointly solved
instances, removing reflection increases wall time by 17.4% and iterations by
20.7%. The 95% runtime-ratio interval `[1.071, 1.314]` excludes one.

Reflection therefore provides a consistent, moderate efficiency improvement,
although its solve-count effect is neutral in aggregate.

### 8.5 Halpern averaging: no empirical performance benefit in this suite

The current results do not support Halpern averaging as a performance
improvement. `no_halpern` solves 56 instances, compared with 53 for Full, and
there are three ablation-only solves with no Full-only solves. On jointly solved
instances, removing Halpern reduces wall time by 36.0% and iterations by 55.1%.

This result should be reported rather than hidden. Possible follow-up work is to
tune the Halpern schedule or study its interaction with restart and reflection,
but the completed one-factor ablation does not justify claiming that the
current Halpern implementation improves empirical performance.

## 9. Reviewer-facing conclusion

The experiment does **not** show that every optional component is beneficial in
every metric:

- diagonal rescaling and restart provide strong robustness improvements;
- reflection gives a moderate and statistically supported efficiency benefit;
- adaptive `eta`, conditional on adaptive `omega` remaining enabled, reduces
  iteration count and appears more robust on the largest cases, but fixed
  `eta` is faster overall;
- the tested Halpern averaging implementation is slower and solves fewer
  instances than its ablation.

This is a more informative conclusion than claiming uniform benefit from every
component. It distinguishes components that are essential for robustness from
components whose current value is iteration efficiency, problem-dependent
behavior, or theoretical structure. The present one-factor data do not by
themselves support a separate conclusion about adaptive primal/dual weight
balancing; that question is answered by the six-stage follow-up linked in
Section 2.2.

## 10. Audit trail and retained attempts

Two infrastructure issues were encountered and corrected without deleting old
data:

1. The first six `ck_n25_m10_o1_1` attempts solved normally but the runner read
   `result_metrics` through an MOI bridge, causing result-extraction errors.
   Metrics are now read from the cuPDCS backend. The six `attempt_001`
   directories remain intact; the selected valid records are `attempt_002`.
2. `integrated_KleinesBeispiel_02_6/no_scaling` produced non-finite diagnostic
   values. JSON has no NaN representation, so two attempts stopped while
   writing `result.json.tmp`. The runner now serializes non-finite numerical
   diagnostics as JSON `null`, which the analyzer treats as unavailable. The
   successful formal record is `attempt_003`; all earlier raw logs remain.

Final validation checks:

```text
records=378
unique instance/configuration pairs=378
configurations with 63 records=6
selected runtime errors=0
selected nonempty raw logs=378
report status=COMPLETE
```

## 11. Result and reproduction files

- `benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/report.md`
- `benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/raw_results.csv`
- `benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/summary_overall.csv`
- `benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/summary_by_size.csv`
- `benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/summary_by_cone_mix.csv`
- `benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2/paired_effects.csv`
- `rebuttal_plan/ablation_study.md`
- `rebuttal_plan/how_to_run_ablation_study.md`
