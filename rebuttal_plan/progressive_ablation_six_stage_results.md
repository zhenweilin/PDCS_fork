# cuPDCS 六级递进式消融实验结果

## 1. 完成状态

将 adaptive primal–dual weight 与 adaptive common step 严格分离后的正式实验
已经完成：

- 数据：`benchmark/represent_data` 中 63 个代表性 CBF 实例；
- 配置：6 个累计配置；
- 正式记录：378/378；
- tolerance：`1e-6`；
- 单实例/配置 time limit：600 秒；
- GPU：NVIDIA H100 80GB HBM3；
- 初始 worker：物理 GPU 2、3、4、5、7；
- 运行中 GPU 6 释放后，通过共享 claim 锁安全增加第六个 worker；
- `run_status=COMPLETED`：378/378；
- `OPTIMAL`：275；
- `TIME_LIMIT`：101；
- `INFEASIBLE`：2，二者均未通过独立精度核验，按未求解处理；
- runtime error：0；
- driver failure：0；
- `result.json`：378/378；
- 非空 `solver.raw.log`：378/378；
- retry attempt：0；
- Halpern restart selection：全部为 0。

正式结果目录：

```text
benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730
```

实验从 `2026-07-30T18:14:34Z` 运行到 `2026-07-30T22:28:00Z`。六张
GPU 分别处理 10–11 个完整实例；同一实例的六个配置始终在同一物理 GPU 和同一
Julia 进程内运行。

## 2. 六级配置

本实验不增加新的 solver 公共 API。实验名称 `adaptive_primal_weight` 映射到
已有兼容开关：

```text
use_adaptive_step_size_weight
```

两个 adaptive 机制的含义为：

1. adaptive primal weight：动态更新 primal–dual weight
   \(\omega\)，但公共步长 \(\eta\) 固定；
2. adaptive step：在动态 \(\omega\) 已打开的基础上，再打开 line search，
   动态更新 \(\eta\)。

六级配置为：

| Stage | Restart | Scaling | Adaptive \(\eta\) | Adaptive \(\omega\) | Reflection | Halpern |
|---|---:|---:|---:|---:|---:|---:|
| Pure PDHG | off | off | off | off | off | off |
| + Restart | on | off | off | off | off | off |
| + Diagonal rescaling | on | on | off | off | off | off |
| + Reflection | on | on | off | off | on | off |
| + Adaptive primal weight | on | on | off | on | on | off |
| + Adaptive step | on | on | on | on | on | off |

关闭 adaptive \(\eta\) 时，求解器根据 power method 得到的算子范数使用固定公共
步长。第五级因此只测试动态 \(\omega\)，不会隐式打开 adaptive line search。

## 3. 评价规则

只有下列三个量都不大于 `1e-6` 才计为 verified solved：

\[
r_{\mathrm p,\infty}^{\mathrm{rel}}\le 10^{-6},\qquad
r_{\mathrm d,\infty}^{\mathrm{rel}}\le 10^{-6},\qquad
\mathrm{relative\ gap}\le 10^{-6}.
\]

未求解、time limit、低精度不可行退出或缺失记录均以 600 秒计入
SGM(10)。相邻阶段 ratio 定义为：

\[
\text{ratio} =
\frac{\text{加入组件后的值}}
     {\text{加入组件前的值}}.
\]

运行时间与迭代数 ratio 只使用前后阶段都 verified solved 的实例。ratio 小于
1 表示加入组件后更快或迭代更少。

## 4. 总体结果

| Stage | Verified solved | 相邻净变化 | SGM(10) wall time | GM iterations | Time limit | Unverified infeasible |
|---|---:|---:|---:|---:|---:|---:|
| Pure PDHG | 37/63 | — | 90.13 s | 49,964 | 26 | 0 |
| + Restart | 37/63 | 0 | 86.87 s | 42,557 | 26 | 0 |
| + Diagonal rescaling | 44/63 | +7 | 69.09 s | 38,459 | 19 | 0 |
| + Reflection | 46/63 | +2 | 55.41 s | 31,932 | 17 | 0 |
| + Adaptive primal weight | 55/63 | +9 | 22.41 s | 18,164 | 7 | 1 |
| + Adaptive step | 56/63 | +1 | 24.22 s | 14,902 | 6 | 1 |

从 pure PDHG 到最终配置：

- verified solved 从 37 增加到 56，净增加 19 个实例；
- SGM(10) 从 90.13 秒降低到 24.22 秒，降低 73.1%；
- verified-solved 实例的 GM iterations 从 49,964 降到 14,902。

两个 `INFEASIBLE` 退出都来自 `isil01` 的最后两级，solver 状态为
`primal_infeasible_low_acc`。二者的独立 verification metric 接近 1，因此均
为 `verified_solved=false`；它们没有抬高表中的求解数。

## 5. 相邻阶段 paired effect

| 加入组件 | Jointly solved | Runtime ratio | 95% bootstrap CI | Iteration ratio | Before-only | After-only |
|---|---:|---:|---:|---:|---:|---:|
| Restart | 37 | 0.9281 | [0.8116, 1.0396] | 0.8518 | 0 | 0 |
| Diagonal rescaling | 35 | 0.7604 | [0.5064, 1.1770] | 0.5421 | 2 | 9 |
| Reflection | 44 | 0.7074 | [0.5592, 0.8484] | 0.6807 | 0 | 2 |
| Adaptive primal weight | 46 | 0.4125 | [0.2904, 0.5715] | 0.3290 | 0 | 9 |
| Adaptive step | 55 | 1.2677 | [1.0438, 1.4872] | 0.8044 | 0 | 1 |

### 5.1 Restart

加入 restart 不改变 solved 数量或 solved set，仍为 37/63。对共同求解的
37 个实例：

- paired iterations 减少 14.8%；
- wall-time 点估计减少 7.2%；
- wall-time 95% CI 包含 1，因此本阶段没有统计显著的时间改善。

这不与 one-at-a-time 实验中 restart 的强作用矛盾。后者是在 scaling、
reflection 和 adaptive 机制已打开的情况下删除 restart，说明 restart 与后续
组件存在明显交互。

### 5.2 Diagonal rescaling

加入 scaling 后：

- solved 从 37 增加到 44，净增加 7；
- 9 个实例由失败变为成功，2 个实例由成功变为失败；
- paired iterations 减少 45.8%；
- wall-time 点估计减少 24.0%，但 CI 包含 1；
- 含失败惩罚的 SGM(10) 从 86.87 秒降到 69.09 秒。

因此 scaling 的主要证据是成功率、惩罚后总体时间和迭代数改善；共同求解实例的
wall-time 不确定性仍较大。

### 5.3 Reflection

加入 reflection 后：

- solved 从 44 增加到 46；
- 前一级的 44 个 solved 实例全部保留，并新增 2 个；
- paired iterations 减少 31.9%；
- paired wall time 减少 29.3%；
- 95% CI `[0.5592, 0.8484]` 不包含 1。

因此 reflection 在 fixed-\(\eta\)、fixed-\(\omega\)、scaled、restarted
PDHG 上带来统计显著的独立加速。

### 5.4 Adaptive primal weight

第五级只打开动态 \(\omega\)，保持 \(\eta\) 固定。结果为：

- solved 从 46 增加到 55，净增加 9；
- 前一级 46 个 solved 实例全部保留；
- paired iterations 减少 67.1%；
- paired wall time 减少 58.8%；
- 95% CI `[0.2904, 0.5715]` 明确低于 1；
- SGM(10) 从 55.41 秒降到 22.41 秒，降低 59.6%。

这是当前递进路径中最大的单步成功率和 wall-time 增益。旧五级实验把动态
\(\omega\) 和动态 \(\eta\) 同时加入，无法确定增益来源；本实验表明大部分最后
阶段增益来自 adaptive primal–dual weight，而不是 adaptive \(\eta\)。

### 5.5 Adaptive common step

最后一级在动态 \(\omega\) 已打开的基础上，再打开 line-search 更新
\(\eta\)。结果为：

- solved 从 55 增加到 56；
- weight-only 阶段的 55 个 solved 实例全部保留；
- 新增求解大规模实例 `nql180`；
- paired iterations 减少 19.6%；
- paired wall time增加 26.8%；
- 95% CI `[1.0438, 1.4872]` 高于 1；
- SGM(10) 从 22.41 秒增加到 24.22 秒，增加 8.1%。

因此 adaptive \(\eta\) 在本套实例上提高迭代效率并增加一个大规模求解实例，
但 line-search/trial overhead 没有转化为总体 wall-time 加速。不能宣称
adaptive \(\eta\) 在该实验中普遍更快；更准确的结论是它提供一定鲁棒性和较少
迭代，而 weight adaptation 提供主要的时间收益。

## 6. Solved set 的相邻变化

### 6.1 Restart

没有 before-only 或 after-only solved instance。

### 6.2 Diagonal rescaling

Before-only：

- `nql180`；
- `wiener_strong_signal100-3-101`。

After-only：

- `100_0_1_w`；
- `10_std`；
- `as_conic_100_100_hard_set1_1_cap10`；
- `clay0203h`；
- `du-opt`；
- `flay02h`；
- `rsyn0805h`；
- `tls12`；
- `turbine07`。

### 6.3 Reflection

新增：

- `gp_dave_1`；
- `strictmin_2D_43_dual`。

没有 before-only instance。

### 6.4 Adaptive primal weight

新增：

- `b1bigflowc`；
- `cx02-100`；
- `db-plane-strain-prism`；
- `enpro48`；
- `fac3`；
- `gams01`；
- `jha88`；
- `varun`；
- `wiener_strong_signal100-3-101`。

没有 before-only instance。

### 6.5 Adaptive common step

新增 `nql180`，没有 before-only instance。

## 7. 按规模分组

| Stage | Small（53） | Medium（6） | Large（4） |
|---|---:|---:|---:|
| Pure PDHG | 34 | 2 | 1 |
| + Restart | 34 | 2 | 1 |
| + Diagonal rescaling | 41 | 3 | 0 |
| + Reflection | 42 | 4 | 0 |
| + Adaptive primal weight | 49 | 5 | 1 |
| + Adaptive step | 49 | 5 | 2 |

Large 组只有 4 个实例，不应单独概括规模趋势。adaptive \(\eta\) 的唯一新增
solve 来自 Large 组，这支持“增加一定鲁棒性”而非“普遍 wall-time 加速”的
表述。

## 8. 按 cone mix 分组

| Stage | EXP without SOC（6） | SOC + EXP（17） | SOC without EXP（40） |
|---|---:|---:|---:|
| Pure PDHG | 4 | 8 | 25 |
| + Restart | 4 | 8 | 25 |
| + Diagonal rescaling | 5 | 8 | 31 |
| + Reflection | 5 | 9 | 32 |
| + Adaptive primal weight | 5 | 14 | 36 |
| + Adaptive step | 5 | 14 | 37 |

adaptive primal weight 的 9 个净新增 solve 分别来自 SOC + EXP 的 5 个实例和
SOC without EXP 的 4 个实例。adaptive \(\eta\) 的 1 个新增 solve 来自
SOC without EXP。

## 9. 与旧五级递进实验的关系

旧五级实验的最后一级同时打开 adaptive \(\omega\) 和 adaptive \(\eta\)，因此
只能测量二者的联合贡献。该历史批次及其 315 条 raw log 均保留在：

```text
benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730
```

本次六级实验从头重跑全部 378 条，而没有把一个新阶段拼接到旧批次。这样每个
相邻 ratio 都使用同批次、同实例、同 GPU、同 Julia 进程内的受控配对时间。

旧五级最终配置求解 55/63；本次六级最终配置求解 56/63。两个 solved set 只相差
`clay0203h`，说明该边界实例存在重复运行敏感性。本报告使用本次完整六级批次，
旧数据继续作为独立重复保留。

## 10. Reviewer-facing conclusion

本次补充实验解决了旧消融中没有严格区分两个 adaptive 机制的问题：

1. adaptive primal–dual weighting 是主要性能组件：在 reflection 之后单独
   打开它，verified solved 增加 9，paired wall time 减少 58.8%，且
   confidence interval 明确低于 1；
2. adaptive common step 再增加 1 个大规模 solve，并减少 19.6% paired
   iterations，但 paired wall time 增加 26.8%；
3. 因此不应把旧五级最后阶段的全部增益都归因于 adaptive line search；
4. 从 pure PDHG 到完整、无 Halpern 的最终配置，verified solved 从 37 增加到
   56，SGM(10) 降低 73.1%。

可用于论文或 rebuttal 的英文表述：

> We further separated the two adaptive mechanisms in a six-stage cumulative
> ablation. Enabling only adaptive primal-dual weighting, while keeping the
> common step size fixed, increased the verified solve count from 46 to 55 out
> of 63. On the 46 jointly solved instances, its paired wall-time ratio was
> 0.412 (95% bootstrap CI [0.290, 0.572]) and its iteration ratio was 0.329.
> Enabling adaptive common-step line search afterwards retained all 55 solves
> and added one large instance, but its paired wall-time ratio was 1.268
> (95% CI [1.044, 1.487]) despite reducing paired iterations by 19.6%. Thus,
> the dominant performance gain of the previously combined adaptive stage
> comes from primal-dual weight balancing; common-step adaptation improves
> iteration efficiency and robustness on one additional large instance, but
> does not reduce overall wall time on this suite.

## 11. 完整性审计

最终自动审计结果：

```text
result_json=378
unique instance/configuration pairs=378
instances=63, each with 6 configurations
each configuration=63 records
run_status=COMPLETED for 378/378
runtime_errors=0
driver_failures=0 on GPUs 2,3,4,5,6,7
nonempty solver.raw.log=378/378
retry attempt=0
nonzero Halpern selection count=0
resolved adaptive flag mismatches=0
CUDA_VISIBLE_DEVICES / gpu_assignments mismatches=0
report status=COMPLETE
```

物理 GPU 的实例分配数为：

```text
GPU 2: 10
GPU 3: 11
GPU 4: 11
GPU 5: 10
GPU 6: 10
GPU 7: 11
```

## 12. 结果与复现文件

- 自动六级报告：
  `benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730/progressive_report.md`
- 全部结果与 raw-log 路径：
  `benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730/raw_results.csv`
- 总体汇总：
  `benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730/summary_overall.csv`
- 相邻阶段 paired effect：
  `benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730/adjacent_effects.csv`
- 按规模汇总：
  `benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730/summary_by_size.csv`
- 按 cone mix 汇总：
  `benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730/summary_by_cone_mix.csv`
- GPU assignment：
  `benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730/gpu_assignments.csv`
- 环境、源码 hash 与初始 GPU 状态：
  `benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730/environment.txt`
- 实验设计：
  `rebuttal_plan/progressive_ablation.md`
- 复现说明：
  `rebuttal_plan/how_to_run_progressive_ablation.md`
- 历史五级结果：
  `rebuttal_plan/progressive_ablation_results.md`
