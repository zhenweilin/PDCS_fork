# cuPDCS 递进式消融实验结果

> **历史五级结果说明（保留，不删除）：** 本报告的最后一级同时打开 adaptive
> primal–dual weight 与 adaptive common step，因此不能区分两个机制的边际
> 贡献。补充的六级受控实验已经将二者严格分离；新结果见
> `rebuttal_plan/progressive_ablation_six_stage_results.md`。本文件及对应
> 315 条 raw log 继续作为独立历史重复保留，以下数值未被覆盖。

## 1. 完成状态

正式实验已经完成。

- 实例：63 个代表性 CBF 实例；
- 配置：5 个累计配置；
- 正式记录：315/315；
- tolerance：`1e-6`；
- 单实例/配置 time limit：600 秒；
- GPU：物理 GPU 0–7；
- `run_status=COMPLETED`：315/315；
- `OPTIMAL`：220；
- `TIME_LIMIT`：94；
- `INFEASIBLE`：1；
- runtime error：0；
- driver failure：0；
- `result.json`：315/315；
- 非空 `solver.raw.log`：315/315；
- Halpern restart selection：全部为 0。

正式结果目录：

```text
benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730
```

## 2. 五级配置

组件按已有 one-at-a-time 消融所建议的重要性顺序加入：

\[
\text{PDHG}
\rightarrow +\text{restart}
\rightarrow +\text{scaling}
\rightarrow +\text{reflection}
\rightarrow +\text{adaptive step}.
\]

| Stage | Restart | Scaling | Adaptive line search | Adaptive weight | Reflection | Halpern |
|---|---:|---:|---:|---:|---:|---:|
| Pure PDHG | off | off | off | off | off | off |
| + Restart | on | off | off | off | off | off |
| + Diagonal rescaling | on | on | off | off | off | off |
| + Reflection | on | on | off | off | on | off |
| + Adaptive step | on | on | on | on | on | off |

关闭 adaptive step 时使用 power method 估计
\(\lVert G\rVert_2\) 后设置固定步长；因此 pure PDHG 和前三级不存在隐藏的
adaptive line search 或动态 primal–dual weight。

## 3. 总体结果

| Stage | Verified solved | 相邻变化 | SGM(10) wall time | GM iterations | Timeout |
|---|---:|---:|---:|---:|---:|
| Pure PDHG | 37/63 | — | 89.53 s | 49,964 | 26 |
| + Restart | 37/63 | 0 | 86.27 s | 42,570 | 26 |
| + Diagonal rescaling | 45/63 | +8 | 66.10 s | 38,986 | 18 |
| + Reflection | 46/63 | +1 | 57.08 s | 32,370 | 17 |
| + Adaptive step | 55/63 | +9 | 25.22 s | 14,135 | 7 |

从 pure PDHG 到最终配置：

- verified solved 从 37 增加到 55，净增加 18 个实例；
- SGM(10) 从 89.53 秒下降到 25.22 秒，下降 71.8%；
- timeout 从 26 个减少到 7 个。

求解数不是每一步都严格单调地包含上一阶段的 solved set，因此还需要结合 paired
transition 阅读。例如 scaling 阶段净增加 8 个，但实际是新求解 10 个、同时失去
2 个；adaptive step 阶段新求解 10 个、同时失去 1 个。

## 4. 相邻阶段 paired effect

下表中的 ratio 定义为“加入组件后 / 加入组件前”。小于 1 表示加入组件后更快
或迭代更少。

| 加入组件 | Jointly solved | Runtime ratio | 95% bootstrap CI | Iteration ratio | Before-only | After-only |
|---|---:|---:|---:|---:|---:|---:|
| Restart | 37 | 0.9406 | [0.8241, 1.0448] | 0.8520 | 0 | 0 |
| Diagonal rescaling | 35 | 0.7466 | [0.5132, 1.0661] | 0.4960 | 2 | 10 |
| Reflection | 45 | 0.7837 | [0.6814, 0.9039] | 0.7482 | 0 | 1 |
| Adaptive step | 45 | 0.5175 | [0.3761, 0.6936] | 0.2948 | 1 | 10 |

### 4.1 Restart

加入 restart 不改变 solved 数量或 solved set，仍为 37/63。在共同求解的 37 个
实例上：

- 迭代数几何平均减少 14.8%；
- wall time 点估计减少 5.9%；
- wall-time 95% CI 包含 1，因此时间改善未达到统计显著。

Restart 在本递进路径中主要减少迭代，而没有单独扩大 600 秒内的 solved set。
这不与原逐项删除实验中“关闭 restart 后 0/63”矛盾：原 full 配置同时具有
scaling、reflection 和 adaptive step，说明 restart 与后续组件之间存在强交互。

### 4.2 Diagonal rescaling

加入 diagonal rescaling 后：

- solved 从 37 增加到 45，净增加 8；
- 10 个实例由失败变为成功，2 个实例由成功变为失败；
- jointly solved 实例的迭代数减少 50.4%；
- wall-time 点估计减少 25.3%，但 CI `[0.5132, 1.0661]` 包含 1；
- 含失败惩罚的 SGM(10) 从 86.27 秒下降到 66.10 秒。

因此 scaling 对成功率和迭代数有明显贡献，但 paired wall-time 的不确定性较大。

### 4.3 Reflection

加入 reflection 后：

- solved 从 45 增加到 46；
- 没有 before-only solved，新增 1 个成功实例；
- jointly solved 实例的迭代数减少 25.2%；
- wall time 减少 21.6%，95% CI `[0.6814, 0.9039]` 不包含 1。

这是一个统计显著的相邻阶段加速，说明 reflection 在 fixed-step、scaled、
restarted PDHG 上具有独立的时间贡献。

### 4.4 Adaptive step

最后加入 adaptive line search 和 adaptive primal–dual weight 后：

- solved 从 46 增加到 55，净增加 9；
- 新增 10 个成功实例，同时失去 1 个；
- jointly solved 实例的迭代数减少 70.5%；
- wall time 减少 48.3%，95% CI `[0.3761, 0.6936]` 不包含 1；
- SGM(10) 从 57.08 秒下降到 25.22 秒。

在当前递进顺序中，adaptive step 产生最大的最后阶段增益。这说明“按旧单项消融
排序”不等于“递进实验中每一步的边际贡献必然递减”；组件间交互非常重要。

## 5. 求解实例集合的变化

### 5.1 Restart

没有 lost 或 gained instance。

### 5.2 Diagonal rescaling

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
- `strictmin_2D_43_dual`；
- `tls12`；
- `turbine07`。

### 5.3 Reflection

新增 `gp_dave_1`，没有 lost instance。

### 5.4 Adaptive step

Before-only：

- `clay0203h`。

After-only：

- `b1bigflowc`；
- `cx02-100`；
- `db-plane-strain-prism`；
- `enpro48`；
- `fac3`；
- `gams01`；
- `jha88`；
- `nql180`；
- `varun`；
- `wiener_strong_signal100-3-101`。

## 6. 按规模分组的求解数

| Stage | Small（53） | Medium（6） | Large（4） |
|---|---:|---:|---:|
| Pure PDHG | 34 | 2 | 1 |
| + Restart | 34 | 2 | 1 |
| + Diagonal rescaling | 41 | 4 | 0 |
| + Reflection | 42 | 4 | 0 |
| + Adaptive step | 48 | 5 | 2 |

Large 组只有 4 个实例，不能据此单独概括规模趋势。值得注意的是 scaling 和
reflection 阶段在这 4 个实例中均为 0/4，而 adaptive step 恢复到 2/4，进一步
说明组件间存在非单调交互。

## 7. 按 cone mix 分组的求解数

| Stage | EXP without SOC（6） | SOC + EXP（17） | SOC without EXP（40） |
|---|---:|---:|---:|
| Pure PDHG | 4 | 8 | 25 |
| + Restart | 4 | 8 | 25 |
| + Diagonal rescaling | 5 | 8 | 32 |
| + Reflection | 5 | 9 | 32 |
| + Adaptive step | 5 | 14 | 36 |

Scaling 的成功率增益主要来自 SOC without EXP；reflection 在 SOC + EXP 中增加
1 个；adaptive step 对 SOC + EXP 和 SOC without EXP 都有明显增益。

## 8. 为什么仍在本批次重复运行最后一级

最后一级的开关与此前 Halpern-candidate 消融中的
`without_halpern_candidate` 相同。旧运行得到 56/63，本次受控递进运行得到
55/63。

两次运行的 solved set 只相差 `clay0203h`：

- 旧运行：GPU5，实例内顺序第 1，289.23 秒，verified solved；
- 本次运行：GPU1，实例内顺序第 5，602.92 秒，time limit。

这表明该边界实例存在运行顺序或数值轨迹敏感性。因此：

1. 旧结果仍可作为 full 配置的独立重复；
2. 本报告使用本批次 55/63，因为只有它与前一级满足“同一实例、同一 GPU、同一
   Julia 进程、随机配置顺序”的 paired timing 控制；
3. 不应把旧 full wall time 与本批次第 4 级直接拼接计算 paired ratio；
4. 对 full 的成功数可以表述为两次正式运行得到 55–56/63，而本次预注册式受控
   递进结果为 55/63。

这也说明保留最后一级复测是必要的，否则无法发现该单实例重复性差异。

## 9. 主要结论

本实验支持以下结论：

1. 完整组件组合相对 pure PDHG 将 solved 数从 37 提升到 55，并将含失败惩罚的
   SGM(10) 降低 71.8%；
2. restart 单独不增加 solved 数，但减少 14.8% 的 paired iterations；
3. diagonal rescaling 净增加 8 个 solved 实例，并将 paired iterations 减少
   50.4%；
4. reflection 净增加 1 个 solved 实例，并带来统计显著的 21.6% paired
   wall-time 降低；
5. adaptive step 在当前组件组合上净增加 9 个 solved 实例，并带来统计显著的
   48.3% paired wall-time 和 70.5% paired-iteration 降低；
6. 组件贡献具有明显交互性，不能只凭 one-at-a-time 消融推断累计边际贡献。

## 10. 建议写入论文或 rebuttal 的文字

> We additionally performed a cumulative ablation starting from fixed-step
> PDHG and successively adding restart, diagonal rescaling, reflection, and
> adaptive step-size control. The number of verified solves evolved as
> 37, 37, 45, 46, and 55 out of 63, while the failure-penalized SGM(10) time
> decreased from 89.53 to 25.22 seconds. Restart reduced paired iterations by
> 14.8% but did not change the solved set at the first stage. Rescaling added
> eight net solves and reduced paired iterations by 50.4%. Reflection yielded
> a statistically significant 21.6% paired speedup, and the final adaptive-step
> stage yielded a statistically significant 48.3% paired speedup and nine
> additional net solves. These results also reveal important interactions:
> component importance inferred from one-at-a-time deletion does not imply
> monotonically decreasing marginal gains in a cumulative construction.

## 11. 文件

- 自动递进报告：
  `benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730/progressive_report.md`
- 全部结果与 raw-log 路径：
  `benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730/raw_results.csv`
- 总体汇总：
  `benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730/summary_overall.csv`
- 相邻阶段 paired effect：
  `benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730/adjacent_effects.csv`
- 按规模汇总：
  `benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730/summary_by_size.csv`
- 按 cone mix 汇总：
  `benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730/summary_by_cone_mix.csv`
- GPU assignment：
  `benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730/gpu_assignments.csv`
- 环境和源码 hash：
  `benchmark/results/rebuttal/progressive_ablation/progressive_importance_600s_all_idle_20260730/environment.txt`
- 实验设计：
  `rebuttal_plan/progressive_ablation.md`
- 复现说明：
  `rebuttal_plan/how_to_run_progressive_ablation.md`
