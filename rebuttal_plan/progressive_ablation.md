# cuPDCS 递进式消融实验设计

## 1. 目的

本实验从 pure PDHG 出发，按照已有 one-at-a-time 消融结果所反映的重要性依次
加入组件：

\[
\text{PDHG}
\rightarrow +\text{restart}
\rightarrow +\text{diagonal rescaling}
\rightarrow +\text{reflection}
\rightarrow +\text{adaptive primal weight}
\rightarrow +\text{adaptive step}.
\]

与逐项删除实验相比，递进式实验直接回答“从基础 PDHG 开始，每加入一个组件，
求解成功数、运行时间和迭代数如何变化”。

## 2. 排序依据

原逐项删除实验在 63 个代表性实例上得到：

- 关闭 restart：verified solved 从 53/63 降到 0/63；
- 关闭 diagonal rescaling：从 53/63 降到 43/63；
- 关闭 reflection：求解总数不变，但 jointly solved 实例的时间增加；
- 旧实验只消融 adaptive `eta`，没有单独消融 adaptive primal weight。

因此本实验将 restart、scaling、reflection、adaptive primal weight 和 adaptive
step 按上述顺序加入。这个顺序是已有实验给出的经验重要性顺序，不是理论上的
全局唯一排序；组件之间仍可能存在交互作用。

## 3. 六级配置

| 配置名 | Restart | Scaling | Adaptive line search | Adaptive weight | Reflection | Halpern |
|---|---:|---:|---:|---:|---:|---:|
| `pdhg` | off | off | off | off | off | off |
| `pdhg_restart` | on | off | off | off | off | off |
| `pdhg_restart_scaling` | on | on | off | off | off | off |
| `pdhg_restart_scaling_reflection` | on | on | off | off | on | off |
| `pdhg_restart_scaling_reflection_adaptive_primal_weight` | on | on | off | on | on | off |
| `pdhg_restart_scaling_reflection_adaptive` | on | on | on | on | on | off |

Halpern candidate 在全部六级配置中关闭，避免把上一组 Halpern 实验引入本次组件
顺序。

这里将两个 adaptive 机制严格分开：

1. adaptive primal weight：固定 \(\eta\)，只动态更新
   \(\omega\)，从而调整 \(\tau=\eta/\omega\) 与
   \(\sigma=\eta\omega\) 的比例；
2. adaptive step：在 adaptive \(\omega\) 已打开的基础上，再打开
   line-search 对公共步长 \(\eta\) 的更新。

在 adaptive step 关闭时，求解器使用 power method 估计
\(\lVert G\rVert_2\)，并采用固定步长

\[
\eta =
\begin{cases}
0.9/\lVert G\rVert_2, & \text{power method converged},\\
0.8/\lVert G\rVert_2, & \text{otherwise}.
\end{cases}
\]

因此 `pdhg` 不包含隐式的 adaptive line search 或动态 primal–dual weighting。

## 4. 正式实验设置

- 数据：`benchmark/represent_data` 中固定的 63 个 CBF 实例；
- tolerance：`1e-6`；
- 每个实例/配置 time limit：600 秒；
- 正式记录：\(63\times6=378\)；
- GPU：启动时检测到的全部空闲 GPU；
- 空闲阈值：显存占用小于 2048 MiB 且 GPU utilization 不高于 10%；
- 每张 GPU 同时最多运行一个 worker；
- 每个实例的六种配置始终在同一张 GPU、同一个 Julia 进程中运行；
- 每个实例内部随机化六种配置的顺序，随机种子为 `20260730`；
- 所有 `solver.raw.log`、`result.json` 和历史 retry attempt 均保留。

同一实例固定在同一张 GPU 上可以控制 GPU 型号和运行环境差异；实例内随机化配置
顺序可以降低 warm-up、温度和时间漂移对某一个固定阶段的系统性偏置。

## 5. 主要指标

### 5.1 Verified solved

只有同时满足下列三个条件才计为 verified solved：

\[
r_{\mathrm p,\infty}^{\mathrm{rel}}\le 10^{-6},\qquad
r_{\mathrm d,\infty}^{\mathrm{rel}}\le 10^{-6},\qquad
\mathrm{relative\ gap}\le 10^{-6}.
\]

### 5.2 SGM(10) wall time

未求解、timeout 或缺失记录统一赋值 600 秒后计算 shifted geometric mean：

\[
\operatorname{SGM}_{10}(t_1,\ldots,t_n)
=
\exp\left(
\frac1n\sum_{i=1}^{n}\log(t_i+10)
\right)-10.
\]

### 5.3 相邻阶段 paired effect

分别比较：

1. `+restart / PDHG`；
2. `+scaling / previous stage`；
3. `+reflection / previous stage`；
4. `+adaptive primal weight / previous stage`；
5. `+adaptive step / previous stage`。

运行时间与迭代数比值均定义为：

\[
\text{ratio} =
\frac{\text{after adding component}}
     {\text{before adding component}}.
\]

ratio 小于 1 表示加入组件后更快或迭代更少。运行时间比同时报告 10,000 次
bootstrap 的 95% confidence interval。

## 6. 输出

正式运行目录包含：

- `raw_results.csv`：全部正式记录与 raw-log 路径；
- `summary_overall.csv`：六级总体统计；
- `summary_by_size.csv`：按 small/medium/large 分组；
- `summary_by_cone_mix.csv`：按 cone mix 分组；
- `adjacent_effects.csv`：五个相邻阶段的 paired effect；
- `gpu_assignments.csv`：每个实例实际使用的物理 GPU；
- `report.md`：通用汇总；
- `progressive_report.md`：递进式汇总；
- `cases/<instance>/<configuration>/attempt_NNN/solver.raw.log`：完整 solver
  raw log。

## 7. 已完成正式批次

六级正式批次已经完成 378/378 条记录：

```text
benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730
```

完整分析与 reviewer-facing 结论见：

```text
rebuttal_plan/progressive_ablation_six_stage_results.md
```
