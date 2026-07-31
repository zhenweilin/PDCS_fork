# cuPDCS 当前核心组件消融实验结果

更新日期：2026-07-31
状态：**正式实验完成（378/378）**
数据集：`benchmark/represent_data` 中 63 个实例
精度：primal residual、dual residual 和 relative gap 均不超过 `1e-6`
时间上限：每个实例/配置 600 秒（10 分钟）

## 1. 实验口径

本报告只研究当前求解器的五个核心机制：

1. diagonal rescaling；
2. adaptive common step \(\eta\)；
3. adaptive primal-dual weight \(\omega\)；
4. restart；
5. reflection。

本次 one-at-a-time 实验共有六个配置：

| Configuration | Scaling | Adaptive \(\eta\) | Adaptive \(\omega\) | Restart | Reflection | Halpern |
|---|---:|---:|---:|---:|---:|---:|
| `full` | on | on | on | on | on | off |
| `no_scaling` | off | on | on | on | on | off |
| `no_adaptive_step` | on | off | on | on | on | off |
| `no_adaptive_primal_weight` | on | on | off | on | on | off |
| `no_restart` | on | on | on | off | on | off |
| `no_reflection` | on | on | on | on | off | off |

六个配置的 `use_halpern` 均为 `false`。因此本报告不含旧 inline-Halpern
主序列，也不把 Halpern 作为第七个 one-at-a-time 配置。Halpern restart
candidate 由独立的二配置实验报告。

## 2. 数据来源

用户指出 `full` 和去掉 adaptive \(\eta\) 的配置已经在 progressive 正式批次中
完成，因此本次没有重复运行这两列：

| Final configuration | Source configuration | Source batch |
|---|---|---|
| `full` | `pdhg_restart_scaling_reflection_adaptive` | progressive |
| `no_adaptive_step` | `pdhg_restart_scaling_reflection_adaptive_primal_weight` | progressive |
| 其余四项 | 同名配置 | one-at-a-time missing-4 |

源目录：

```text
benchmark/results/rebuttal/progressive_ablation/
  progressive_six_stage_600s_all_idle_20260730

benchmark/results/rebuttal/ablation/
  one_at_a_time_core_missing4_600s_all_idle_20260731
```

最终统一视图：

```text
benchmark/results/rebuttal/ablation/
  one_at_a_time_core_600s_20260731
```

组合脚本验证了两个源批次具有完全相同的 63 个 instance id 和 SHA-256。
最终目录通过 378 个相对符号链接引用原始记录，不复制或删除 raw logs；
`source_map.csv` 记录了每一列的来源。

## 3. 总体结果

| Configuration | Verified solved / 63 | SGM(10) wall time (s) | GM iterations（solved） | Time limit | Other failure |
|---|---:|---:|---:|---:|---:|
| `full` | **56** | 24.22 | 14,902 | 6 | 1 infeasible |
| `no_scaling` | 44 | 56.61 | 22,650 | 18 | 1 numerical error |
| `no_adaptive_step` | 55 | **22.41** | 18,164 | 7 | 1 infeasible |
| `no_adaptive_primal_weight` | 47 | 59.92 | 25,333 | 16 | 0 |
| `no_restart` | 45 | 63.35 | 22,506 | 18 | 0 |
| `no_reflection` | 55 | 27.35 | 15,455 | 8 | 0 |

SGM(10) 给所有未通过三项 `1e-6` 验证的记录赋值 600 秒，因此没有丢弃失败
case。六个配置均有 63 条正式记录，solver/infrastructure runtime error 为 0。

主要观察：

- Full 求解数最高，为 56/63。
- 去掉 scaling 后少解 12 个；去掉 restart 后少解 11 个；固定
  \(\omega\) 后少解 9 个。这三个机制对鲁棒性的贡献最明显。
- 固定 \(\eta\) 和去掉 reflection 均少解 1 个。它们仍有正贡献，但求解数上的
  边际贡献小于 scaling、restart 和 adaptive \(\omega\)。
- `no_adaptive_step` 的 SGM(10) 比 Full 略低，但它在共同求解实例上需要更多
  迭代，且少解一个实例。因此 adaptive \(\eta\) 主要改善鲁棒性和迭代数，
  不能简单表述为“在本批次中降低了 wall time”。

## 4. 与 Full 的配对结果

下表的 ratio 定义为 Ablated / Full；大于 1 表示消融后更慢或需要更多迭代。

| Ablation | Jointly solved | Wall-time ratio | 95% bootstrap CI | Iteration ratio | Full-only solved | Ablation-only solved |
|---|---:|---:|---:|---:|---:|---:|
| No scaling | 43 | 1.878 | [1.422, 2.556] | 2.696 | 13 | 1 |
| Fixed \(\eta\) | 55 | 0.789 | [0.671, 0.961] | 1.243 | 1 | 0 |
| Fixed \(\omega\) | 47 | 2.246 | [1.617, 3.161] | 2.760 | 9 | 0 |
| No restart | 45 | 2.409 | [1.737, 3.380] | 2.849 | 11 | 0 |
| No reflection | 55 | 1.104 | [0.990, 1.255] | 1.113 | 1 | 0 |

计时解释必须区分数据来源：

- Full 与 Fixed \(\eta\) 来自同一个 progressive 正式批次，因此这一对的
  wall-time ratio 是严格的同批次配对结果。
- No scaling、Fixed \(\omega\)、No restart 和 No reflection 来自第二天在同一
  台 H100 机器上的 missing-4 正式批次。它们使用相同实例、时限、精度和随机化
  规则，但与 Full 的跨批次 wall-time ratio 只作为辅助证据。
- 求解成功数、失败类型和最终精度不依赖计时是否同批次，可以直接比较。
- 严格的逐阶段同批次计时证据见 progressive ablation 报告。

## 5. 按问题规模分层

| Size | Full | No scaling | Fixed \(\eta\) | Fixed \(\omega\) | No restart | No reflection |
|---|---:|---:|---:|---:|---:|---:|
| Small（53） | **49** | 40 | **49** | 43 | 42 | 48 |
| Medium（6） | **5** | 2 | **5** | 3 | 2 | **5** |
| Large（4） | **2** | **2** | 1 | 1 | 1 | **2** |

中等规模实例最清楚地显示 scaling 和 restart 的作用：Full 求解 5/6，
而 No scaling 和 No restart 均只求解 2/6。大规模子集只有 4 个实例，
不宜据此对组件排序。

## 6. 按 cone mix 分层

| Cone mix | Total | Full | No scaling | Fixed \(\eta\) | Fixed \(\omega\) | No restart | No reflection |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXP without SOC | 6 | 5 | 5 | 5 | 5 | 5 | 5 |
| SOC + EXP | 17 | **14** | 10 | **14** | 8 | 8 | **14** |
| SOC without EXP | 40 | **37** | 29 | 36 | 34 | 32 | 36 |

最明显的差异出现在混合 SOC+EXP 实例：固定 \(\omega\) 或去掉 restart
都从 14/17 降到 8/17；去掉 scaling 降到 10/17。

## 7. 对“每个 option 是否有用”的回答

结果支持保留全部五个核心机制，但其作用强弱不同：

1. **Restart：强贡献。** 去掉后少解 11 个，SGM(10) 从 24.22 秒升至
   63.35 秒，共同求解实例的迭代数约为 Full 的 2.85 倍。
2. **Diagonal rescaling：强贡献。** 去掉后少解 12 个，是求解数下降最多的
   消融；共同求解实例的迭代数约为 Full 的 2.70 倍。
3. **Adaptive \(\omega\)：强贡献。** 固定 \(\omega\) 后少解 9 个，
   共同求解实例的迭代数约为 Full 的 2.76 倍。
4. **Reflection：较温和但一致的正贡献。** 去掉后少解 1 个，SGM(10)
   增加约 13%，共同求解实例迭代数增加约 11%。
5. **Adaptive \(\eta\)：鲁棒性/迭代数贡献。** 它多解 1 个，并使共同求解
   实例的迭代数降低约 19.6%；但 Full 的 wall time 高于 Fixed \(\eta\)，
   表明本实现中的自适应更新有运行开销。因此不能声称该选项在此数据上必然
   加速 wall time。

综合看，Full 获得最高求解数；scaling、restart 和 adaptive \(\omega\)
是主要鲁棒性来源，reflection 和 adaptive \(\eta\) 提供较小但可测的补充。

## 8. 完整性检查

正式结果满足：

```text
missing-4 records             252/252
composed records              378/378
records per configuration      63/63
run_status=COMPLETED          378/378
solver/infrastructure errors        0
broken result symlinks              0
empty selected raw logs             0
Halpern-enabled selected records    0
```

`no_restart` 的 restart count 为 0；六个配置的 Halpern restart candidate
选择次数均为 0，与开关设计一致。

## 9. 结果和图

主要表格：

```text
benchmark/results/rebuttal/ablation/
  one_at_a_time_core_600s_20260731/
    source_map.csv
    raw_results.csv
    summary_overall.csv
    summary_by_size.csv
    summary_by_cone_mix.csv
    paired_effects.csv
    report.md
```

当前 one-at-a-time 图：

```text
rebuttal_plan/figures/one_at_a_time_ablation_solved.pdf
rebuttal_plan/figures/one_at_a_time_ablation_sgm.pdf
rebuttal_plan/figures/one_at_a_time_ablation_paired_effects.pdf
```

旧 inline-Halpern 结果仅作为历史记录保留在：

```text
rebuttal_plan/ablation_study_results_legacy_inline_halpern.md
benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2
```

旧 raw data 没有被删除，但不得用于当前论文图或当前结论。
