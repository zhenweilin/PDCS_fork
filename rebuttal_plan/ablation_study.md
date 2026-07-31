# cuPDCS 当前核心组件 one-at-a-time 消融实验设计

更新日期：2026-07-31
数据目录：`benchmark/represent_data`
求解器：GPU 版本 cuPDCS
精度：`1e-6`
单实例/配置时间上限：600 秒（10 分钟）

## 1. 为什么需要重新实验

2026-07-28 的旧 one-at-a-time 批次使用了已经废弃的 inline-Halpern
主序列。旧批次中的 `full`、`no_scaling`、`no_adaptive_step`、
`no_restart` 和 `no_reflection` 均含 inline Halpern，因此这些配置不能用于
评价当前求解器的核心组件。

旧数据并非计算错误，但算法语义已经过时。其设计和结果已分别归档为：

```text
rebuttal_plan/ablation_study_legacy_inline_halpern.md
rebuttal_plan/ablation_study_results_legacy_inline_halpern.md
```

旧的 raw logs 继续保留在：

```text
benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2
```

不得把该旧批次与当前 progressive 或当前 one-at-a-time 结果混合。

## 2. 当前实验只研究五个核心机制

当前 one-at-a-time 实验完全不考虑 Halpern。六个配置均设置：

```text
use_halpern = false
```

Halpern 只作为 restart candidate 的效果由独立二配置实验回答，见：

```text
rebuttal_plan/halpern_candidate_ablation_results.md
```

本实验的五个核心机制为：

1. diagonal rescaling；
2. adaptive common step \(\eta\)；
3. adaptive primal-dual weight \(\omega\)；
4. restart；
5. reflection。

## 3. 六个严格单变量配置

| Configuration | Scaling | Adaptive \(\eta\) | Adaptive \(\omega\) | Restart | Reflection | Halpern |
|---|---:|---:|---:|---:|---:|---:|
| `full` | on | on | on | on | on | off |
| `no_scaling` | off | on | on | on | on | off |
| `no_adaptive_step` | on | off | on | on | on | off |
| `no_adaptive_primal_weight` | on | on | off | on | on | off |
| `no_restart` | on | on | on | off | on | off |
| `no_reflection` | on | on | on | on | off | off |

除表中关闭的一个目标机制外，其余五个内部开关保持与 `full` 相同。

关闭 adaptive \(\eta\) 时，公共步长由 power method 得到的算子范数固定：

\[
\eta =
\begin{cases}
0.9/\lVert G\rVert_2,&\text{power method converged},\\
0.8/\lVert G\rVert_2,&\text{otherwise}.
\end{cases}
\]

关闭 adaptive \(\omega\) 时仍允许 adaptive \(\eta\)，但不再动态改变
\(\tau=\eta/\omega\) 与 \(\sigma=\eta\omega\) 的比例。

当前实现中的 \(\omega\) 更新发生在 restart 边界。因此 `no_restart` 虽然保持
`use_adaptive_step_size_weight=true`，但不会产生由 restart 触发的实际
\(\omega\) 更新。这是算法结构上的依赖效应，不是 runner 额外关闭了第二个开关；
报告中应明确说明。

## 4. 复用已经完成的两组记录

progressive 六级正式批次已经包含两个完全匹配的配置：

| One-at-a-time name | Progressive source configuration |
|---|---|
| `full` | `pdhg_restart_scaling_reflection_adaptive` |
| `no_adaptive_step` | `pdhg_restart_scaling_reflection_adaptive_primal_weight` |

源目录为：

```text
benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730
```

这两组均有 63 条正式记录，Halpern 均关闭，因此不重复运行。

## 5. 只补跑四个缺失配置

需要新运行：

```text
no_scaling
no_adaptive_primal_weight
no_restart
no_reflection
```

正式补充批次共：

```text
63 instances × 4 configurations = 252 solves
```

运行命令：

```bash
cd /home/zhenwei/PDCS_fork

bash benchmark/progressive_ablation/run_progressive_ablation.sh \
  --experiment one_at_a_time_core_missing4 \
  --configs no_scaling,no_adaptive_primal_weight,no_restart,no_reflection \
  --run-id one_at_a_time_core_missing4_600s_all_idle_20260731 \
  --output-dir \
    benchmark/results/rebuttal/ablation/one_at_a_time_core_missing4_600s_all_idle_20260731 \
  --gpus auto \
  --julia ./.julia-bin/julia-1.12.6/bin/julia \
  --time-limit 600 \
  --tolerance 1e-6 \
  --order-seed 20260731 \
  --resume true
```

`--gpus auto` 只选择启动时显存占用小于 2048 MiB 且 utilization 不高于 10%
的 GPU，每张 GPU 只运行一个 worker。

## 6. 正式结果组合

四个缺失配置完成后，执行：

```bash
python3 benchmark/ablation/compose_core_one_at_a_time.py \
  --missing-run \
    benchmark/results/rebuttal/ablation/one_at_a_time_core_missing4_600s_all_idle_20260731 \
  --output-dir \
    benchmark/results/rebuttal/ablation/one_at_a_time_core_600s_20260731 \
  --tolerance 1e-6 \
  --time-limit 600 \
  --bootstrap-seed 20260731
```

组合脚本会：

1. 验证两个源批次使用相同的 63 个实例及 SHA-256；
2. 为六个配置建立只读符号链接视图，不复制或删除 raw logs；
3. 写出 `source_map.csv`，记录每个配置的真实来源；
4. 生成 378 条统一的 `raw_results.csv` 和分析报告。

最终目录应包含：

```text
benchmark/results/rebuttal/ablation/one_at_a_time_core_600s_20260731/
├── manifest.csv
├── source_map.csv
├── raw_results.csv
├── summary_overall.csv
├── summary_by_size.csv
├── summary_by_cone_mix.csv
├── paired_effects.csv
├── report.md
└── cases/
```

## 7. 指标

只有同时满足以下三项才计为 verified solved：

\[
r_{\mathrm p,\infty}^{\mathrm{rel}}\le 10^{-6},\qquad
r_{\mathrm d,\infty}^{\mathrm{rel}}\le 10^{-6},\qquad
\mathrm{relative\ gap}\le 10^{-6}.
\]

报告：

- verified solved / 63；
- 失败按 600 秒惩罚的 SGM(10) wall time；
- 成功实例上的 geometric-mean iterations；
- primal residual、dual residual 和 relative gap；
- 按规模和 cone mix 的分层结果。

由于 `full` 与四个新消融来自不同的受控正式批次，跨批次 paired wall-time
ratio 作为辅助结果解释；成功数、失败类型和最终数值精度不依赖计时配对。
严格的逐阶段 paired timing 由 progressive 同批次实验报告。

## 8. 完整性要求

正式结果必须满足：

```text
新补充记录：252/252
组合记录：378/378
每个配置：63/63
runtime error：0
所有选中记录均有非空 solver.raw.log
所有 resolved_flags.halpern 均为 false
```

任何 PARTIAL 报告不得用于论文或 rebuttal。
