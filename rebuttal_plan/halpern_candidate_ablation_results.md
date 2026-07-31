# Halpern 作为 restart candidate 的额外消融实验结果

## 1. 实验状态

正式实验已经完成。

- 正式记录：126/126；
- 实例：63 个代表性 CBF 实例；
- 配置：`with_halpern_candidate` 和 `without_halpern_candidate`；
- GPU：仅使用物理 GPU 5；
- tolerance：`1e-6`；
- 单个实例/配置 time limit：600 秒；
- 运行状态：126 条均为 `COMPLETED`；
- 终止状态：112 条 `OPTIMAL`，14 条 `TIME_LIMIT`；
- runtime error：0；
- 完整且非空的 raw solver log：126/126；
- 每个配置均为 56/63 个 verified solved。

正式结果目录为：

```text
benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5_20260730
```

`environment.txt` 和全部 126 个 `result.json` 均确认
`CUDA_VISIBLE_DEVICES=5`。实验期间两个配置按实例内随机顺序串行运行，因此没有
其它 cuPDCS 实验与其竞争 GPU 5。

## 2. 本次实现与旧 Halpern 设计的区别

主序列始终使用 reflection：

\[
z_{k+1}
=
(1+\beta_k)\widehat z_{k+1}
-\beta_k z_k .
\]

Halpern 点只作为辅助 restart candidate：

\[
z^{\mathrm{hal}}_{k+1}
=
\alpha_k z_{k+1}
+(1-\alpha_k)z_{t,0}.
\]

当 restart 条件成立时，打开 Halpern candidate 的配置在以下三个候选点中选择
restart merit 最小者：

\[
z^{\mathrm{restart}}
\in
\left\{
z^{\mathrm{current}},
z^{\mathrm{mean}},
z^{\mathrm{hal}}
\right\}.
\]

关闭 Halpern candidate 时只比较 current 和 mean。无论 Halpern candidate
是否被选中，普通的下一次 PDHG 迭代均从 reflection 主序列继续；只有在该候选点
被 restart 选择后，它才会成为下一 epoch 的起点。

这与旧消融实验中的“每一步都将 Halpern mixing 写回主序列”不同。本报告不能与
旧实验混为同一种算法语义。

## 3. 总体结果

| 配置 | Verified solved | SGM(10) wall time | GM iterations | Timeout |
|---|---:|---:|---:|---:|
| 使用 Halpern candidate | 56/63 | 26.02 s | 15,384 | 7 |
| 不使用 Halpern candidate | 56/63 | 25.63 s | 15,368 | 7 |

SGM(10) 对未求解实例使用 600 秒惩罚。两种配置求解了完全相同的 56 个实例，
均有 7 个 timeout。

在共同求解成功的 56 个实例上，以“关闭 candidate / 打开 candidate”定义比值：

| 指标 | 几何平均比值 | 95% bootstrap CI |
|---|---:|---:|
| Wall time | 0.9522 | [0.8438, 1.0740] |
| Iterations | 0.9989 | — |

wall-time 置信区间包含 1，因此本实验没有发现 Halpern candidate 带来统计显著的
总体加速。点估计上，关闭 candidate 快约 4.8%；但迭代数仅相差约 0.1%，说明
两种设计的总体迭代工作量几乎相同。

## 4. Restart candidate 实际选择次数

| 配置 | Current | Mean | Halpern | 总选择次数 |
|---|---:|---:|---:|---:|
| 使用 Halpern candidate | 493 | 223 | 59 | 775 |
| 不使用 Halpern candidate | 561 | 138 | 0 | 699 |

打开 candidate 后，Halpern 点被选择 59 次，占该配置全部 restart candidate
选择的 7.6%。63 个实例中有 20 个至少选择过一次 Halpern candidate，单个实例
最多选择 10 次。这说明第三个候选点确实被算法使用，并非恒为无效分支。

关闭 candidate 后 `restart_halpern_count` 在全部 63 个实例上严格为 0，验证了
消融开关没有残留 Halpern 选择路径。

不同配置采用的 restart 点会改变后续轨迹，所以两种配置的 restart 总次数不要求
完全相同。

## 5. 按规模分组

| 规模 | 使用 candidate：solved / SGM | 不使用 candidate：solved / SGM |
|---|---:|---:|
| Small（53） | 49/53 / 18.59 s | 49/53 / 17.84 s |
| Medium（6） | 5/6 / 57.03 s | 5/6 / 61.29 s |
| Large（4） | 2/4 / 294.14 s | 2/4 / 321.86 s |

Medium 和 Large 组的点估计偏向使用 Halpern candidate，Small 组则偏向关闭
candidate。Medium 与 Large 的样本数分别只有 6 和 4，因此不能据此声称存在
稳定的规模相关加速。

## 6. 按 cone mix 分组

| Cone mix | 使用 candidate：solved / SGM | 不使用 candidate：solved / SGM |
|---|---:|---:|
| EXP without SOC（6） | 5/6 / 21.47 s | 5/6 / 19.98 s |
| SOC + EXP（17） | 14/17 / 50.75 s | 14/17 / 51.47 s |
| SOC without EXP（40） | 37/40 / 19.44 s | 37/40 / 19.00 s |

三个 cone mix 的 solved 数完全一致，运行时间点估计也没有呈现一致方向。

## 7. 与旧 inline-Halpern 消融结果的关系

旧设计把 Halpern mixing 写入每一次主迭代。旧实验中，关闭 Halpern 后：

- paired wall-time ratio 为 0.640；
- paired iteration ratio 为 0.449；
- verified solved 从 53/63 增加到 56/63。

这说明旧 inline-Halpern 设计会明显拖慢主序列。改为本次 candidate-only
设计后：

- 两种配置均求解 56/63；
- paired iteration ratio 变为 0.9989；
- paired wall-time ratio 的 95% CI 包含 1。

因此，candidate-only 设计消除了旧实现中显著的迭代数和成功率损失。它能够在
部分 restart 中提供更好的候选点，但当前 63 个实例尚未证明它能带来稳定的总体
时间收益。

## 8. 建议写入论文或 rebuttal 的结论

建议采用以下谨慎结论：

> We removed Halpern averaging from the main reflected PDHG trajectory and
> retained it only as an optional restart candidate. Across 63 representative
> instances, enabling this candidate and disabling it both solved 56 instances.
> The paired iteration ratio was 0.999, while the paired wall-time ratio was
> 0.952 with a 95% bootstrap confidence interval of [0.844, 1.074]. The Halpern
> point was nevertheless selected in 59 restarts across 20 instances. Thus, the
> candidate-only design avoids the substantial slowdown observed when Halpern
> mixing was applied to every main iteration, although this test does not show a
> statistically significant aggregate speedup from the auxiliary candidate.

当前证据支持保留 Halpern 点作为可选 restart candidate，但不支持把它描述为
已经获得稳定加速的核心组件。

## 9. 结果与复现文件

- 自动生成报告：
  `benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5_20260730/report.md`
- 每条正式记录及 raw-log 路径：
  `benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5_20260730/raw_results.csv`
- 总体汇总：
  `benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5_20260730/summary_overall.csv`
- 按规模汇总：
  `benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5_20260730/summary_by_size.csv`
- 按 cone mix 汇总：
  `benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5_20260730/summary_by_cone_mix.csv`
- paired 分析：
  `benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5_20260730/paired_effects.csv`
- 环境与源码 hash：
  `benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5_20260730/environment.txt`
- 复现步骤：
  `rebuttal_plan/how_to_run_halpern_candidate_ablation.md`
- 数学公式：
  `formula.md`
