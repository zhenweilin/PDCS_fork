# cuPDCS 旧 inline-Halpern 消融实验设计（历史归档）

更新日期：2026-07-28
实验数据目录：`/home/zhenwei/PDCS_fork/benchmark/represent_data`
实验对象：GPU 版本 cuPDCS
本文档状态：实验设计与执行记录。

执行设置（2026-07-29）：正式实验的每个 instance/configuration time
limit 为 **600 秒（10 分钟）**，termination tolerance 为 **`1e-6`**。
失败惩罚时间同样使用 600 秒。正式 run 为
`benchmark/results/rebuttal/ablation/ablation_600s_20260728_r2`；较早的
`ablation_600s_20260728_r1` 是用于发现终止候选导出问题的保留审计记录，
不得与正式结果混合。

## 1. 实验目的

审稿人希望说明 Algorithm 1 中各组成部分对 cuPDCS 性能和稳定性的贡献。本实验采用 one-factor-at-a-time 消融：以完整 cuPDCS 为基线，每次只关闭一个组成部分，其余算法、输入、停止条件和硬件环境保持不变。

主要消融以下五项：

1. diagonal rescaling；
2. adaptive step；
3. restart；
4. reflection；
5. Halpern iteration。

需要回答的问题是：

- 每一项技术是否提高求解成功率？
- 每一项技术对运行时间、迭代次数和矩阵向量乘次数有什么影响？
- 每一项技术是否改善最终 primal infeasibility、dual infeasibility 和 relative duality gap？
- 不同规模、不同锥类型的问题是否表现出不同的消融效果？

本实验不是简单比较当前代码中几个名称相似的 option。五项消融必须对应 Algorithm 1 中可独立辨认的数学操作，并且需要验证关闭某一项后没有意外关闭其他项。

## 2. 数据集范围

### 2.1 固定数据目录

正式实验只使用：

```text
/home/zhenwei/PDCS_fork/benchmark/represent_data
```

不得移动、覆盖或修改原始 `.cbf.gz` 文件。运行前为每个压缩文件及其解压后的内容记录 SHA-256，用于确认不同机器使用的是完全相同的数据。

### 2.2 当前数据清点

截至 2026-07-28，该目录中共有 63 个 CBF v3 实例。静态读取 CBF header 得到：

| 分类方式 | 实例数 |
|---|---:|
| 含 SOC、不含 EXP | 40 |
| 同时含 SOC 和 EXP | 17 |
| 含 EXP、不含 SOC | 6 |
| 总计 | 63 |

63 个实例都只包含当前求解器支持的锥类型；未发现 PSD、rotated SOC 或未知锥类型。所有实例还可能包含 free、linear equality 或 linear inequality 块。

按照论文中以 `ACOORD` 非零元个数划分规模的方法：

- small：`nnz(A) < 50,000`，53 个；
- medium：`50,000 <= nnz(A) <= 500,000`，6 个；
- large：`nnz(A) > 500,000`，4 个。

4 个 large 实例为：

| 实例 | `nnz(A)` |
|---|---:|
| `joint_FC_12.cbf.gz` | 1,834,217 |
| `qssp180.cbf.gz` | 1,681,746 |
| `nql180.cbf.gz` | 894,453 |
| `db-plane-strain-prism.cbf.gz` | 631,305 |

6 个 medium 实例为：

| 实例 | `nnz(A)` |
|---|---:|
| `100_0_1_w.cbf.gz` | 265,232 |
| `integrated_KleinesBeispiel_02_6.cbf.gz` | 231,949 |
| `strictmin_2D_43_dual.cbf.gz` | 196,364 |
| `nb.cbf.gz` | 193,898 |
| `LogExpCR.cbf.gz` | 132,827 |
| `cx02-100.cbf.gz` | 66,234 |

正式脚本必须重新生成 machine-readable manifest，不能只依赖上面的人工清点结果。

### 2.3 数据 manifest

计划生成：

```text
benchmark/results/rebuttal/ablation/<run_id>/manifest.csv
```

每个实例至少记录：

- `instance_id` 和文件名；
- 压缩文件 SHA-256；
- 解压内容 SHA-256；
- CBF version；
- 变量数和约束数；
- `ACOORD` 非零元数；
- 各锥类型的块数和总维数；
- `cone_mix`：`soc_only`、`soc_exp` 或 `exp_only`；
- `size_class`：`small`、`medium` 或 `large`；
- parser/conversion 是否成功及失败原因。

注意：当前仓库中尚未发现可直接对这 63 个 CBF 文件运行 cuPDCS 的完整 importer/runner。因此，正式实验前必须实现 CBF-to-MOI 转换，或找回论文主实验所使用的完全相同转换程序。parser/conversion 失败属于实验基础设施失败，不能计为某一个算法配置的求解失败。

## 3. 比较原则与实验变量

### 3.1 完整基线

`Full` 表示 Algorithm 1 的完整 GPU 实现：

- diagonal rescaling：on；
- adaptive step：on；
- restart：on；
- reflection：on；
- Halpern：on。

所有新增开关的默认值都应为 `true`，从而保持已发布用户 API 的默认调用方式不变。历史参数 `use_accelerated` 必须继续被接受但保持无操作，不得将其改作本消融实验的开关。

### 3.2 主实验配置矩阵

| 配置名 | Scaling | Adaptive step | Restart | Reflection | Halpern |
|---|:---:|:---:|:---:|:---:|:---:|
| `full` | on | on | on | on | on |
| `no_scaling` | off | on | on | on | on |
| `no_adaptive_step` | on | off | on | on | on |
| `no_restart` | on | on | off | on | on |
| `no_reflection` | on | on | on | off | on |
| `no_halpern` | on | on | on | on | off |

主实验共：

```text
63 instances × 6 configurations = 378 formal solves
```

这些配置构成论文中的主要消融结论。不要用“同时关闭多项”的结果替代单项消融。

### 3.3 可选的 plain PDHG sanity baseline

可以额外运行 `plain_pdhg`，但应单独报告，不纳入五项 one-at-a-time 效应：

- 五项全部关闭；
- 不进行 primal-weight 自适应；
- primal weight 固定为 1；
- 固定步长 `eta = 0.9 / ||G||_2`；
- 不 restart、不 reflection、不 Halpern averaging。

该设置与论文 Appendix E 的 plain PDHG 定义对齐。若运行，共增加 63 个 solve，总数变为 441。

## 4. 五项消融的严格定义

### 4.1 Diagonal rescaling

完整版本对原问题做 diagonal rescaling。消融版本应令 primal 和 dual scaling vectors 全部为 1，并在相同的 GPU 主循环与相同的 cone projection 路径中求解未缩放问题。

要求：

- 不允许因为 `use_scaling=false` 而切换到另一套旧算法实现；
- 不允许改变 cone projection strategy；
- 固定步长或 adaptive step 中使用的 operator 必须与实际求解的数据一致；
- 输出解必须反变换到原始问题坐标，再计算统一的残差；
- 记录 scaling 所花时间，并在关闭 scaling 时记录为 0。

建议接口：

```julia
use_scaling::Bool = true       # MOI wrapper
use_preconditioner::Bool = true # direct solver API，保持现有兼容性
```

两个入口必须解析为同一个内部 `use_diagonal_rescaling` 标志。

### 4.2 Adaptive step

这里的 adaptive step 指 Algorithm 1 中根据局部进展/线搜索更新 `eta` 的机制，而不是只调整 primal/dual step-size weight。

关闭 adaptive step 时：

```text
eta = 0.9 / ||G_hat||_2
```

其中 `G_hat` 是当前配置实际使用的算子：打开 scaling 时为缩放后的算子，关闭 scaling 时为原算子。整个 solve 中 `eta` 保持不变，line-search trial/rejection 次数必须为 0。

要求新增独立选项：

```julia
use_adaptive_step::Bool = true
```

现有 `use_adaptive_step_size_weight` 只描述 primal-weight 相关机制，不能当作 adaptive step 消融的替代品。主实验中除 `plain_pdhg` 外，primal-weight 策略保持为 Full 的默认设置。

### 4.3 Restart

关闭 restart 时：

- 不触发 KKT、duality-gap、artificial、force 或其他 restart；
- `restart_count` 必须严格等于 0；
- 当前 restart anchor 在 solve 过程中保持不变；
- 不重置 inner iteration counter；
- 终止时可以用统一的 candidate selection 选最终输出，但不得由此重启迭代。

建议新增统一主开关：

```julia
use_restart::Bool = true
```

现有 `use_kkt_restart` 和 `use_duality_gap_restart` 只作为 `use_restart=true` 时的从属 criterion switches。也可以修复现有 `use_adaptive_restart` 的语义，使它真正成为 restart master switch，但必须保持向后兼容并写入日志。

需要特别说明：当前 primal-weight 更新发生在 outer restart 边界。因此 `no_restart` 会同时移除“由 restart 触发的 weight 更新”，这是 restart 机制的依赖效应，应在论文中明确说明，而不能把它误称为完全独立的第二项消融。

### 4.4 Reflection

定义一次基础 PDHG 更新得到 `zhat_(t,k+1)`，令上一迭代为 `z_(t,k)`。打开 reflection 时：

```text
z_reflected = (1 + beta) * zhat_(t,k+1) - beta * z_(t,k)
```

关闭 reflection 时：

```text
beta = 0
z_reflected = zhat_(t,k+1)
```

要求新增：

```julia
use_reflection::Bool = true
```

关闭 reflection 不能同时关闭 Halpern anchor mixing。

### 4.5 Halpern iteration

令本次 restart 的锚点为 `z_(t,0)`，并令：

```text
alpha_k = (k + 1) / (k + 2)
```

打开 Halpern 时：

```text
z_(t,k+1) = alpha_k * z_reflected + (1 - alpha_k) * z_(t,0)
```

关闭 Halpern 时：

```text
z_(t,k+1) = z_reflected
```

要求新增：

```julia
use_halpern::Bool = true
```

关闭 Halpern 不能自动关闭 reflection，也不能改变 restart criterion。

## 5. 正式实验前的代码审计和阻塞项

以下问题必须先修复，否则当前代码无法产生可信的五因素消融结果。

### 5.1 不要用旧分支拼接消融配置

`src/pdcs_gpu/rpdhg_alg_gpu_gen.jl` 当前根据多个 option 选择不同的主循环，但若干 GPU 函数只被引用、没有实现，例如：

- `pdhg_main_iter_average_diagonal_rescaling_no_restarts!`；
- `pdhg_main_iter_average_diagonal_rescaling_adaptive_restarts!`；
- `pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight!`；
- `pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving!`。

当前完整路径实际落到：

```text
pdhg_main_iter_average_diagonal_rescaling_restarts_adaptive_weight_resolving_aggressive!
```

因此不要为六种配置维护六套易漂移的主循环。建议改为一个经过测试的 GPU production loop，在更新点使用五个正交的布尔标志。这样每个消融只改变对应公式。

### 5.2 `use_adaptive_step_size_weight` 不等于 adaptive step

完整主循环始终调用 adaptive step-size iteration。现有同名 option 主要影响 step-size weight，并不能关闭 Algorithm 1 的 adaptive `eta`。必须增加 `use_adaptive_step` 并测试固定 `eta` 路径。

### 5.3 现有 restart options 不能完全关闭 restart

仅将 `use_kkt_restart=false` 和 `use_duality_gap_restart=false` 不足以保证无 restart，因为 artificial/force 等条件仍可能执行。需要 master switch，并保证所有 restart 入口都服从它。

### 5.4 Reflection 与 Halpern 当前耦合

`src/pdcs_gpu/cuda/utils.cu` 中 `reflection_update` 同时承担 reflection、Halpern mixing 和 average update。为了做正交消融，至少应在同一 kernel 内显式传入 `use_reflection`、`use_halpern` 和 restart anchor，或拆成数学含义清楚的更新步骤。

推荐统一公式：

```text
reflected =
    use_reflection
    ? (1 + beta) * zhat - beta * z_previous
    : zhat

z_new =
    use_halpern
    ? alpha_k * reflected + (1 - alpha_k) * z_restart_anchor
    : reflected
```

### 5.5 当前 Halpern anchor 实现与 Algorithm 1 不一致

Algorithm 1 要求最后一项使用 restart anchor `z_(t,0)`。当前 `reflection_update` 使用当前 `primal_sol`/`dual_sol` 作为最后一项；代数展开后 anchor 项会被抵消，因此不能视为忠实的 Halpern 实现。

正式实验前必须：

1. 保存并向 GPU kernel 传入 primal 和 dual restart anchors；
2. 按第 4.5 节公式更新；
3. restart 时只在确定执行 restart 后更新 anchor；
4. 用 CPU 标量 reference 对 GPU kernel 做逐元素比较。

修复会改变 Full 配置的迭代轨迹。因此回归测试应以独立 KKT correctness 和 Algorithm 1 公式一致性为准，而不是强求与错误旧版本 bitwise 一致。

### 5.6 不使用未测试的 `method=:halpern` 旧路径

当前该路径带有 “not test yet” 提示，并且分支中没有可靠地设置完整 `main_loop!`。本实验应使用上面建议的统一 production loop，不应把这个旧 option 当作 Halpern 开关。

### 5.7 严格解析并记录 options

MOI wrapper 需要：

- 只接受已登记的 option；
- 对未知 option 报错，避免拼写错误被静默忽略；
- 将六个内部 resolved flags 完整写入 `config.json` 和 raw log；
- 在启动 solve 前打印 configuration fingerprint；
- 保持 `use_accelerated` 可传入但明确记录为 ignored compatibility option。

## 6. 实现后的正确性门槛

任何一项未通过时都不得开始 378 个正式 solve。

### 6.1 更新公式单元测试

为 primal 和 dual 随机小向量构造 CPU reference，覆盖：

| `use_reflection` | `use_halpern` | 预期 |
|:---:|:---:|---|
| false | false | `z_new = zhat` |
| true | false | 只执行 reflection |
| false | true | `zhat` 与 restart anchor 做 Halpern mixing |
| true | true | 完整 Algorithm 1 更新 |

检查：

- GPU 与 CPU reference 的最大绝对/相对误差；
- anchor 不是当前 iterate 的别名；
- restart 后 anchor 正确更新；
- weighted average 的 `eta_cum` 和系数正确。

### 6.2 短轨迹一致性

选择至少一个小 SOC 实例和一个含 EXP 的小实例，保存第 1、10、100 次迭代：

- current primal/dual；
- lag primal/dual；
- restart anchor；
- running average；
- `eta`；
- primal weight；
- restart decision。

GPU 轨迹与 CPU reference 或逐步公式检查一致。数组只在该 correctness test 中保存，正式 benchmark 不保存逐迭代向量。

### 6.3 每个开关的断言

- `no_scaling`：所有 scaling vectors 为 1，原始 `A,b,c` 不被修改；
- `no_adaptive_step`：`eta` 恒为 `0.9/||G_hat||_2`，line-search count 为 0；
- `no_restart`：`restart_count=0`，anchor 在全程不变；
- `no_reflection`：effective `beta=0`，Halpern 仍可打开；
- `no_halpern`：anchor coefficient 为 0，reflection 仍可打开；
- `full`：五项全部打开，更新公式与 Algorithm 1 一致。

### 6.4 Solver regression

每个配置至少运行：

- 一个 LP/linear-cone demo；
- 一个 SOC demo；
- 一个 primal diagonal exponential-cone demo；
- represent_data 中一个小 SOC、一个小 EXP、一个 SOC+EXP 实例。

所有返回向量必须 finite，cone projection 检查必须通过，统一 KKT verifier 必须工作。

### 6.5 编译产物一致性

运行脚本必须：

- 根据目标 GPU 架构重新编译生产 kernel；
- 不使用 profile/instrumented kernel；
- 记录 `.cu`、`.ptx` 和 shared library 的 SHA-256；
- 在 raw log 中记录实际加载的 kernel 路径；
- 六种配置使用同一套编译产物。

## 7. 公平运行协议

### 7.1 共同设置

与论文主实验一致：

```text
termination tolerance = 1e-6
time limit            = 600 seconds (10 minutes) per solve
```

一个实例只有在统一的独立 verifier 中同时满足下列条件才记为 solved：

```text
primal infeasibility <= 1e-6
dual infeasibility   <= 1e-6
relative duality gap <= 1e-6
```

不能仅依赖各配置自己的 solver status。

其他共同设置：

- 相同初始 primal/dual iterate；
- 相同 cone ordering 和 projection strategy；
- 相同 termination-check frequency；
- 相同 maximum iteration；
- 相同 Float64 精度；
- 相同 parser、canonicalization 和 presolve；
- 不允许为某个消融配置单独调参。

如果论文主实验的数据已经 presolve/relax，则直接复用该输入；若尚未处理，则所有配置必须使用同一次确定性的转换结果，不能配置间分别 presolve。

### 7.2 GPU 和软件环境

六种配置应在同一台机器、同一张空闲 GPU 上串行运行。记录：

- physical GPU index；
- GPU UUID、型号、显存；
- driver version；
- Julia CUDA runtime；
- system CUDA toolkit/nvcc；
- Julia version；
- PDCS git commit 和工作树是否 dirty；
- `Project.toml`/`Manifest.toml` SHA-256；
- kernel 和 shared-library SHA-256。

启动 batch 前检查：

- GPU 没有其他计算进程；
- utilization 低于预设阈值，建议小于 10%；
- 可用显存满足最大实例；
- CPU governor、线程数和 BLAS 线程数固定。

运行期间如果发现其他进程进入该 GPU，当前 solve 标记为 `contaminated`，之后重跑，不能直接混入正式统计。

### 7.3 Warm-up 与计时边界

JIT、package loading 和首次 CUDA context 初始化不计入 solver time。正式计时前按 cone family 做小规模 warm-up，并执行 `CUDA.synchronize()`。

每个 solve 记录四段时间：

1. `parse_seconds`：CBF 读取和 canonicalization；
2. `preprocess_seconds`：scaling、operator-norm estimation 等 solver preprocessing；
3. `iteration_seconds`：正式迭代；
4. `solver_total_seconds = preprocess_seconds + iteration_seconds`。

论文主表以 `solver_total_seconds` 为主；另记录：

```text
end_to_end_seconds =
    parse_seconds + solver_total_seconds
```

计时区间前后都必须同步 GPU，避免只测到异步 kernel launch。

### 7.4 运行顺序

实例是统计独立单元。对每个实例，六种配置的顺序通过一个固定并提交到结果目录的 order seed 随机排列，以减少温度、系统负载和长时间漂移造成的偏差。

建议默认：

```text
order_seed = 20260728
```

不要在正式结果产生后改变 seed。生成的 `run_order.csv` 必须保存。

每个实例/配置进行一次 formal solve，与论文主 benchmark 对齐。若另外做短实例 timing repeats，应明确标记为 timing-only；同一实例的重复运行不是独立样本，必须先取该实例的中位数，不能把重复次数当作扩大样本量。

### 7.5 中断与续跑

每个 solve 写入独立目录，先写临时状态，完整 flush 后再原子生成 `DONE` 标记。续跑时：

- 有有效 `DONE` 和匹配 config/hash 的任务跳过；
- 无 `DONE` 的任务重跑；
- config、input hash 或 git commit 不一致时不得复用；
- 不删除旧结果，创建新 `run_id`；
- raw solver log 必须保留。

## 8. 每次 solve 必须记录的字段

### 8.1 身份与环境

- `run_id`；
- `instance_id`；
- `configuration`；
- configuration fingerprint；
- input hashes；
- run-order position；
- GPU UUID；
- git commit；
- start/end UTC timestamp。

### 8.2 状态与性能

- native solver status；
- independently verified status；
- failure class；
- `parse_seconds`；
- `preprocess_seconds`；
- `iteration_seconds`；
- `solver_total_seconds`；
- `end_to_end_seconds`；
- total iterations；
- primal/dual matrix-vector product counts；
- cone projection count；
- restart count；
- line-search trials 和 rejections；
- peak GPU memory。

### 8.3 数值质量

在原始问题坐标上记录：

- primal infeasibility；
- dual infeasibility；
- relative duality gap；
- primal objective；
- dual objective；
- maximum non-finite count；
- cone violation。

还应记录：

- initial/final/min/max `eta`；
- initial/final/min/max primal weight；
- effective reflection coefficient 的 summary；
- resolved 五个 flags。

正式 benchmark 只保存标量 trace（例如每次 termination check 一行）；不要保存每次迭代的完整 primal/dual GPU 向量，以避免占用大量磁盘和内存。

## 9. 失败分类

统一使用以下类别：

- `SOLVED`：独立 verifier 三项均不超过 `1e-6`；
- `TIME_LIMIT`：达到 600 秒（10 分钟）；
- `ITERATION_LIMIT`；
- `OUT_OF_MEMORY`；
- `NUMERICAL_ERROR`：NaN、Inf、非法步长或 kernel 数值失败；
- `INACCURATE`：solver 声称 solved，但统一 verifier 未通过；
- `RUNTIME_ERROR`：程序、CUDA 或 kernel 错误；
- `CONTAMINATED`：GPU 被其他进程干扰，需要重跑；
- `PARSER_ERROR` / `CONVERSION_ERROR`：基础设施错误，不纳入算法胜负。

失败实例不得悄悄从分母中删除。主表分母固定为 63；parser/conversion 问题必须先修复后再形成最终表。

## 10. 统计分析

### 10.1 主汇总

对六种配置分别报告：

- verified solved / 63；
- shifted geometric mean runtime；
- verified-solved 实例上的 geometric mean iterations；
- verified-solved 实例上的 geometric mean matrix-vector products；
- primal、dual、gap 的 median 和 maximum；
- peak GPU memory。

运行时间沿用论文的 shifted geometric mean：

```text
SGM(10; t_1,...,t_n) =
    exp(mean(log(t_i + 10))) - 10
```

timeout 或其他算法失败的运行时间赋值为 600 秒后计入 SGM。必须在结果脚本中把这个规则写成显式代码。

只在 verified-solved 上计算 iteration/matvec 几何均值时，要同时报告样本数，防止“只剩容易实例”造成误读。

### 10.2 配对比较

每个消融配置与同一实例上的 `full` 配对。对双方都 verified solved 的实例计算：

```text
runtime_ratio = ablated_time / full_time
iteration_ratio = ablated_iterations / full_iterations
matvec_ratio = ablated_matvecs / full_matvecs
```

报告 geometric mean paired ratio 和 95% confidence interval。置信区间使用以实例为单位的 10,000 次 bootstrap；不能以 iteration 或 timing repeat 为单位重采样。

除 jointly solved ratio 外，还要单独报告：

- Full solved、消融失败的实例数；
- 消融 solved、Full 失败的实例数；
- 两者都失败的实例数。

### 10.3 分层结果

总体 63 个实例由 small 问题主导，因此至少给出两个分层表：

1. 按规模：small / medium / large；
2. 按锥组合：SOC without EXP / SOC+EXP / EXP without SOC。

分层样本较小时只做描述性统计，不宣称强统计显著性。

## 11. 建议的论文表格与图

### 11.1 主表

| Configuration | Components enabled | Solved/63 | SGM(10) time | GM iterations | GM matvecs | Median/Max primal residual | Median/Max dual residual | Median/Max gap | Peak GPU memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | S+A+R+F+H | | | | | | | | |
| −Scaling | A+R+F+H | | | | | | | | |
| −Adaptive step | S+R+F+H | | | | | | | | |
| −Restart | S+A+F+H | | | | | | | | |
| −Reflection | S+A+R+H | | | | | | | | |
| −Halpern | S+A+R+F | | | | | | | | |

其中：

```text
S = scaling
A = adaptive step
R = restart
F = reflection
H = Halpern
```

### 11.2 配对效应表

| Ablation | Jointly solved | Runtime ratio vs Full | 95% CI | Iteration ratio | Matvec ratio | Full-only solved | Ablation-only solved |
|---|---:|---:|---:|---:|---:|---:|---:|
| −Scaling | | | | | | | |
| −Adaptive step | | | | | | | |
| −Restart | | | | | | | |
| −Reflection | | | | | | | |
| −Halpern | | | | | | | |

### 11.3 图

建议最多使用三类图：

1. verified fraction solved versus time；
2. verified fraction solved versus matrix-vector products；
3. 五项消融相对 Full 的 paired log-runtime ratio forest plot。

性能曲线中的失败实例保持未解决，不得只画成功子集。

## 12. 计划实现的脚本与目录

以下是本实验需要新增的接口规划；截至本文档更新时尚未实现，不能把这些命令当作当前已经可运行。

```text
benchmark/ablation/
├── inspect_represent_data.py
├── run_ablation_case.jl
├── run_ablation.sh
└── analyze_ablation.py

test/
├── test_ablation_update_gpu.jl
└── test_ablation_trace_gpu.jl
```

建议结果目录：

```text
benchmark/results/rebuttal/ablation/<run_id>/
├── environment.txt
├── config.json
├── manifest.csv
├── run_order.csv
├── kernel_hashes.txt
├── cases/
│   └── <instance>/<configuration>/
│       ├── config.json
│       ├── solver.raw.log
│       ├── scalar_trace.csv
│       ├── result.json
│       └── DONE
├── raw_results.csv
├── summary_overall.csv
├── summary_by_size.csv
├── summary_by_cone_mix.csv
├── paired_effects.csv
└── figures/
```

### 12.1 规划中的数据检查命令

```bash
cd /home/zhenwei/PDCS_fork

python3 benchmark/ablation/inspect_represent_data.py \
  --input-dir benchmark/represent_data \
  --output benchmark/results/rebuttal/ablation/<run_id>/manifest.csv
```

该脚本必须验证：

- 文件数恰为 63；
- hash 可重复；
- 每个 CBF 可完整解析；
- 每个锥类型受 cuPDCS 支持；
- manifest 中的 size/cone 分类与正式 runner 使用的 canonical model 一致。

### 12.2 规划中的 preflight

```bash
cd /home/zhenwei/PDCS_fork

PDCS_SKIP_GPU_PRECOMPILE=1 \
CUDA_VISIBLE_DEVICES=<idle_gpu> \
julia -O1 --project=. test/test_ablation_update_gpu.jl

PDCS_SKIP_GPU_PRECOMPILE=1 \
CUDA_VISIBLE_DEVICES=<idle_gpu> \
julia -O1 --project=. test/test_ablation_trace_gpu.jl
```

之后用 runner 的 `--smoke` 模式在 SOC、EXP 和 mixed 三类问题上分别测试六种配置。

### 12.3 规划中的正式运行命令

```bash
cd /home/zhenwei/PDCS_fork

RUN_ID=ablation_$(date -u +%Y%m%dT%H%M%SZ)

PDCS_SKIP_GPU_PRECOMPILE=1 \
CUDA_VISIBLE_DEVICES=<idle_gpu> \
bash benchmark/ablation/run_ablation.sh \
  --input-dir benchmark/represent_data \
  --output-dir benchmark/results/rebuttal/ablation/${RUN_ID} \
  --configs full,no_scaling,no_adaptive_step,no_restart,no_reflection,no_halpern \
  --tol 1e-6 \
  --time-limit 600 \
  --order-seed 20260728 \
  --resume
```

可选 plain PDHG：

```bash
PDCS_SKIP_GPU_PRECOMPILE=1 \
CUDA_VISIBLE_DEVICES=<idle_gpu> \
bash benchmark/ablation/run_ablation.sh \
  --input-dir benchmark/represent_data \
  --output-dir benchmark/results/rebuttal/ablation/${RUN_ID} \
  --configs plain_pdhg \
  --tol 1e-6 \
  --time-limit 600 \
  --order-seed 20260728 \
  --resume
```

### 12.4 规划中的分析命令

```bash
python3 benchmark/ablation/analyze_ablation.py \
  --run-dir benchmark/results/rebuttal/ablation/${RUN_ID} \
  --tolerance 1e-6 \
  --timeout-value 600 \
  --sgm-shift 10 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260728
```

分析脚本必须拒绝以下输入：

- 63 个实例不全；
- 同一个 case 出现重复且无法确定正式结果；
- config fingerprint 不一致；
- input hash 不一致；
- 缺少独立 residual；
- `CONTAMINATED` 尚未重跑；
- parser/conversion 尚有失败。

## 13. 完成标准

只有同时满足以下条件，实验才可用于 rebuttal 或论文：

- [ ] CBF importer/runner 已实现并通过 63 个实例的解析；
- [ ] 五个开关是正交的，且默认 API 行为保持兼容；
- [ ] `use_accelerated` 仍被接受并保持 ignored；
- [ ] Halpern 使用真正的 restart anchor；
- [ ] 四组 reflection/Halpern 公式测试通过；
- [ ] 固定步长、无 restart、无 scaling 断言全部通过；
- [ ] SOC、EXP、mixed smoke tests 全部通过；
- [ ] 六种配置使用相同 production kernels；
- [ ] 全部 378 个正式任务有可审计状态；
- [ ] 所有声称 solved 的结果通过独立三项 KKT verifier；
- [ ] 所有 timeout 和算法失败保留在 63 个实例的固定分母中；
- [ ] raw logs、manifest、environment、hash 和 run order 均保留；
- [ ] 总体和分层表均由脚本从 raw results 自动生成。

## 14. 建议的 reviewer 回应逻辑

最终文字应先说明实验采用 63 个代表性 CBF 实例，在同一 GPU、`1e-6` 统一精度和每个配置 600 秒（10 分钟）上限下，以完整 cuPDCS 为基线逐项关闭 scaling、adaptive step、restart、reflection 和 Halpern。随后报告：

1. 每个配置通过独立 KKT 检查的 solved count；
2. 含失败惩罚的 SGM(10) 时间；
3. 配对的 runtime、iteration 和 matvec ratio；
4. small/medium/large 以及 SOC/EXP cone mix 的分层结果；
5. 对 restart 与 restart-triggered weight update 这一依赖关系作透明说明。

结论必须由正式结果决定。若某一机制只在特定规模或锥组合下有效，应按分层结果陈述，不能预先宣称五项机制都会在所有问题上带来加速。
