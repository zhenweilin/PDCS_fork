# 旧 `pdhg_clp` GPU 代码恢复与 EXP hard-case 诊断

## 结论摘要

已从 Gitee 完整恢复历史 GPU 分支，并确认旧代码能够在当前 H100 上编译和运行。
但是，“旧版可以解大多数 hard cases”不能直接解释为当前 GPU solver 的算法回归，
因为历史记录混合了以下几种不同口径：

1. 2025-01-04 的 `e61feea` 使用 `l2` residual 停止，而当前实验使用三项
   `l_inf` residual/gap 验证；旧 solver 的 `OPTIMAL` 不一定通过当前标准。
2. `e61feea` 仍包含一个后来由 `4f15301` 明确修复的 diagonal exponential-cone
   projection 公式错误，因此不能作为正确的正式版本。
3. paper-release 的 `1e-6` GPU 脚本给每个 case 的 time limit 是 5 小时，当前
   2,100-case 对比给的是 1 小时；求解数量不能跨预算直接比较。
4. 即使固定 CPU/CUDA 随机种子，旧 GPU 代码的 sparse/reduction/atomic 浮点归约
   也不是 bitwise deterministic。微小数值差异会被 adaptive step 和 restart
   的离散分支放大，所以单条旧日志不能代表稳定表现。
5. 用户提供的 `oldlog.log` 进入的是 non-aggressive resolving 路径；已恢复的所有
   历史 GPU 提交只实现 aggressive GPU 函数。因此该日志不是可由已提交 GPU
   源码产生的正式 GPU 证据，本报告的性能结论不使用它。
6. 在相同 3600 秒、`1e-6` 严格口径下，正确 paper-era GPU 版本实际解出 3/6，
   当前 No-Halpern 与 Inline-Halpern 分别解出 1/6、二者并集为 2/6。旧版确实
   额外解出 `batchs101006m`，所以不能把全部差异都归为旧统计口径；但也没有
   证据支持“旧版普遍优于当前版”。

## 恢复的源码谱系

| 版本 | 提交 | 作用与限制 |
|---|---|---|
| Gitee `gpu_branch` 末端 | `80cb4a85d87e1bee19f2bbb19468fa2682f15da7` | 从 2025-01-04 的另一条早期分叉延伸；GPU 源码基本对应 `dee3515`，仍用 `l2` 停止且含后来确认的 EXP projection bug |
| 针对特例的早期版本 | `e61feeacc36a1c52db9294c27a66c176ac537d2a` | commit message 为 `0104 optimal for batchs101006m`；仍用 `l2` 停止且含后来确认的 EXP projection bug |
| 最终旧 GPU 版本 | `3432afc69f96891168c13c926d08cdaacd5b0e41` | 包含 EXP 修复和 `l_inf` 停止；除 Makefile 外，GPU solver 源码与 `origin/paper_release` 相同 |
| paper release | `e095ae50c879d9e7915fa8c4834bb04ef8b34bb9` | 新增正式测试脚本；`PDCP_GPU_cbf_folder_1e-6.jl` 使用 18,000 秒限制 |

Gitee 凭据仅由 `gitee_askpass.sh` 按 Git 的 Username/Password 请求读取，没有写入
remote URL 或日志。`external/gitee.txt` 已设置为权限 `600`。

## 恢复时发现的上游编译错误

`3432afc` 的提交说明是 `fix suff total_threads cal bug`，它把变量从
`total_thread` 改成 `total_threads`，但漏改了两处循环，导致原提交不能编译：

```text
sufficient_block_proj.cu: identifier "total_thread" is undefined
```

隔离 worktree 中只做了这两处机械改名。补丁保存在：

```text
external/old_can_solve_scripts/patches/pdhg_clp_3432afc_compile_total_threads.patch
```

修复后使用 CUDA 12.6 的 `nvcc` 和 `sm_90` 成功生成三个 PTX、`utils.ptx` 和
`libfew_block_proj.so`。主仓库当前源码没有被这一补丁修改。

## 为什么 `e61feea` 的“OPTIMAL”不能直接使用

### 停止标准不同

`e61feea:code/src/rpdhg_clp_gpu/termination.jl` 使用：

```julia
max(info.l_2_rel_primal_res, info.l_2_rel_dual_res, info.rel_gap) < rel_tol
```

提交 `7f8eb25` 才将其改为：

```julia
max(info.l_inf_rel_primal_res, info.l_inf_rel_dual_res, info.rel_gap) < rel_tol
```

在 1200 秒、`1e-6` 的 6 个 GPU case 上，早期版本自报 3 个 `OPTIMAL`，但按当前
三项 `l_inf` 标准复核仅 1 个通过：

| Case | 旧状态 | 时间 (s) | 最大 `l_inf` 指标 | 当前标准通过 |
|---|---:|---:|---:|---:|
| `batch` | `OPTIMAL` | 642.70 | `1.2507e-6` | 否 |
| `batchs101006m` | `OPTIMAL` | 1116.88 | `6.8723e-6` | 否 |
| `batchs121208m` | `TIME_LIMIT` | 1200.13 | `3.6641e-2` | 否 |
| `batchs151208m` | `TIME_LIMIT` | 1200.32 | `3.5640e-3` | 否 |
| `batchs201210m` | `TIME_LIMIT` | 1200.26 | `1.8223e2` | 否 |
| `enpro56` | `OPTIMAL` | 372.94 | `9.9531e-7` | 是 |

完整 raw logs、JSON 和 CSV 位于：

```text
external/old_can_solve_scripts/results/
  pdhg_clp_e61feea_gpu_hardcases_1200s_20260810T181001Z/
```

### EXP projection 公式后来被确认有误

提交 `4f15301` 将 diagonal EXP projection 中：

```cpp
dr_inv[0] / ds_squared[0]
```

修正为：

```cpp
1.0 / (dr_inv[0] * ds_squared[0])
```

`e61feea` 早于该修复。因此它对 `batchs101006m` 的特殊表现不能作为正确性证据，
也不应将该版本恢复到当前 solver。

Gitee 的 `origin/gpu_branch` 也不是 `3432afc` 的后继。它与 `e61feea` 从
`dc2061a` 分叉，末端 `80cb4a8` 的 GPU 源码仍保留同一错误表达式和 `l2`
停止标准。因此该分支只用于追溯旧日志，不是可回滚的正确版本。按用户建议实际
切换到该分支、重新编译并测试两个关键 case 后，1200 秒内严格通过 0/2：

| Case | 旧状态 | 时间 (s) | 最大 `l_inf` 指标 | 当前标准通过 |
|---|---:|---:|---:|---:|
| `batch` | `TIME_LIMIT` | 1200.47 | `5.0769e-2` | 否 |
| `batchs101006m` | `TIME_LIMIT` | 1200.20 | `5.3645e1` | 否 |

完整记录位于：

```text
external/old_can_solve_scripts/results/
  pdhg_clp_gpu_branch_80cb4a8_gpu_keycases_1200s_20260810T185857Z/
```

## 模型 loader 对轨迹的影响

历史正式脚本使用 `MOI.Utilities.Model{Float64}()` + `MOI.read_from_file`，随后
`MOI.copy_to`。直接复制 `CBF.Model` 得到相同维数和谱估计，但桥接/锥排序的细节
会改变数值轨迹。对 `e61feea/batch`：

| Loader | 旧 solver 时间 (s) | 自报状态 | `l_inf` dual | 当前标准 |
|---|---:|---:|---:|---:|
| direct CBF | 642.70 | `OPTIMAL` | `1.2507e-6` | 未通过 |
| historical MOI | 464.09 | `OPTIMAL` | `1.1347e-6` | 未通过 |

两者结论一致：loader 会改变路径，但旧 `l2` 停止标准才是自报 `OPTIMAL` 与当前
验证不一致的直接原因。正式 paper-era 复跑固定使用 `--model-loader moi`。

## 同一旧 GPU 版本的重复轨迹并非 bitwise deterministic

`3432afc` 同时固定了 CPU `Random.seed!(1234)` 和 `CUDA.seed!(1)`，但相同源码、
数据、MOI loader、参数及 H100 型号的两次独立运行，在约 1200 秒处已经走到不同
的 restart 分支：

| Case / run | Iterations | Restarts | `l_inf` dual | Relative gap |
|---|---:|---:|---:|---:|
| `batch`, 1200 秒预实验 | 3,820,000 | 42 | `3.625e-7` | `1.720e-5` |
| `batch`, 3600 秒实验在 1199.81 秒处 | 3,730,000 | 53 | `4.596e-6` | `4.226e-5` |
| `batchs101006m`, 1200 秒预实验 | 3,582,000 | 51 | `4.203e-4` | `1.744e-3` |
| `batchs101006m`, 3600 秒实验在 1199.80 秒处 | 3,444,000 | 43 | `3.329e-4` | `4.731e-3` |

这是 GPU sparse/reduction/atomic 浮点顺序与 adaptive/restart 离散决策共同造成的
运行轨迹敏感性证据；它不等价于已经证明存在数据竞争或 projection 正确性错误。
因此应比较固定预算下的重复统计，而不能用单条最优旧日志认定算法回归。

## 当前 No-Halpern 与 paper-era 的一个实质算法差异

源码对比发现，即使当前设置 `use_halpern=false`，人工/强制 restart 的行为也已
不再等同于 paper-era：

- `3432afc` 在 artificial/`force_restart` 条件触发后只增加 restart 计数并
  `break`，所以下一 epoch 总是从当时的 running mean 开始；
- 当前代码会调用 `evaluate_restart_kkt_candidates!` 和
  `install_restart_candidate!`，在 current 与 running mean 中按 KKT merit 选点。

在当前 No-Halpern 的 `batchs101006m` 一小时 raw log 中，19 次
force/artificial restart 有 13 次选择 current、6 次选择 mean；而本次正确旧版
严格求解该 case 的运行中共有 14 次此类 restart，且按旧代码全部保留 mean。
因此，`use_halpern=false` 并不意味着当前轨迹已经恢复为 paper-era 轨迹。
这是一个可复现实质差异，也是旧版在该 case 上更好的合理候选原因；要确认因果，
应另做“仅将 force/artificial restart 固定为 mean”的单变量消融，不能仅凭一次
运行宣称该改动必然导致回归。

## paper-era `3432afc` 正式对比

### 1200 秒预实验

使用历史正式脚本的 MOI loader、GPU aggressive resolving、`1e-6` tolerance，
paper-era 正确版本在 1200 秒内严格解出 1/6。所有 case 均正常完成，没有编译、
CUDA 或 Julia 运行错误：

| Case | 旧状态 | 时间 (s) | 最大 `l_inf` 指标 | 当前标准通过 |
|---|---:|---:|---:|---:|
| `batch` | `TIME_LIMIT` | 1200.32 | `1.7204e-5` | 否 |
| `batchs101006m` | `TIME_LIMIT` | 1200.33 | `1.7435e-3` | 否 |
| `batchs121208m` | `TIME_LIMIT` | 1200.35 | `1.2108e-1` | 否 |
| `batchs151208m` | `TIME_LIMIT` | 1200.65 | `1.2910e-1` | 否 |
| `batchs201210m` | `TIME_LIMIT` | 1200.05 | `4.5063e-1` | 否 |
| `enpro56` | `OPTIMAL` | 1039.16 | `6.1950e-7` | 是 |

完整 raw logs、JSON、CSV 和 Markdown 汇总位于：

```text
external/old_can_solve_scripts/results/
  pdhg_clp_3432afc_gpu_hardcases_1200s_20260810T182258Z/
```

### 3600 秒同预算实验

六例在六张独立 H100 上全部正常结束，无编译、CUDA 或 Julia 运行错误。使用与
当前实验相同的 3600 秒上限和三项 `l_inf`/gap 严格复核后，正确旧版解出 3/6：

| Case | 旧状态 | 时间 (s) | 最大 `l_inf` 指标 | 当前标准通过 |
|---|---:|---:|---:|---:|
| `batch` | `OPTIMAL` | 2399.12 | `9.4878e-7` | 是 |
| `batchs101006m` | `OPTIMAL` | 2113.61 | `5.7853e-7` | 是 |
| `batchs121208m` | `TIME_LIMIT` | 3600.36 | `3.2723e-4` | 否 |
| `batchs151208m` | `TIME_LIMIT` | 3600.50 | `2.0070e-2` | 否 |
| `batchs201210m` | `TIME_LIMIT` | 3600.24 | `7.6869e-3` | 否 |
| `enpro56` | `OPTIMAL` | 1564.45 | `9.1166e-7` | 是 |

完整 raw logs、JSON、CSV、Markdown 汇总、输入/kernel SHA-256 及逐例比较位于：

```text
external/old_can_solve_scripts/results/
  pdhg_clp_3432afc_gpu_hardcases_3600s_20260810T183244Z/
```

### 与当前两个一小时配置的逐例比较

表中数值为三项严格指标的最大值；`S` 表示在 `1e-6` 下通过，`F` 表示未通过：

| Case | 当前 No-Halpern | 当前 Inline-Halpern | `3432afc` |
|---|---:|---:|---:|
| `batch` | F, `8.887e-6` | S, `5.748e-7`, 1801.34 s | S, `9.488e-7`, 2399.12 s |
| `batchs101006m` | F, `8.551e-5` | F, `1.380e-4` | S, `5.785e-7`, 2113.61 s |
| `batchs121208m` | F, `8.572e-5` | F, `7.966e-5` | F, `3.272e-4` |
| `batchs151208m` | F, `1.074e-3` | F, `2.761e-2` | F, `2.007e-2` |
| `batchs201210m` | F, `1.069e0` | F, `1.403e-3` | F, `7.687e-3` |
| `enpro56` | S, `9.713e-7`, 1191.05 s | F, `3.098e-6` | S, `9.117e-7`, 1564.45 s |

因此，旧版的实际优势集中在 `batchs101006m`，并在 No-Halpern 对比下也包括
`batch`；但当前 Inline-Halpern 解 `batch` 更快，当前 No-Halpern 解 `enpro56`
更快，而且当前方法在三个共同未解 case 中各有更小的终止指标。结论应表述为
“存在 case-dependent 的轨迹回归/改进”，而不是全局新旧优劣。

历史 paper-release 的原始预算是每例 18,000 秒。本次同预算实验刻意使用 3600
秒来与当前结果公平比较；旧论文脚本可能在额外四小时内解出更多 case，但不能把
该五小时 solved count 与当前一小时 solved count 直接并列。

## 可复现入口

完整恢复、编译和运行命令见 `README.md`。核心脚本为：

```text
recover_pdhg_clp_gpu.sh  # 私有历史 fetch、隔离 worktree、补丁和 CUDA 构建
run_all.sh               # 六 case 多 GPU 调度
run_old_case.jl          # 单 case、完整 verbose=2 raw log 和严格复核
summarize.py             # 只读 raw records 后生成 CSV/Markdown
```
