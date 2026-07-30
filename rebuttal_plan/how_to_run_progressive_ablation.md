# 递进式消融实验复现说明

## 1. 实验顺序

正式实验按重要性顺序累计加入组件：

```text
PDHG
  -> + restart
  -> + diagonal rescaling
  -> + reflection
  -> + adaptive primal weight
  -> + adaptive step
```

全部六级配置均关闭 Halpern。第 5 级只打开 adaptive primal–dual weight
\(\omega\)，同时保持公共步长 \(\eta\) 固定；最后一级再打开 adaptive
line search 更新 \(\eta\)。

详细定义见：

```text
rebuttal_plan/progressive_ablation.md
```

## 2. 进入仓库并检查输入

```bash
cd /home/zhenwei/PDCS_fork

find benchmark/represent_data \
  -maxdepth 1 -type f -name '*.cbf.gz' | wc -l
```

必须得到：

```text
63
```

## 3. 检查 Julia 与 CUDA

```bash
./.julia-bin/julia-1.12.6/bin/julia --version
nvidia-smi
/usr/local/cuda/bin/nvcc --version
```

本实验不使用 Nsight 或 NVIDIA performance counters，不需要 profiling
权限。

## 4. 编译 production kernel

```bash
make -C src/pdcs_gpu/cuda utils.ptx \
  CUDA_HOME=/usr/local/cuda \
  ARCH=sm_90
```

正式实验使用 production `utils.ptx`，不能将任何 `*_profile.ptx` 替代它。

## 5. 检查六级开关

```bash
CUDA_VISIBLE_DEVICES=0 \
PDCS_SKIP_GPU_PRECOMPILE=1 \
JULIA_PKG_PRECOMPILE_AUTO=0 \
./.julia-bin/julia-1.12.6/bin/julia \
  --startup-file=no \
  --compiled-modules=existing \
  -O1 \
  --project=. \
  test/test_progressive_ablation_flags.jl
```

预期：

```text
PROGRESSIVE_ABLATION_FLAGS_PASS
```

## 6. 使用全部空闲 GPU 启动正式实验

```bash
cd /home/zhenwei/PDCS_fork

bash benchmark/progressive_ablation/run_progressive_ablation.sh \
  --run-id progressive_six_stage_600s_all_idle \
  --output-dir \
    benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle \
  --gpus auto \
  --julia ./.julia-bin/julia-1.12.6/bin/julia \
  --time-limit 600 \
  --tolerance 1e-6 \
  --order-seed 20260730 \
  --resume true
```

`--gpus auto` 会使用启动时所有满足下列条件的 GPU：

- `memory.used < 2048 MiB`；
- `utilization.gpu <= 10%`。

每张选中的 GPU 启动一个 worker。worker 从共享队列动态领取尚未运行的实例，
因此困难实例不会导致静态 GPU 分区严重失衡。

## 7. 指定 GPU

如果另一台机器只允许使用部分 GPU，例如 GPU 1、3、6：

```bash
bash benchmark/progressive_ablation/run_progressive_ablation.sh \
  --run-id progressive_six_stage_600s_gpu136 \
  --output-dir \
    benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_gpu136 \
  --gpus 1,3,6 \
  --julia ./.julia-bin/julia-1.12.6/bin/julia \
  --time-limit 600 \
  --tolerance 1e-6 \
  --order-seed 20260730 \
  --resume true
```

脚本仍会检查显式指定的 GPU 是否空闲；非空闲时拒绝启动。

## 8. 运行管理

### 8.1 运行中追加后来释放的 GPU

`--gpus auto` 只在主脚本启动时检测一次空闲卡。如果实验运行期间又有一张卡
释放，可以从**另一个终端**将一个 worker 加入原调度会话，不需要停止或重启
主脚本。例如向正在运行的批次加入 GPU 6：

```bash
cd /home/zhenwei/PDCS_fork

RUN_DIR=benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle

find "$RUN_DIR/scheduler_sessions" \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
```

上一步会打印主脚本当前的 session ID，例如
`20260730T181434Z_2881101`。将这个**原样值**传入：

```bash
bash benchmark/progressive_ablation/add_progressive_worker.sh \
  --run-dir "$RUN_DIR" \
  --session-id 20260730T181434Z_2881101 \
  --gpu 6
```

追加脚本会在启动前重新检查 GPU 满足 `memory.used < 2048 MiB` 且
`utilization.gpu <= 10%`，否则拒绝启动。它与主脚本共享同一个
`claim.lock` 和 `claims/` 目录，所以不会重复领取同一实例；同一 session
也不允许重复添加两个 GPU 6 worker。

保持主脚本终端和追加 worker 终端都在运行，直到二者均显示完成。初始 GPU
写在 `environment.txt` 的 `physical_gpus` 中；运行中追加的 GPU 及其实际
任务以 `gpu_assignments.csv` 为准。若主脚本先结束而追加 worker 尚未结束，
应等待追加 worker 完成，再按第 11 节命令重新生成最终分析。

### 8.2 中断后恢复整个批次

使用与首次运行完全相同的 `--output-dir` 和参数，重新执行命令：

```bash
bash benchmark/progressive_ablation/run_progressive_ablation.sh \
  --run-id progressive_six_stage_600s_all_idle \
  --output-dir \
    benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle \
  --gpus auto \
  --julia ./.julia-bin/julia-1.12.6/bin/julia \
  --time-limit 600 \
  --tolerance 1e-6 \
  --order-seed 20260730 \
  --resume true
```

具有 case-level `DONE` 的任务会跳过。失败或被中断的任务创建
`attempt_002`、`attempt_003` 等新目录；旧 attempt 和 raw log 不会覆盖。

不要删除 `cases/`、`scheduler_sessions/` 或旧 attempt。

## 9. 查看进度

```bash
RUN_DIR=benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle

find "$RUN_DIR/cases" \
  -mindepth 3 -maxdepth 3 -name DONE -type f | wc -l

find "$RUN_DIR/cases" \
  -path '*/attempt_*/result.json' -type f | wc -l

nvidia-smi
```

正式目标均为：

```text
378
```

实例到 GPU 的实际映射保存在：

```bash
column -s, -t "$RUN_DIR/gpu_assignments.csv" | less -S
```

## 10. 检查最终报告

```bash
grep -F \
  'Status: **COMPLETE** (378/378 formal records).' \
  "$RUN_DIR/progressive_report.md"
```

只有 `COMPLETE` 报告可以用于论文。

查看总体结果：

```bash
column -s, -t "$RUN_DIR/summary_overall.csv" | less -S
```

查看每一步增加组件后的 paired effect：

```bash
column -s, -t "$RUN_DIR/adjacent_effects.csv" | less -S
```

查看 raw-log 路径：

```bash
column -s, -t "$RUN_DIR/raw_results.csv" | less -S
```

每条 raw log 的实际位置为：

```text
cases/<instance>/<configuration>/attempt_NNN/solver.raw.log
```

## 11. 单独重新生成分析

```bash
CONFIGS=pdhg,pdhg_restart,pdhg_restart_scaling,pdhg_restart_scaling_reflection,pdhg_restart_scaling_reflection_adaptive_primal_weight,pdhg_restart_scaling_reflection_adaptive

python3 benchmark/ablation/analyze_ablation.py \
  --run-dir "$RUN_DIR" \
  --configs "$CONFIGS" \
  --tolerance 1e-6 \
  --timeout-value 600 \
  --sgm-shift 10 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260730 \
  --report "$RUN_DIR/report.md"

python3 \
  benchmark/progressive_ablation/analyze_progressive_ablation.py \
  --run-dir "$RUN_DIR" \
  --time-limit 600 \
  --tolerance 1e-6 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260730 \
  --report "$RUN_DIR/progressive_report.md"
```

## 12. 本机已完成的参考批次

本次六级正式结果为 378/378：

```text
benchmark/results/rebuttal/progressive_ablation/progressive_six_stage_600s_all_idle_20260730
```

详细结果总结：

```text
rebuttal_plan/progressive_ablation_six_stage_results.md
```

该批次在 GPU 2、3、4、5、7 启动；GPU 6 后来释放后，按第 8.1 节加入同一
scheduler session。实际实例到 GPU 的映射以该目录中的
`gpu_assignments.csv` 为准。
