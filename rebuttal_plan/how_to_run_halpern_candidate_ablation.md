# Halpern restart-candidate 额外消融实验复现说明

## 1. 实验目的

新的求解路径始终使用 reflection 生成主迭代序列。Halpern 更新不再反馈到
下一次 PDHG 迭代，而只作为 restart 时的第三个候选点。

本实验比较：

| 配置 | 主序列 | Restart candidates |
|---|---|---|
| `with_halpern_candidate` | reflection | current、mean、Halpern |
| `without_halpern_candidate` | reflection | current、mean |

除 `use_halpern` 外，两个配置的 scaling、adaptive step、restart 和
reflection 设置完全相同。

## 2. 正式设置

- 数据：`benchmark/represent_data` 中固定的 63 个 CBF 实例；
- GPU：只允许使用物理 GPU 5；
- tolerance：`1e-6`；
- 每个实例/配置 time limit：600 秒；
- 每个实例内随机化两个配置的运行顺序；
- 所有历史 attempt、`solver.raw.log` 和 `result.json` 均保留；
- 失败任务使用 `--resume true` 创建新的 attempt，不覆盖旧数据。

## 3. 编译生产 kernel

在仓库根目录运行：

```bash
cd /home/zhenwei/PDCS_fork
make -C src/pdcs_gpu/cuda utils.ptx \
  CUDA_HOME=/usr/local/cuda \
  ARCH=sm_90
```

这里编译的是生产版 `utils.ptx`，不得使用任何 `*_profile.ptx` 做正式计时。

## 4. 运行实现测试

```bash
cd /home/zhenwei/PDCS_fork
CUDA_VISIBLE_DEVICES=5 \
PDCS_SKIP_GPU_PRECOMPILE=1 \
JULIA_PKG_PRECOMPILE_AUTO=0 \
./.julia-bin/julia-1.12.6/bin/julia \
  --startup-file=no \
  --compiled-modules=existing \
  -O1 \
  --project=. \
  test/test_ablation_update_gpu.jl
```

预期输出：

```text
ABLATION_UPDATE_GPU_PASS
```

## 5. 启动正式实验

```bash
cd /home/zhenwei/PDCS_fork
bash benchmark/halpern_candidate_ablation/run_halpern_candidate_ablation.sh \
  --run-id halpern_candidate_600s_gpu5 \
  --output-dir benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5 \
  --julia ./.julia-bin/julia-1.12.6/bin/julia \
  --time-limit 600 \
  --tolerance 1e-6 \
  --order-seed 20260730 \
  --resume true
```

脚本固定物理 GPU 5，并在 GPU 5 非空闲时拒绝启动。脚本不接受其他 GPU。

## 6. 中断后继续

重复执行第 5 节的同一命令即可。存在 case-level `DONE` 的任务会跳过；缺失
任务会创建 `attempt_002`、`attempt_003` 等新目录。

不要删除旧 attempt，也不要手工创建 `DONE`。

## 7. 输出结构

```text
benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5/
├── manifest.csv
├── run_order.csv
├── environment.txt
├── raw_results.csv
├── summary_overall.csv
├── summary_by_size.csv
├── summary_by_cone_mix.csv
├── paired_effects.csv
├── report.md
└── cases/
    └── <instance>/
        ├── with_halpern_candidate/
        │   ├── DONE
        │   └── attempt_001/
        │       ├── solver.raw.log
        │       ├── result.json
        │       └── DONE
        └── without_halpern_candidate/
            └── ...
```

`result_metrics` 额外记录：

- `restart_count`；
- `restart_current_count`；
- `restart_mean_count`；
- `restart_halpern_count`。

关闭 Halpern candidate 时，`restart_halpern_count` 必须严格等于 0。

## 8. 完整性检查

正式实验应满足：

```bash
RUN_DIR=benchmark/results/rebuttal/halpern_candidate/halpern_candidate_600s_gpu5

find "$RUN_DIR/cases" -mindepth 3 -maxdepth 3 -name DONE -type f | wc -l
```

预期为：

```text
126
```

并检查：

```bash
grep -F 'Status: **COMPLETE** (126/126 formal records).' "$RUN_DIR/report.md"
```

只有 `COMPLETE` 报告可用于论文。
