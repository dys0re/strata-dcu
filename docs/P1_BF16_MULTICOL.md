# P1：gfx936 的 BF16 多列权重复用

2026-10-22 更新：后续测试执行[GPU 权重 ≤50% 标准](DCU_TEST_STANDARD.md)。本文原始 P1 速度使用 19,000 专家配置，属于超出该预算的历史结果；新标准的重测单独保存。

基线为 `089c5b4`。目标是降低一个请求内多个 MTP 验证位置的投影成本，适用于本机 Qwen3.8-F、BW/gfx936、AutoRound INT4 主模型及 BF16 非专家投影。正式端到端对照双方都开启 P0 自适应 MTP；没有把开启 MTP 本身的收益计入 P1。

## 实现

`STRATA_BF16_Q8_VEC4=1` 的原内核对每列分别分配 block。新增 `STRATA_BF16_Q8_COLS=2|4` 后，一个 block 为一段相同的权重同时处理最多两列或四列激活。每列仍使用原先的四个 FMA 累加器、求和括号和 lane 归约顺序。输入的 Q8_1 字节、BF16 权重及输出布局不变；不生成新权重文件。

默认不启用该开关。仅 gfx936/wave64 且原 vec4 开关打开时才生效，捕获计算图之前读取环境。T=1 始终保留原路径。最终复用条件为输入宽度至少 2048、输出行数至少 2048、矩阵至少 24MiB，其余形状保留原内核。

四列模式遇到 T=2 时用两列 kernel；其他窗口按四列分块，最后不足四列时只读写有效列。内核不分配内存、不同步主机、不重排权重，可直接捕获到原验证计算图。

## 测量方法和探索记录

生产 dispatch 的独立诊断程序 `bf16_q8_mmvq_bench` 使用 CUDA/HIP events 测量一个包含 16 次投影调用的图，预热后采集七次，报告中位数和范围。它逐位比较每列与独立单列调用的输出，并检查末尾 guard；这里的微秒数是算子时间，不是模型 TPS。

最初未筛选形状时，大输出头与 GDN 投影有收益，但小矩阵明显变慢。随后 8MiB 门槛的边界测试又发现约 10MiB 矩阵会回退，因此最终改为上述更保守的宽矩阵/24MiB 条件。原始 CSV 和失败的性能选择均保留，不将只对大矩阵的收益推广到全部投影。

正式端到端顺序为旧、新、新、旧。每个引擎先热身 128 tokens，再将英文、中文和代码请求各运行三次、每次 256 tokens，串行单请求，不复用前缀。每个方案包含两次独立引擎启动、每种工作负载六次请求；同一启动内的三次请求不是独立硬件复现实验。固定 19,000 专家槽、INT8 KV、512 上下文和 12 个 CPU workers，关闭专家交换、PCIe 分流和 route-tail skip。decode 排除模型加载与 prefill。

主对照不启用阶段计时插桩或 logits 文件输出。数值检查、带阶段计时的诊断与性能测量分开保存。所有远程新增构建与测试文件均位于 `/root/private_data/strata-dcu/p1`。

### 输出头算子时间

| 验证列数 | 原内核 μs | 新路径 μs | 比值（原/新） |
| --- | ---: | ---: | ---: |
| 1 | 1054.77 | 1054.67 | 1.000 |
| 2 | 2104.67 | 1633.89 | 1.288 |
| 3 | 3155.22 | 2481.13 | 1.272 |
| 4 | 4204.87 | 3039.26 | 1.384 |
| 8 | 8405.80 | 6065.56 | 1.386 |

输入宽度 2560、输出 248,320 行；每个数是七次图内测量的中位数。T=1 使用同一个原内核。算子收益不等于整机收益，其他投影、GDN/attention、CPU 专家和调度仍需执行。

## 主对照结果

| 工作负载 | P0 中位数（范围） | P1 中位数（范围） | 中位数变化 |
| --- | ---: | ---: | ---: |
| 英文故事 | 34.97（32.41–35.96） | 37.08（36.49–38.17） | +6.04% |
| 中文解释 | 40.17（39.39–40.38） | 41.34（40.65–42.53） | +2.92% |
| Python 代码与说明 | 46.46（44.87–47.37） | 48.54（47.96–51.38） | +4.49% |

单位为 token/s。每格包含两次引擎启动、六个请求；范围是本组最小值到最大值，不是置信区间。没有跨硬件、长上下文或统计显著性的结论。54 个请求的输出 token IDs 与 P0 对照一致，其中包含热身、长请求、短问答、固定 seed 重复及边界请求。STOP/恢复另记录在原生协议日志中。

自动核验与逐次数据见 [原始材料](../profile_output/p1/primary-results.json)。这些数字是在持久盘缓存位置上重新测得，未与迁移前的单次结果混合。

## 既有融合开关的评估

这些是同一新引擎上的独立探索，每个候选一次启动、每种长请求三次测量，不与主表合并。`QDEDUP` 在共享专家跨流开启时不生效，因此为它另设相同的关闭跨流控制组。所有候选仍开启自适应 MTP。

| 配置 | 英文 | 中文 | 代码 | 输出检查 |
| --- | ---: | ---: | ---: | --- |
| 关闭共享跨流，对照 | 33.02 | 37.87 | 47.22 | 17 请求与 P0 一致 |
| 关闭共享跨流 + QDEDUP | 32.16 | 38.41 | 49.35 | 17 请求与 P0 一致 |
| 原共享跨流 + QFUSE | 不作为等价速度结果 | — | — | 13/17 请求出现 token 差异 |

表内速度单位为 token/s。QDEDUP 本身在同流控制下的效果随任务变化，但关闭共享跨流后的英文、中文均慢于主表 P1。因此保留原跨流设置，QDEDUP 不开启。QFUSE 的进程正常完成且短问答通过，但英文从第一个 token 就与基线不同，中文和代码也有差异；没有把它的 TPS 计作等价加速，也不启用该开关。未对这条非等价路径作完整质量评测。

源码检查还发现 `QFUSE_HC` 的当前入口受 `STRATA_USE_HIP` 条件保护，本机使用的是 `STRATA_HIP_GFX906` 兼容后端。本轮未迁移、未测试该分支；不能将设置环境变量视为它已生效。

原始检查见 [融合结果](../profile_output/p1/fusion-results.json)。

## 诊断文件清单

| 文件 | 类型 | 位置 | 用途 |
| --- | --- | --- | --- |
| `CMakeLists.txt` | 修改  | 1228 | 注册诊断程序与两组独立开关的回归测试 |
| `src/kernels/bf16_q8_mmvq_bench.cpp` | 新建  | 1 | 新增独立生产 dispatch 算子诊断、图内 GPU 计时、逐位与越界检查 |
| `src/kernels/bf16_q8_mmvq_parity.cpp` | 修改 | 24 | 扩展 CPU double 参考和单列逐位对照，覆盖 1–8 列及可复用形状的尾行 |
| `profile_output/p1/benchmark.py` | 新建  | 1 | 新增串行模型测试驱动，保存命令、输出、性能、STOP/恢复与采样检查 |
| `profile_output/p1/analyze.py` | 新建  | 1 | 核对增强 MTP、公平配置及输出一致性，汇总各次原始测量 |
| `profile_output/p1/benchmark-primary.py` | 新建  | 1 | 主对照实际使用的冻结驱动版本，后续版本只增加显式共享流控制 |
| `profile_output/p1/verify_equivalence.py` | 新建  | 1 | 模型请求、首 token logits、seed、STOP 和边界的严格核对 |
| `profile_output/p1/analyze_variants.py` | 新建  | 1 | 检查融合候选是否生效及输出是否等价，保留非等价结果 |
| `profile_output/p1/audit_remote.py` | 新建 | 1 | 源码、模型 inventory、缓存 SHA、部署及进程清理的复现记录 |
| `profile_output/p1/` 下的日志与 CSV/JSON | 新建  | artifacts.tsv | 原始证据；单独诊断目录可清理，正常运行不依赖这些材料 |

这里的计时仅在独立诊断程序或显式 profiling 运行启用；正常推理没有新增时间戳插桩。


## 等价检查与运行入口

最终自适应模式和固定 T=8 各自的 11 个请求均与 P0 对照逐 token 相同，各 13 组首 token logits（每组 248,320 个有限 FP32 值）逐位相同。固定 T=8 的长请求实际平均验证宽度约 7.9，不只是设置了参数；带阶段计时的这些结果不用于 TPS 主表。两组均通过短问答、固定 seed 重复、单 token 输出限制、501-token 输入加 3-token 输出的合法边界，以及原生 STOP 后恢复。

CPU double 参考、单列逐位对照和 BF16 转换检查共三组、每组 128 种形状通过；最终生产 dispatch 的图内算子诊断两组各 96 种形状通过，含完整输出头、1–8 列、尾行、非完整列块和输出末尾 guard。原始证据和严格核验入口位于 [profile_output/p1](../profile_output/p1/)。

远程原入口 `/root/strata-dcu/build-gfx936/strata` 已通过软链接选择 `/root/private_data/strata-dcu/p1/build/strata`。P0 引擎备份保存在 `/root/private_data/strata-dcu/p1/strata-baseline-089c5b4`。开启本次复用及增强 MTP：

```bash
STRATA_BF16_Q8_COLS=4 bash /root/strata-dcu/run-autoround.sh \
  --mtp-mode adaptive --prompt '请详细解释 prefill 与 decode。' --max-new 256
```

原启动脚本已默认设置 `STRATA_BF16_Q8_VEC4=1`。用 `STRATA_BF16_Q8_COLS=0` 关闭本次多列复用；保持其他参数可单独对照。不要为本机等价推理打开 `STRATA_QFUSE=1`；本轮没有修改这个已有开关的实现。

复现生产 kernel 的检查：

```bash
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
p1dir=/root/private_data/strata-dcu/p1
cmake --build "$p1dir/build" --target strata bf16_q8_mmvq_parity bf16_q8_mmvq_bench -j 6
ctest --test-dir "$p1dir/build" -R '^bf16_q8_mmvq' --output-on-failure
STRATA_BF16_Q8_VEC4=1 STRATA_BF16_Q8_COLS=4 "$p1dir/build/bf16_q8_mmvq_bench"
```

复现模型测试时使用新的输出目录，驱动拒绝覆盖已有材料；历史主对照的冻结版本为 `benchmark-primary.py`。其余 `benchmark.py` 增加了显式跨流控制，用于融合消融。完整配置记录于各组 `command.json`，代码增量见 `p1-core.patch`（对 `089c5b4` 的隔离 index 应用检查通过）。

本轮仅验证 BW/gfx936、DTK 26.04、bs=1、INT8 KV、512 上下文。未验证 CUDA/SYCL 编译、其他硬件、长上下文、并发批处理或完整模型质量评测；没有将模拟测试或微内核速度写成模型 TPS。原始 checkpoint 使用 inventory 核对，没有重新对全部大模型文件和主专家包做全量 SHA-256 扫描；四个 MTP 运行文件另做全量 SHA-256 核对。源码与二进制审计见 `audit.json`。

原启动脚本另完成 64-token 冒烟：自动专家缓存、增强 MTP 开启、实际执行 T=2 窗口、退出 0，64 个输出 IDs 与主对照前缀相同。该入口检查没有并入固定 19,000 槽的 TPS 主表。检查结束后无残留推理进程，cgroup 未记录 OOM；`/root` 本地占用约 954MiB。

远程 `p1/build` 是当前引擎入口的软链接目标，清理日志时应保留它。逐文件证据 SHA-256 见 `profile_output/p1/artifacts.tsv`。
