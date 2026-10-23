# 原生 DFlash2：功能验收与速度记录

2026-10-23 清理后的重新验证见 [cleanup-20261023](cleanup-20261023/README.md)。以下保留 2026-10-22 的原始验收与速度记录；新账单已包含临时专家槽。

2026-10-22，Qwen3.8-F AutoRound，BW/gfx936，单卡 BS=1。原生网络已接入：完整五层、动态卷积、候选选择器、七候选上限、滑动 K/V 和 graph。主模型特征在 GPU 直接交给草稿，复用原有 BF16 head/embedding。未训练新草稿，未改变主模型权重。

## 先看 QA

| 检查 | 结果 |
| --- | --- |
| 固定七候选开发版 | 13 个请求、195 个输出 token IDs，与逐 token 参考全部一致 |
| 最终自适应原生版 | 14 个请求、198 个输出 token IDs，与逐 token 参考全部一致 |
| 实际答案与格式 | OK、四则运算、中文首都、大小写、排序 JSON、指定 JSON 字段通过 |
| 生成代码执行 | is_even 函数通过 -20..20 的 41 个输入；使用受限 AST/空 builtins 执行 |
| 多轮消息 | 正确回忆前轮给定的 ORBIT |
| 协议与边界 | 两次相同 seed 的采样、单 token、501-token 提示边界、STOP 与恢复通过 |
| 长提示与请求恢复 | 2100-token 提示后准确返回 ORBIT，下一请求返回 OK；上下文配置 4096 |
| 独立缓存检查 | 完整七候选、重复、请求重置、增量/完整重算、跨 2048 窗口、环形覆写通过 |
| graph/eager | 九个独立草稿用例的隐藏状态和候选 IDs 在两种原生执行模式下逐位一致 |
| 完整生成回归 | 三种 256-token 生成及 128-token 预热，与此前逐 token 参考 IDs 全部一致 |

功能驱动会校验实际回答、执行生成的函数，并核验每个请求 DONE 状态及 token 上限。各次命令、输出、完整 token IDs、权重账单和 cgroup 事件均保留。固定七候选 QA 在并行 top-k merge 优化之前执行；最终自适应 QA、长输入及生成回归使用最终二进制。完整窗口的最终 graph/eager 及缓存检查另有记录。

首次独立数值对照以 0.025 的隐藏状态相对 L2 阈值失败，日志保留于 `witness-check.log`。后续按功能 QA 验收，把跨实现的数值距离保留为诊断：本组最大相对 L2 约 0.06956，部分草稿候选与 PyTorch 不同。没有声称原生算子与 PyTorch 数值等价，也没有把首次失败改写为数值通过。最终输出一致性由主模型验证器和上述真实 QA 检查。

## 权重与速度

GPU 主模型专家固定 8300，其余 16276 在 RAM（41.909 GiB），未扫描或增加专家数。原生草稿参数 **1206131200 字节**，直接共享主模型已有 **1271398400 字节** BF16 输出头。全部 GPU 权重 **32783001600 字节 / 47.717196%**，低于物理显存一半 **34351349760 字节**。PLE 在磁盘；新派生包、构建与日志全部在持久盘。

原生自适应 1–7 候选、graph、512 上下文上限、INT8 KV、12 CPU workers、prefill 32、prompt/conversation cache 关闭，QFUSE/QDEDUP/route-tail-skip 关闭，VEC4/COLS=4。先预热 128 tokens，再每类单次生成 256 tokens：

| 工作负载 | decode TPS |
| --- | ---: |
| 英文故事 | 22.26 |
| 中文解释 | 24.33 |
| Python LRU 代码 | 33.06 |

decode 包含原生草稿、特征处理及验证，排除加载和 prefill。这是每类一次的速度记录，不作统计显著性结论。本轮没有超过既有增强 MTP，也低于此前 IPC 三候选记录。DFlash2 保持显式启用；未切换默认草稿方法。没有逐算子时间线来确定每项性能损失的占比。

## 入口与复现

使用 [原生接入文档](../../docs/DFLASH2_NATIVE.md)。驱动为 `tools/bench_dflash2_native.py`，`--qa-only` 优先执行功能用例；`--long-checks --context 4096` 执行长输入；`tools/report_dflash2_native.py` 直接校验原始数据并生成 `summary.json`。

最终二进制在远程 `/root/private_data/strata-dcu/dflash-native/strata-final`，SHA-256 `2aba1cff1291d1af0479188e50050a7b78ff4b619fd90ab5f0e155e4e8f5c42d`。真实运行平台仅为 HIP/gfx936。CUDA 13.3 + g++-15 对新增 dflash.cu 的编译通过；完整 CUDA 引擎、SYCL、Metal 未运行验证。最初本机默认 GCC 版本超过 nvcc 支持范围，改用已安装的 g++-15 后通过，没有跳过编译错误。

独立草稿输入大文件可由 `tools/test_dflash2_native.py prepare` 的 seed=1729 重建，原件保留远程；本地归档只保留小型输出及报告。模型/原生包/二进制不进入 Git。`build.log.gz` 是第一轮构建日志的无损压缩。

## 插桩与文件

| 文件 | 变更 | 记录内容 |
| --- | --- | --- |
| src/prefill/dflash.cu | 新增原生实现及计时 | 草稿调用数、特征行数、草稿墙钟耗时、实际参数字节 |
| src/program/generate.cpp | 追加预算账单字段 | native_draft 与 external_draft 分开计入同一权重上限 |
| tools/bench_dflash2_native.py | 新增功能/速度驱动 | 实际命令、源码/二进制 hash、QA 答案、输出 IDs、协议、内存事件 |
| tools/test_dflash2_native.py、src/prefill/dflash_witness.cpp | 新增独立诊断 | 真实权重前向、graph/eager、缓存和跨实现数值距离 |
| tools/report_dflash2_native.py | 新增原始结果校验 | 最终输出一致性、预算、协议、单轮速度汇总 |
| tools/autoround_pack.py | 新增正常入口 | 实际 backend、窗口/graph/adaptive 选项与生成报告 |

核心功能文件还包括 dflash_native.hpp、dflash_bridge.hpp、verify/prefill 的特征访问器、CMakeLists.txt 和 dflash2_pack.py，详见源码 diff。插桩未伪造 token、标签或耗时。

独立 fresh agent 按文档原样执行正常启动入口，退出 0，返回 OK；generation.json 的 native/graph/adaptive/最大宽度 8 和权重账单全部吻合，没有发现该入口的文档偏差。最终进程审计无运行中的推理/草稿任务，cgroup OOM 为 0，系统层占用 954 MiB。审计由 tools/audit_dflash2_native.py 保存为 audit.json；公共入口原始材料见 cli-native/。
