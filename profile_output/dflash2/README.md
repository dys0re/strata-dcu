# DFlash2 接入与本机测试

2026-10-22，Qwen3.8-F AutoRound、BW/gfx936，BS=1。已接入真实预训练 DFlash2：主模型仍由 Strata 执行，PyTorch 进程运行草稿，通过私有 socket 提交候选。实现了两条 prefill 路径的特征抓取、提交前缀的草稿 K/V、HIP Graph 和按实测成本选择候选数。现有 AutoRound 入口支持 `--dflash2 DIR`，默认增强 MTP 保留。

本轮代码有收益，英文和中文下降；没有将 DFlash2 设置成默认草稿方法。

## 实测速度

| 工作负载 | 增强 MTP | DFlash2 graph，固定 3 候选 | DFlash2 graph，自适应 1–3 候选 | 自适应相对 MTP |
| --- | ---: | ---: | ---: | ---: |
| 英文故事 | 29.47 | 24.22 | 27.25 | -7.54% |
| 中文解释 | 37.52 | 30.23 | 31.93 | -14.90% |
| Python LRU 代码与说明 | 38.26 | 41.86 | 41.81 | +9.30% |

单位 token/s，为每类三次 256-token 生成的中位数，各模式一次独立启动。各次速度、范围和完整命令见 [summary.json](summary.json) 及各目录 command.json。每次先预热 128 tokens；decode_ms 排除载入和 prefill，包含特征下载、进程通信及草稿运行。上下文上限 512、INT8 KV、12 workers、PLE direct IO、greedy 主测试；QFUSE/QDEDUP 和 route-tail-skip 关闭，VEC4=1、COLS=4，prompt/conversation cache 关闭。没有给出统计显著性、其他任务或长上下文结论。

增强 MTP 使用 8,500 个 GPU 主模型专家；DFlash2 使用 8,300 个，以容纳新增的草稿和复制输出头。两者都执行 GPU 全部模型权重不超过 50% 的要求，没有进行专家数扫描。这是各自满足预算的运行配置，不能将变化解释成单个算子的加速。

## 权重与内存

DFlash2 网络 603,065,600 个 BF16 参数，占 1,206,131,200 字节；草稿进程复制主模型 BF16 输出头 1,271,398,400 字节，外部 GPU 权重共 **2,477,529,600 字节（2.307379 GiB）**，全部纳入 C++ 启动预算。

全部 GPU 权重为 **34,054,400,000 字节 = 31.715631 GiB，占总显存 49.5678%**，低于 31.992188 GiB 上限。CPU 放 16,276 个主模型专家，41.909409 GiB。embedding 在主机，PLE 保留在原磁盘文件。

初始化时设备总占用 33.937500 GiB，包含 PyTorch allocator、图内工作区、KV、激活和驱动；这不是峰值，也不能把设备总占用当作权重占用。各次 weight-usage.json 是实际账单。

## 正确性与回归

固定及自适应 DFlash2 的全部 34 个请求、5,140 个输出 token IDs，与相同 8,300 专家布局、相同输入/采样参数的逐 token 参考完全一致。包含三种长回答及其重复、短问答、两个 seed=1729 采样请求、单 token 上限和 501+3 上下文边界。两种模式的 STOP 取消与后续请求恢复均通过。

增强 MTP 回归的 10 个请求输出与之前 weight50/run1 一致。未观察到 OOM 或被拒绝的内核启动。独立 fresh agent 按环境说明原样执行真实权重 witness，退出 0；重复草稿、增量缓存与完整重算相同，12 个 graph 生效。

以上是这些用例的验证，未做完整质量基准；只在此 HIP/gfx936 平台构建和执行，CUDA/SYCL/Metal 未验证。

## 草稿耗时与收益判断

固定 graph 模式有 984 次 graph 调用，中位草稿耗时约 7.35 ms；自适应模式的 graph 草稿中位耗时 **7.16 ms**。该耗时包含 worker 的输入准备和 GPU 前向/输出选择，不含 C++ 侧特征下载。自适应整个测试中候选数 1/2/3 的调用数为 865/156/228。

代码请求能利用更长接受前缀；英文和中文更多选择短窗口。日志里仍有较多 CPU 专家工作，验证额外位置的成本会增加。由此判断，在此专家 offload 配置下，减少草稿串行性并不能保证所有任务加速。尚未做完整逐算子时间线来量化每项瓶颈，不能将全部差距归因于 IPC 或 CPU。

## 失败与范围

最初 smoke1 因短提示词通过 verifier 读取、未转发该路径的特征而失败，诊断为 position 16 vs 0；已补齐，失败记录保留。修复后的 eager smoke2 四个请求均与逐 token 参考一致，64-token 英文为 19.91 token/s。随后加入 graph 和自适应选窗；eager 数值仅是开发诊断。

首次 CLI 验证沿用了引擎的 route-tail-skip 默认值，该结果不用于速度或全模型正确性结论。正式 `--dflash2` 入口已与基准一致，关闭此近似开关及未验证的融合分支，单独复跑。完整使用方式见 [接入文档](../../docs/DFLASH2_IPC.md)。

## 文件与插桩清单

| 文件 | 类型 | 作用 |
| --- | --- | --- |
| include/strata/core/dflash_bridge.hpp | 新增 | 五层残差均值抓取、socket 协议、外部权重字节握手 |
| include/strata/core/verify.hpp、src/core/verify.cpp | 修改 | 验证窗口抓取并导出五层特征 |
| include/strata/prefill/prefill.hpp、src/prefill/prefill.cpp | 修改 | 批量 prefill 特征抓取及导出 |
| src/program/generate.cpp | 修改 | 两条 prompt 路径、只发送提交前缀、候选验证、显存预算和支持范围检查 |
| tools/dflash2_worker.py | 新增 | 参考模型绑定、动态/静态 K/V、graph、选窗、每轮分阶段墙钟统计及 witness |
| tools/strata_dflash2.py | 新增 | 启动、私有 socket、权重/耗时统计、进程清理 |
| tools/autoround_pack.py | 修改 | `--dflash2` 单次生成入口、协议适配、实际权重和草稿报告 |
| tools/download_dflash2.py | 新增 | HF Mirror / ModelScope、断点传输、大小和 SHA-256 校验 |
| third_party/dflash/model.py、LICENSE、VERSION | 新增 | 未改动的上游参考模型及 MIT 许可证/固定版本 |
| profile_output/dflash2/benchmark.py | 新增 | 实际命令/源代码哈希、请求/输出、预算/内存事件、STOP/恢复 |
| profile_output/dflash2/analyze.py、report.py、audit_remote.py | 新增 | 校验原始结果、输出一致性、统计报告及来源/部署审计 |
| profile_output/dflash2/env-spec.json、.aris/compute/scnet-dflash2.md | 新增 | 声明式环境记录和 fresh agent 验证说明 |
| docs/DFLASH2_IPC.md、envs.md | 新增/修改 | 入口、范围、参数、预算和实测记录 |

插桩仅记录实际执行。没有修改主模型权重或原始 Safetensors，没有把 PLE 搬进显存。原始材料、checkpoint 下载回执、witness 和失败日志保留。大权重、构建及远程日志均在个人持久盘。

`complete-source.patch` 是相对当前 HEAD `089c5b4` 的完整源码包，包含此前 P1、权重预算和本轮 DFlash2 改动；已用隔离 Git index 验证可应用，未修改真实暂存区。`audit.json` 记录实际源文件/二进制哈希、正式入口、旧二进制备份和清理状态。单次 CLI 的实际生成报告在 `cli-cache/generation.json`。
