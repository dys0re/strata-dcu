# gfx936：Q4_0 多 token CPU 内核与 DFlash2 小 batch 投影

2026-10-23，基于 `c4932aa`，Qwen3.8-F AutoRound，BW/gfx936，BS=1。模型权重及量化不变。

2026-10-25补充：[输入修正复用](DCU_Q4_INPUT_BIAS.md) 增加显式单位置优化，
本机增强MTP默认开启。下文单位置继续ggml的说明适用于关闭新开关的路径，历史测速数据保留。

## 使用

引擎提供两个显式开关，默认关闭：

- `STRATA_CPU_Q4_MT=1`：Q4_0 CPU gate/up 和 down 为多 token 复用权重解码；单 token 继续用 ggml。需要 AVX-512 VL/BW/VNNI 与现有 CPU 能力检查通过。没有编译此内核或 CPU 不支持时保持原路径。
- `STRATA_DFLASH_SMALL_GEMV=1`：原生 DFlash2 在 gfx936 使用 BF16 小 batch 投影并直接写 BF16 输出。1–8 行支持；预填充的较大 batch 继续使用原 GEMM。完整词表输出头在三行以上继续使用 BLAS，因为实测新内核在这些形状更慢。

本机 `strata-test/run-autoround.sh` 默认启用两项，可分别设置为 `0` 复测旧路径。默认草稿仍是增强 MTP；显式传入 `--dflash2` 才使用 DFlash2。

远程正常入口：

```bash
bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --prompt 'Reply with exactly OK.' --max-new 16
```

答案应为 `OK`。`/root/private_data/strata-dcu/autoround/generation.json` 的 `kernel_options.cpu_q4_mt` 和 `kernel_options.dflash_small_gemv` 应均为 true。这些字段记录请求的选项；实际 CPU/GPU 分发由硬件检查决定。

正式引擎位于持久盘 `/root/private_data/strata-dcu/q4-multitoken-20261023/strata-final`，SHA-256 为 `d8e293371f4bc399baf7b1562ef516af475b7870791ecd60f2d8f9bb4a1b9ac5`。

## 实现

CPU 内核把 gate/up 放在同一循环内，复用激活读取和 Q4_0 解码。每片最多四个 token，大组均衡切片。使用 256-bit EVEX/VNNI 匹配 32 元素量化块；第一版把两个块拼成 512-bit 的实现，在两候选时反而较慢，未采用。另处理 INT8 `-128` 的符号翻转边界。普通量化激活的完整专家测试与 ggml 参考结果逐位一致；不是依赖输出文本一致来验证算子。

GPU 内核使用 wave64、合并读取及最多四列的权重复用，FP32 累加后直接按原规则舍入为 BF16。没有把激活改成 INT8，也没有新增权重副本。不同的求和顺序可以改变 BF16 末位及草稿候选；主模型验证仍执行。

没有恢复此前删除的路由预测器或调度策略。SYCL/Metal 未编译此 CPU 内核时，通过构建宏保持其原有分发，不新增未解析符号。

## 端到端结果

相同提示，每次预热 128 tokens，然后英文、中文、代码各三次 256-token 生成。每个配置为一次独立引擎启动，表中是三次中位数，不作统计显著性结论。关闭提示缓存、专家交换和 PCIe 专家分流；512 上下文、INT8 KV、12 workers、PLE direct IO。

| 路径 | 英文 TPS | 中文 TPS | 代码 TPS |
| --- | ---: | ---: | ---: |
| 增强 MTP，CPU 优化关闭 | 29.97 | 38.02 | 39.17 |
| 增强 MTP，CPU 优化开启 | 31.59 | 38.88 | 40.29 |
| 变化 | +5.4% | +2.3% | +2.9% |
| 原生 DFlash2，两项关闭 | 21.90 | 24.40 | 33.71 |
| 原生 DFlash2，两项开启 | 29.89 | 33.19 | 43.58 |
| 变化 | +36.5% | +36.0% | +29.3% |

DFlash2 在本轮代码任务超过增强 MTP，英文和中文仍较慢，因此没有将默认草稿切换为 DFlash2。CPU 和 GPU 两项对 DFlash2 的提升未在本轮单独做端到端消融；CPU 的独立端到端收益见增强 MTP 的开关对照。

固定 GPU 专家数：增强 MTP 8500、原生 DFlash2 8300。全部 GPU 权重分别为 33,542,205,952 字节（48.822253%）与 32,827,238,400 字节（47.781584%），含临时专家槽。没有通过增加专家数取得收益。

MTP 开关对照使用同一个冻结二进制。DFlash2 基线二进制在最后增加跨后端编译保护宏之前构建；该宏在当前构建恒为 1，对此主机保留相同 CPU 内核和分发代码。各二进制 SHA、源码快照及配置均记录在原始材料，不能把不同构建描述为同一二进制。

## 验证与边界

- CPU：独立标量解码、奇数块、带符号/零 scale、INT8 极值、行切片、1–17 token，共 23,562 项检查；实际 gate/up、隐藏量化及 down 对照 ggml。AVX-512 主机与强制 AVX2 回退均通过。另在实际不支持 AVX-512 的本地 Xeon 上通过回退测试。
- GPU：覆盖所有草稿投影形状及尾行小形状的 104 组测试，独立 CPU double 参考、输出 guard、多列/单列一致性和图/eager 一致性通过。部分微基准未被生产分发选用，尤其是三行以上的完整词表头。
- 真实回答：14 个语义/格式/采样/边界用例，198 输出 tokens，包括四则运算、JSON、生成代码执行、多轮记忆；STOP 与恢复通过。
- 生命周期：九个真实权重草稿缓存用例在 eager/graph 均通过，覆盖重复、重置、增量/重算、2048 窗口和环形覆写。与 PyTorch 的最大隐藏状态相对 L2 约 0.058，只作诊断，不声称跨后端数值等价。
- 长输入：2100-token 输入准确返回 ORBIT，随后请求返回 OK；配置上下文 4096。不是完整长上下文质量评测。
- 完整 HIP 引擎运行验证通过。新增 GPU 单元及修改后的 DFlash 单元通过 CUDA 13.3 编译；没有验证完整 CUDA、SYCL、Metal 引擎运行。

原始记录与汇总见 [本轮材料](../profile_output/q4-multitoken-20261023/README.md)。所有大文件留在持久盘，测试禁用 core dump。
