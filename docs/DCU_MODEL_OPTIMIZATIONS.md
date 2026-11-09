# Qwen3.8-F 的权重、MTP 与草稿输出头

本文保留模型变体的准备方法及各自日期的验证记录。历史 `/root/private_data` 路径、
引擎摘要和瞬时状态不代表当前部署；最新环境见 [envs.md](../envs.md)，
当前队列测量见 [CPU 队列](DCU_HETERO_SCHEDULING.md)。

这组实现基于提交 faf59ad5，继续固定增强 MTP 的 8500 个 GPU 专家、原生/联合模式的 8300 个。
全部 GPU 模型权重仍受物理显存 50% 上限约束。PLE 和大文件继续放持久盘。
本机增强 MTP 入口已默认选择 HC Q8、Q8 投影/输出头与蒸馏 MTP；筛选头默认关闭。
原生 DFlash2 和联合模式保留此前的默认权重。

## HC 与非专家矩阵

`STRATA_HC_Q8=1` 启用原有 HC Q8_0 路径。它在载入时从 BF16 生成八位权重，
由 NativeDense 计入权重账单，跳过相应 BF16 常驻副本；预填充需要的还原缓冲单列为运行时工作区。
它与 STRATA_HC_REQ8 是两条不同路径，不要混用。

`tools/dcu_model_variants.py dense` 从当前 dense.gguf 的原始 BF16 投影直接生成 Q8_0。
它不修改 AutoRound INT4 专家，不量化 router、indexer、HC、PLE 或 embedding；其余张量及元数据保留。
每 32 个权重一组，存 FP16 缩放值和 32 个 INT8 值，避免先从其他量化副本重新量化。

```bash
export STRATA_GGUF_PY=/root/strata-dcu/llama.cpp/gguf-py
python3 /root/strata-dcu/Strata/tools/dcu_model_variants.py dense \
  --source /root/private_data/strata-dcu/autoround/dense.gguf \
  --out /root/private_data/strata-dcu/model-specific-20261024/dense-q8.gguf
```

`--scope projections` 只处理主要投影；`--scope head` 只处理输出头；默认 all 处理两者。
完整变体含 301 个矩阵，少存 3,323,904,000 字节。输出旁的 JSON 保存源、输出摘要与逐矩阵字节数。

公开入口用 `--dense-gguf FILE` 选择变体。入口显式启用 STRATA_DENSE_Q8_VARIANT，
允许同一 canonical BF16 矩阵附上 Q8_0 执行副本，维度和矩阵名仍必须匹配。
不传选项时保留原有类型一致性检查。权重使用实际载入字节记账，不用省下的空间增加专家数。
主模型的投影/输出头量化会改变数值，验收看实际答案、格式和代码执行。
预填充沿用现有的可复用权重还原工作区，按矩阵或行片还原到 FP16 后计算，
不在 GPU 缓存一整套额外 BF16 权重。decode 直接读取 Q8_0 矩阵。

原生 DFlash2 目前仍要求共享 BF16 输出头及 embedding；与它搭配时使用 projections 范围，
不要传把输出头也改成 Q8 的 all/head 变体。

## 已发布的蒸馏 MTP

源为 jesdga/Qwen3.8-Flash-Next-Basalt，固定发布版本 a229744b82e18c3e12057ab944062c0c713c7006，
只下载 convert/mtp/rt 的 dense.bin、dense.txt 和说明，不下载整个模型包。
远程 HF Mirror 连接超时，同名 ModelScope 模型返回 404；本机 HF Mirror 下载后中转至持久盘。

dense.bin 为 116,099,072 字节，SHA256：
`dfe18647e0c95d05302907be88f021590119645a1567fcd770c8d309df852d75`。
发布说明注明它在 IQ3_S 输出上自蒸馏。它是否适合本机 AutoRound 目标模型由实际验收决定。

本地原始 dense 的摘要与发布的 dense-original.bin 相同，原始专家摘要也与发布版本一致。
准备工具仍逐项核对名称、类型、形状、长度、范围和不重叠性，并按本地布局写入，
保留本地 experts.bin、draft_vocab.bin 的符号链接。原始运行目录不改写。

```bash
python3 /root/strata-dcu/Strata/tools/dcu_model_variants.py mtp \
  --base-runtime /root/private_data/strata-dcu/autoround/mtp \
  --source /root/private_data/strata-dcu/model-specific-20261024/basalt-mtp-source \
  --out /root/private_data/strata-dcu/model-specific-20261024/mtp-tuned
```

运行入口增加 `--mtp-runtime DIR`，按现有 source-bound ready.json 核验。
这个参数只在 MTP 确实参与时允许使用，原生 DFlash2 单独运行不需要它。

## 两位草稿头筛选

`STRATA_DRAFT_HEAD_PROXY=1` 仅改变 MTP 草稿头，主模型验证头不调用此功能。
当前支持宽度 2560、BF16 或 Q8_0 输出头。载入时生成两位权重副本，每组另存 FP16 缩放值；
副本包含在 MTP 权重账单及载入前预算中，scratch 不计作模型权重。

每次单位置草稿先扫描整个草稿词表；每 32 行保留四个候选，再按分数及行号确定性地取最多 64 个，
候选还必须位于最高近似分数以下五个 logit 以内。用原始权重重新计算这些行；其他行保留近似值。
因此草稿分数、概率和选择都可能变化。多个位置的调用继续使用完整头，不能把它理解成跳过主模型验证。

筛选方法及初版存储布局改编自 Basalt 588a4776 的 head_proxy.cuh，MIT 许可与作者说明见
third_party/basalt-head-proxy/LICENSE 和源文件头。DCU 路径另外处理兼容后端的计算成本。

## 正常测速

2026-10-24，BW/gfx936，BS=1，上下文 512、INT8 KV、12 workers 加主控线程。
每配置独立启动一次，预热 128 tokens，英文、中文和代码分别重复三次 256-token 生成，
表内为 decode 中位数，排除载入与 prefill；测试时没有同时构建、准备权重或记录逐层事件。

| 配置 | 英文 TPS | 中文 TPS | 代码 TPS |
|---|---:|---:|---:|
| faf59ad5 原二进制 | 33.72 | 42.01 | 44.62 |
| Q8 投影＋输出头 | 36.63 | 46.16 | 56.04 |
| 再加蒸馏 MTP | 37.19 | 46.10 | 56.22 |
| 再加 HC Q8 | 36.20 | 48.51 | 56.98 |
| Q8＋蒸馏 MTP＋筛选头（HC 保持原路径） | 36.54 | 48.15 | 56.59 |

样本只有三次，不作统计显著性或其他硬件上的收益结论。
这些组合中选择 HC Q8＋Q8 投影/输出头＋蒸馏 MTP；筛选头保留为可选项。
权重为 29,373,360,352 字节，专家仍为 8500 个，没有把省出的显存变成更多专家。

## 验收记录

原始记录见 [本轮报告](../profile_output/model-specific-20261024/README.md)。
独立检查者按文档原样执行一次，增强 MTP、原生 DFlash2、联合模式均通过；GPU 回到 2 MiB。
底层检查区分权重准备、算子数值、图重复执行、真实回答与正常吞吐；微基准不代替端到端结果。


同组短提示的 prefill 中位数（英文／中文／代码）从 689.1／627.8／576.8 ms
变为 562.6／518.5／453.2 ms。这只覆盖本轮三个短提示，不代表长上下文吞吐。

生产编译选项下，完整 BF16 草稿头约 455 μs，筛选约 570 μs；完整 Q8 头约 296 μs，
筛选约 563 μs。筛选中的候选排序占约 325 μs，当前 DCU 上省下的读权重时间不足以覆盖它。
因此保留功能，默认关闭，不能把其他量化改动的收益归给筛选。

后续补充测量期间 CPU 专家处理明显变慢，旧二进制也复现了约 9 TPS 的短英文生成。
原因尚未定位。该阶段与上表分开记录，不用它估计筛选收益，也不宣称机器当前持续可达到上表速度。

恢复此前的增强 MTP 权重选择：

```bash
STRATA_DCU_MODEL_DEFAULTS=0 bash /root/strata-dcu/run-autoround.sh --prompt 'Reply with exactly OK.' --max-new 16
```

本机配置是持久盘的 `model-specific-20261024/selected-model.json`。
显式 `--dense-gguf`、`--mtp-runtime` 和 `STRATA_HC_Q8`、`STRATA_DRAFT_HEAD_PROXY` 可覆盖各项。
这些默认值只由本机 wrapper 启用，通用引擎不自行选择机器上的权重变体。

2026-10-25同一配置新增可选布尔值`cpu_q4_input_bias`，当前本机增强MTP选择true，
用于[CPU INT4输入修正复用](DCU_KERNEL_OPTIMIZATIONS.md#cpu-q4-专家)。显式`STRATA_CPU_Q4_INPUT_BIAS=0`可关闭。
`STRATA_DCU_MODEL_DEFAULTS=0`也会停用该配置项的默认注入；用户显式环境值仍然有效。


2026-10-25 补充：后段骤降的主要原因已通过同进程换核对照定位为固定核心上的宿主用户态争用。
当前默认组合的256-token英文生成从9.72TPS恢复到33.24TPS，移回后12.18TPS。
这次只定位原因，没有更改生产绑核默认值；详见 [诊断记录](../profile_output/cpu-slowdown-20261025/README.md)。

## PLE 四位表（2026-11-03）

在一台 BW/gfx936、32 CPU / 121.5 GiB 内存的 worker 上，
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` 保留了 BF16 PLE：
128 个 `[2500012, 160]` 存储分片，共 320001536 行。已从原始 BF16 直接生成独立的
IQ4_NL GGUF，没有改写原 checkpoint 或再次量化专家。每 32 个数保存 16 字节编码和
2 字节 FP16 scale，每行 90 字节，实际为 4.5 bits/weight。

输出为独立的 `ple-iq4_nl.gguf`，
大小 28800138816 字节（28.80 GB / 26.82 GiB），原分片为 102400512256 字节。
输出 SHA256 为 `744182bb2bb330a6a006c94fa3c5be669915ed75f9ec9d58e2ca91bd29dae899`。
原始分片仍保留，因此此次生成副本没有回收原文件占用的空间。

转换使用项目固定的 GGML `3cf03257f219afbe7334045ff7c6a06ac68c627d`，
调用 `quantize_iq4_nl` 的标准 scale 搜索，不使用 imatrix。
在全部原始分片中采样 16639 行、2662240 个值，Q4_0 / IQ4_NL 的相对重建 RMSE
分别为 8.5593% / 7.6013%。这是权重重建误差，不是模型准确率。
两种格式同为 90 字节/行；IQ4_NL 使用非均匀码本，Q4_0 使用均匀刻度。
Q4_K 通常需要 256 元素块，不能直接用于这里的 160 维行；Q4_K_M/S 则是整模型的混合精度配方。

`tools/ple_quant_pack.py` 分块读取，内存随 worker 数和 chunk 大小有界。示例：

```bash
python3 tools/ple_quant_pack.py \
  --model /path/to/Qwen3.8-Flash-Next-W4A16-AutoRound \
  --libggml /path/to/pinned-build/bin/libggml-base.so \
  --format IQ4_NL --workers 28 \
  --source-sha256 59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf \
  --out /persistent/ple-iq4_nl.gguf
```

工具保留 `.part`，源文件 SHA256、输出完整回读 SHA256 和抽样重新量化均通过后才发布结果。
不覆盖已有结果。当前不支持恢复未完成的量化；部分结果保留供检查，重新执行应选择新的输出名。
转换结束后的 4352 行覆盖原始分片边界、随机位置和 EOF，写出的编码与 GGML 重新量化逐位一致，
独立 NumPy 解码与 GGML 解码逐位一致。使用实际 Strata `PleTable` 源码编译的
`tools/check_ple_table.cpp` 在 direct / mmap 两种路径上也与独立解码逐位一致。

`tools/autoround_pack.py run` 新增可选的 `--ple-gguf /persistent/ple-iq4_nl.gguf`，
并将显式选择记录到生成报告。不传时继续使用原来的 BF16 descriptor。
上述转换检查不代替完整模型质量基准。后续已部署完整文本推理缓存并完成单卡生成检查；
MMAC 的模型级验证范围见 [MMAC 接入记录](DCU_Q8_MMAC_PLAN.md)。
原始 HF checkpoint 的索引仍依赖 BF16 源分片，不能把删除该片后的目录当作完整 HF checkpoint。

原始结果：[转换与对照](../profile_output/ple-iq4-20261103/conversion.json)、
[Strata 读取检查](../profile_output/ple-iq4-20261103/reader_check.json)。
