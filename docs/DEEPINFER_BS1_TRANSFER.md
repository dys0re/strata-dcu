# DeepInfer → Strata：Qwen3.8-F、bs=1 的迁移评估

检查日期：2026-10-21。DeepInfer 源码固定在 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`；Strata 基线为本次性能提交 `6ec15a8`。本报告是源码对照与工程判断，未运行 DeepInfer，未实现以下迁移，也没有新的迁移后 TPS 数据。

后续进展（2026-10-22）：P0 的实现与本机验证记录见 [单请求成本感知 MTP](MTP_ADAPTIVE.md)。下文保留最初的静态评估，其他优先级仍为建议。

这里的 bs=1 指一个并发请求；投机解码可在这个请求内一次验证多个候选位置。默认继续使用当前单卡 BW/gfx936、主模型 AutoRound INT4、CPU 专家补集，以及磁盘上的 PLE。若限制每次 forward 也只能处理一个位置，则下述多位置验证方案不适用。

DeepInfer 官方当前只列出 DeepSeek-V4.1-Flash，硬件为 NVIDIA SM100 和 Ascend 950；它不能直接作为本机 Qwen3.8-F 的同条件性能对照。[官方说明](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)

## 结论与优先级

| 优先级 | 可借鉴的特性 | 对 bs=1 的价值 | Strata 已有能力与实际缺口 |
| --- | --- | --- | --- |
| P0 | 按接受率和实测成本选择 MTP 验证长度 | 避免花时间验证低收益候选，让单请求每轮有效前进更多 token | 已有 MTP、`--spec-min-p`、窗口成本 EMA 和 lookup 策略；缺的是本地 AutoRound 入口的 MTP 接入，以及对所有 MTP 前缀长度的完整成本选择 |
| P1 | 小验证窗口的权重复用及算子融合 | 一个请求验证 2–8 个位置时，尽量一次读取权重供多列激活复用 | 已有多 token 验证、融合路由和部分量化融合；新 BF16 内核仍按 `blockIdx.y` 分列执行，可做多列 tile；gfx936 的融合路径还需要逐项验证 |
| P1 | 稳定采样与跨验证形状的一致性检查 | 窗口自适应和拒绝回滚必须保持目标采样分布、状态和 RNG 语义 | 已有采样 parity、按 seed/位置的随机数及窗口验证测试；需覆盖本机 AutoRound、GDN/PLE、可变窗口的组合 |
| P2 | GPU 元数据接力、延后 CPU 后处理 | 有机会隐藏 token 输出、下一轮元数据准备的空隙 | 已有捕获图、设备规划、单 token 自提交和映射缓冲；不能把 CPU 专家真实计算或磁盘依赖当作可直接消除的调度开销 |
| P2 | Mega-MoE 的融合执行思路 | 减少路由、激活量化、专家计算、combine 的中间读写和启动 | 优先作用于显存命中专家；保留 CPU miss 分支。DeepGEMM 的 SM100/MXFP4 内核和 NVLink 通信不能直接用于 gfx936/Q4_0 |
| 功能项 | 原生结构化输出 | 单请求也能从采样阶段保证 JSON/schema 合法，减少格式失败 | 当前 `serve/structured.py` 是生成后验证，明确没有原生 grammar decoder；这是实质功能缺口，但不属于通用 decode TPS 优化 |

最先做的应是 **本地 Qwen MTP 接入与成本测量**，随后补齐小窗口多列权重复用和自适应窗口策略。不要先引入面向多请求排序的完整调度器。

## 1. 可变长度验证：迁移选择方法，保留 Qwen 草稿模型

DeepInfer 的 `MTPVarlen` 同时使用验证性能曲线、在线 forward 成本修正和 sampler 成本估计，按预计有效 token / 耗时选择预算；DSpark 的接受率来自单独的 Confidence Head。其 GPU 调度会进行前缀概率累乘和候选排序。[成本选择代码](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp)、[GPU 长度分配](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py)

bs=1 无须跨请求全局排序。设验证 k 个草稿，A_j 是前 j 个草稿全部接受的概率，则可比较：

```text
score(k) = (1 + Σ[j=1..k] A_j) /
           (draft_ms(k) + verify_ms(k+1) + commit_ms(k) + other_round_ms(k))
```

候选必须包含 k=0，即不投机的真实单 token 路径。接受率低或草稿昂贵时允许退回基线；验证窗口变长不自动意味着更快。所有成本都应来自这台 DCU 的完整轮次实测，不能搬用 SM100 的曲线。

Strata 的 `src/spec/draft_policy.cpp:60` 已按成本在给定的 MTP 窗口与 lookup 窗口间选择，并搜索 lookup 长度；但 lookup 不存在时直接返回传入的 MTP 长度。`src/program/generate.cpp` 的 MTP 长度还主要受固定上限和 `--spec-min-p` 控制。这是可扩展的具体位置，不需要另起一个大调度框架。

本地检查发现 `model_extra_tensors.safetensors` 中已有 **1,565 个 BF16 MTP 张量，总计 5,214,301,696 bytes（4.856GiB）**；没有名称包含 confidence 的 MTP 张量。它们不能直接填入 DeepSeek 的 DFlash/ConfidenceHead。现有 `tools/mtp_pack.py`、`tools/mtp_rt.py` 期待另一种已融合的 MTP 张量布局，并会派生量化草稿缓存；当前 `tools/autoround_pack.py` 在导出时排除 `mtp.*`，运行时也没有传入 `--mtp`。

建议先适配本地 MTP 张量名称和布局，单独生成草稿缓存，保持主模型缓存和原 Safetensors 不变。原样 BF16 草稿已有约 4.86GiB，不能忽略它对当前 51.2GiB 主模型专家缓存的挤占；若用派生量化草稿，应把接受率变化及主模型 cache miss 一起计入收益。中文请求还要检查现有 draft-vocab 的 CJK 覆盖。

第一版可用已观测的接受深度和校准后的草稿概率估计 A_j，无须先训练一个新 Confidence Head。选窗与采样的依赖也必须正确：DeepInfer 明确约束后续候选的置信度不能反过来影响前面候选是否被验证，并用历史轮次信息分配预算。移植时需要检查这个约束，不能只照抄分数公式。[草稿与 Confidence Head](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/mtp/dflash.py)

官方报告的动态验证约 4–7% 收益来自约 60 个并发序列/rank 的生产点；其 16 序列样本反而为 −1.5%。这不是 bs=1 的性能证据，也不是可承诺给本机的收益。[官方 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#speculative-decoding)

## 2. 融合与小 M：结合本机数据，保留模型公式

DeepInfer 的 Mega-MoE 将 gate、共享缓冲区和 routed/shared 专家计算连接成融合路径；其 attention 辅助内核还合并 norm、RoPE、量化及 cache 写入。可借鉴的是减少中间张量、针对实际小形状选择布局和跨验证列复用权重。[MoE runtime](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mlp.py)、[融合变换](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_transform.py)

本机具体落点：

1. `src/kernels/cuda/native_mmvq.cu` 的 BF16/Q8_1 路径为多个验证列复用同一段权重，尤其是大输出头，而非每列分别扫描。
2. `src/core/verify.cpp` 已有 `STRATA_QFUSE`、`STRATA_QFUSE_HC`、`STRATA_VERIFY_QDEDUP` 等路径，先验证这些路径在 gfx936 的正确性和收益，避免重复写已有融合。
3. 仅为 VRAM-hit 专家分支考虑更大粒度融合；CPU 补集继续使用现有数据交接。完整 Mega-MoE 依赖的权重格式、设备能力和通信机制不适合直接复制。

Qwen 的 GDN、QSA、PLE、路由公式仍需保留。不能为了使用 FlashMLA 而替换 attention，也不能把 AutoRound 的整数 INT4 code / FP16 scale 当作 MXFP4 的浮点 code / ue8m0 scale。任何新增重排都应先证明解码权重不变；融合导致的浮点变化则单独验证。

## 3. 已有或不应直接迁移的部分

| DeepInfer 特性 | 当前判断 |
| --- | --- |
| CUDA Graph | Strata 已有捕获与复用；重点是图内剩余数据依赖和 HIP 调度，不是再加一层 graph |
| Engram 异步 fetch / 首次消费时等待 | Strata 已在 Layer 0 运行期间重叠 PLE 读取，`src/core/verify.cpp:2893` 之后才收集并放行 Layer 1；无需将磁盘 PLE 搬进显存来模仿 Engram |
| tokenizer compression、FP8 Engram 表 | Qwen 的 PLE hash/行号及权重格式属于模型语义，不能直接换成 DeepSeek 的映射或压缩格式 |
| SWA bounded replay / YOCO compact | 依赖 DeepSeek 的 SWA/CED/YOCO 结构，且官方描述的 SWA 恢复本身是近似；不能给 Qwen GDN 状态只重放最后 128/140 tokens 就声称等价 |
| prefix/state cache | Strata 已有保存 GDN、PLE、QSA/indexer 状态的 conversation snapshot；继续改进可降低多轮 TTFT，但不直接提升同一轮持续 decode TPS |
| Wide-EP、EPLB、DP/CP、P/D 分离、3FS | 当前单卡 bs=1 优先级很低；以后增加硬件和并发再另评估。部分生产存储与均衡代码也尚未开源 |

Engram 对照：[fetch hooks](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/engram.py)。ZOS 对照：[worker 的设备元数据修正与延迟后处理](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py)。现有计时中的同步等待不能全部解释为调度空隙，因此 ZOS 的收益需要独立时间线证明。

结构化输出是独立的服务能力迁移：DeepInfer 在采样前推进 grammar automaton 并施加合法 token mask；Strata 目前生成后检查 JSON/schema。应使用 Qwen tokenizer 构造 automaton 和 mask，并与投机验证同步回滚；它主要改善格式保证，不应承诺普通文本 decode 更快。

## 建议的实施顺序与验收

1. 以 `6ec15a8`、bs=1、MTP off 为控制组，记录当前约 33–37 token/s 的工作负载及所有缓存预算。
2. 接入本地 Qwen MTP，先比较草稿数 0/1/2/3/4，记录 draft/verify/commit、各深度接受率、CPU 专家数、显存和 TTFT。中英文、代码各自报告，不先保证 MTP 一定更快。
3. 对验证窗口补齐 BF16 多列权重复用和 gfx936 融合，分别做开关对照。
4. 在这些实测曲线上加入单序列的成本选窗，始终保留 k=0 回退；不迁移无用的跨请求全局排序。
5. 检查 EOS、停止词、拒绝回滚、GDN/卷积/PLE 历史、相同 seed 的采样分布，以及 T=1/2/4/8 的前缀验证。新窗口策略不能靠丢专家或截断历史换速度。

比较口径固定原主模型权重、KV 格式、上下文、并发数和硬件；MTP 增加的内存预算必须公开。时间线与吞吐测试分开，避免重复使用插桩后的 TPS。此报告中的优先级和迁移方案均为工程推断，只有之前的 `6ec15a8` 性能优化有本机实测数据。
