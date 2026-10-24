# HC 自检回退与配置复核（2026-10-23）

本轮是实际 gfx936 小算子诊断，未重新加载大模型，未改变生产源码、权重或入口。当前生产二进制仍为 `1551c2b1cc1624a29ab9b17f2276599c2a35648c58babc417e2c6ad98947b1b1`。

## 原日志报什么

`staged/split read differs ... lo ... T=3 ... at 640 (3bef675e, not 3bef673e)` 是随机权重/输入启动自检的逐位比较失败，不是回答质量检查。两个数为 0.007306023500859737 与 0.0073060085996985435，差 1.4901161193847656e-8，相对差约 2.04e-6。

HC 是每层对四路残差做归一化、降到 320 维、再展开并混合的部分。`split` 将归一化按 token 和残差流分给更多 block；`staged` 还调整 down 投影的数据暂存。HIP 上该 staged 实现没有 CUDA cp.async，仍是普通复制；尚未证明它在这台 DCU 上更快。它们与另一个显式 `STRATA_GR_SPLIT` K 维切分开关不是同一实现。

## 原因：最早差异在归一化

把原自检扩展到 rs（归一化系数）、xn（归一化后输入）、lo（低维结果）、inject、mixed 和更新后的残差；1–8 token，带/不带前序残差写回。同一个 seed 20260930。单 token 的融合参考不提供 xn，因此只比较其有定义的输出。

默认 AMD fast 路径的首个分歧为第三个 token 的第四路归一化系数：`3f7eef5b` 与 `3f7eef5c`，相差 1 ULP（5.96046447754e-8）。然后 xn、lo、inject、mixed 出现微小差异；原检查先比较 lo，因而日志没有指出真正最早的分歧。

| 编译/运行对照 | 数组比较数 | 有差异的比较数 | 非有限值 |
| --- | ---: | ---: | ---: |
| 原编译方式，GR_FAST=0 | 185 | 0 | 0 |
| 原编译方式，GR_FAST=1（生产默认） | 185 | 70 | 0 |
| 整个诊断翻译单元 -ffp-contract=off，GR_FAST=0/1 | 各185 | 均0 | 均0 |
| 仅两个 fast 归一化函数 contract(off)，GR_FAST=0/1 | 各185 | 均0 | 均0 |

这组对照将原因定位到 fast 归一化表达式的隐式浮点乘加融合：编译器合并乘法和加法，改变中间舍入；两种执行组织得到不同的最低有效位。仅在两个函数约束这一行为，就消除了全部已测分歧。没有更换矩阵权重、放宽判定阈值或跳过输出检查。

`norm-contract-candidate.patch` 是两处局部编译约束候选，**未应用到生产**。这些是小算子结果，不是模型 QA 或端到端加速证据。此轮没有测这些候选路径的整机 TPS。

## 配置确实有哪些关闭

- 当前公布的 benchmark：`--adapt-every 0 --adapt-swaps 0`，禁用原版运行中交换热门专家；`--pcie-frac 0`，未命中专家全部交 CPU；`--suffix-draft 0`，关闭从已有文本寻找候选。
- AutoRound 正常入口始终传 `--adapt-every 0 --suffix-draft 0`；DFlash 入口额外硬编码 `--pcie-frac 0 --adapt-swaps 0`。单独 MTP 的普通入口没有强制 PCIe share 为零，不能把 benchmark 的全部设置说成每一种入口都相同。
- 核心引擎原默认 adapt_every=4、adapt_swaps=96、suffix_draft=3；native pack 的 PCIe share 会探测后选择。增强 MTP 当前验证条件又要求 suffix_draft=0，恢复该项涉及组合逻辑。
- benchmark 还明确关闭 route-tail skip。它会跳过部分专家，改变实际计算和命中率统计；是否收益必须结合 QA 与真实配置检查。
- QFUSE/QDEDUP 等有些原本就是可选项，不能全部声称为“关闭了默认优化”。

50% GPU 权重上限本身不要求禁用专家交换或分流。本轮只确认设置与作用，未测恢复它们的 TPS，因此不把这些关闭直接认定为两倍差距的已证实原因。R9700 的实际缓存命中率尚未取得，38.2% 只能描述本机该请求。

## 量化形式：支持机制，尚未证明本机归因

[ICML 2023 SmoothQuant](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) 用 CUTLASS INT8 GEMM 替换浮点矩阵计算，Table 7 中 OPT-30B、A100、BS=1 的 decode 为 1.27–1.35× 加速；同文也发现 LLM.int8 降低内存却常慢于 FP16。存储位数减少不自动等于加速。

[MLSys 2024 FlashDecoding++](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf) §6.3.2–3：NVIDIA 上有收益的 flat GEMM 双缓冲在 AMD MI210 上没有收益，AMD 主要依靠适合实际形状的 GEMV/执行方式选择。

所以 BF16/Q8 差异是有机制支持的待验证因素，不是当前 2× 性能差距的结论。需要本机同形状、同验证宽度的成对算子测量和模型 QA，不能搬用上面的加速比。

## 产物与临时诊断改动

| 文件 | 操作与用途 |
| --- | --- |
| hc_probe.cu | 原 fused_gr.cu 的独立副本，仅扩展自检输出；没有改原内核 |
| hc_norm_probe.cu | 在副本的两个归一化函数添加局部 contract(off) |
| build_probe.py | 从现有 Ninja 元数据提取相同编译/链接参数，构建三个版本，串行执行六组对照 |
| diagnostic.patch | 独立诊断相对生产源码的完整差异 |
| norm-contract-candidate.patch | 两处局部约束候选，不含诊断代码 |
| *.csv、*.log、commands.json、source.json、summary.json | 原始比较、编译记录、精确命令和哈希 |

构建及二进制保存在 `/root/private_data/strata-dcu/hc-diagnosis-20261023/`。没有依赖安装，没有开启主模型的新功能，没有把显存预算用于增加专家数。
