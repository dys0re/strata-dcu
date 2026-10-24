# Qwen3.8-F DCU 测试标准

2026-10-22 起，后续速度测试要求：**全部 GPU 模型权重分配合计不超过物理总显存的 50%**。

计入主模型 pack（包括共享专家及路由器等）、原生 dense 投影、主输出头、GPU 路由专家，以及 MTP 的投影、专家、独立输出头和额外量化副本。以实际分配字节统计，重复副本分别计入。主机映射的 embedding 和磁盘上的 PLE 不计为 VRAM 权重；PLE 当前读取行属于运行时工作数据。

KV、状态、激活、临时工作区、运行时读取行、计算图、索引表及驱动开销单独记录，不用“空闲显存 50%”或“专家缓存 50%”代替权重约束。启动日志输出权重字节账单和当时设备总占用；后者减去权重是包含分配器及驱动开销的余量，不能当作精确 KV 用量。

本机启动脚本与 `tools/autoround_pack.py run` 默认固定 8,500 专家，并开启增强 MTP；`strata-test/run-autoround.sh` 固定设置 `STRATA_MAX_WEIGHT_VRAM_FRACTION=0.5`。引擎先按预算约束自动缓存，再在 drafter 绑定完成后核验实际权重分配，超限退出。显式专家数超出预算也退出。该约束当前支持单 GPU native-pack 路径；未覆盖的多卡、批量 drafter 等配置会明确拒绝，不发布有效结果。

本轮重测固定 8,500 个 GPU 主模型专家，其余 16,076 个留在 RAM，不扫描专家数。增强 MTP（adaptive）开启，BF16 VEC4 和 COLS=4 开启，QFUSE/QDEDUP 关闭；BS=1、上下文上限 512、INT8 KV、12 个 CPU workers、磁盘 PLE。提示与此前 P1 的英文故事、中文解释、Python LRU 相同，每次生成 256 tokens，预热后每类重复三次。decode 以引擎 DONE 的 decode_ms 计算，排除载入和 prefill。

复现驱动：[benchmark.py](../profile_output/weight50/benchmark.py)。须使用支持权重预算的新二进制；驱动在发送请求前检查账单，缺少或超过上限则失败。历史 P1 及更早结果保留原始条件，其中 19,000 或自动满显存专家配置不符合本标准，不能作为本标准下的速度结果。

本轮已完成重测：[权重账单、六次中位数和原始材料](../profile_output/weight50/README.md)。英文/中文/代码为 29.48 / 37.66 / 38.51 token/s；全部权重占显存 48.758%。

DFlash2 等外部草稿进程的 GPU 权重也必须计入同一个预算，不能只统计 Strata 进程。当前 [DFlash2 实验](DFLASH2_IPC.md) 使用 8,300 个主模型 GPU 专家以容纳额外草稿输出头副本，全部权重 49.5678%；原增强 MTP 基线仍为 8,500 个专家。两者没有进行专家数扫描。

[原生 DFlash2](DFLASH2_NATIVE.md) 继续固定 8300 个 GPU 主模型专家，复用主模型输出头。草稿参数独立计为 `native_draft`，全部 GPU 权重含 16 个临时专家槽（44,236,800 字节），合计 32827238400 字节，占显存 47.781584%；`external_draft=0`。预算包含两类草稿字段，不能因草稿改为进程内执行而漏计。节省的重复头空间不用于专家数扫描。

2026-10-23 清理后复测：验证器临时专家槽统一计入 staging。增强 MTP、8500 个常驻专家的账单为 33542205952 字节（48.822253%）；上文旧测试保留历史统计值。

## 外部性能参照：R9700 的 Q4（2026-10-23）

后续以 Strata 上游公开的 **Radeon AI PRO R9700 32 GB + EPYC 7452、UD-Q4_K_XL、专家无需读盘**结果作为主要外部工程参照。它的短故事为 **63–64 token/s**，代码为 **85–88 token/s**，不是此前误用的 RTX 5070 / IQ3_S 约 53 token/s。

来源：[上游 QUANTS.md 的 Speed, and the bound](https://github.com/Niko1221/Strata/blob/main/docs/QUANTS.md#speed-and-the-bound)。公开条件：GPU 限功耗 225 W，EPYC 7452 32 核、8 通道 DDR4-2400，32 线程顺序读约 106 GB/s；40K 上下文配置、INT8 KV、MTP、自动专家缓存，取两次请求中的第二次。该记录不是本项目复跑的数据，也没有按本机的 50% 权重规则重新测量。

用它定位具体差距，内部改动的收益仍须在本机同一模型及权重预算下实测。当前固定专家数与全部 GPU 权重 ≤50% 的要求继续执行。比较时必须分别核对：

- 专家和非专家矩阵格式：上游 UD-Q4_K_XL 是混合 Q4/Q5/Q8 专家，主要非专家矩阵为 Q8_0；本机为 AutoRound INT4 专家与 BF16/F32 非专家矩阵。不能只凭名称中的 Q4 认为每轮读写量相同。
- 每轮候选数、有效输出数、CPU/GPU 实际承担的专家调用数和缓存命中率。固定专家数不代表相同的实际工作分配。
- 当前真实内核及回退路径。本机 gfx936 使用 wave64 兼容后端，不等同于 R9700 的 gfx12 HIP 路径；修复前 HC staged/split 因自检差异回退到 plain；本轮修复归一化的乘加合并差异后，两者通过严格自检，staged 的 QA 与实际速度结果见 [DCU 运行选项](DCU_RUNTIME_OPTIONS.md)。
- 原版功能的实际开关。本机测速显式关闭专家交换、PCIe 分流和 route-tail skip；不能将该配置默认视为上游测速配置，也不能未经实测声称打开它们能补齐差距。

上游文字提供的 CPU 1.2 ms 是按 RAM 带宽估算的项，不是可直接与本机完整专家池耗时相除的实测 CPU 时间。上游使用 STRATA_VERIFY_PROFILE 的阶段记录也应与正常测速区分。当前本机时序及配置见 [验证流程生命周期](../profile_output/verify-lifecycle-20261023/README.md)。

2026-10-24：入口现显式暴露缓存交换、锁页 RAM、PCIe 比例及搬运方式、suffix 查找选项，默认选择以本机实测和 QA 为准。测试仍固定各模式的专家槽数，未修改量化格式或 CPU 线程池。

同日后续的 [层内交接改动](DCU_LAYER_SCHEDULING.md) 修改了 CPU 任务安排及结果通知，专家数和权重预算不变。
测速须同时记录 `STRATA_VERIFY_HIT_PLAN`、`STRATA_CPU_EXPERT_PIPELINE`、`STRATA_CPU_EARLY_RESULT`、
`STRATA_PCIE_STAGED_MAX` 和 `STRATA_POOL_IDLE_US`。工作线程占用包含忙等；带逐层记录的运行仅用于定位，
正式吞吐取不加这些记录的运行。不得把不同模式的专家数或草稿权重差异算作调度收益。


2026-10-24 的 [模型变体验收](DCU_MODEL_OPTIMIZATIONS.md) 继续固定 8500/8300 个专家。
当前增强 MTP 的 Q8 组合全部权重为 29,373,360,352 字节；HC Q8 载入替代的副本按实际分配计入，
草稿筛选额外副本计入 `mtp`，`mtp_proxy` 只是其中的明细，不能重复相加。
还原工作区独立于模型权重；不建立整套常驻 BF16 投影副本。
报告须记录 dense 文件摘要、MTP ready.json 来源、`STRATA_DENSE_Q8_VARIANT`、
`STRATA_HC_Q8`、`STRATA_DRAFT_HEAD_PROXY`，以及原有调度选项。
本轮加入四项语义与代码执行 QA，共 18 项；不以长输出 token 一致作为通过条件。
