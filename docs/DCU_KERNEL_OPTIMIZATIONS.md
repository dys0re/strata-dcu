# DCU 内核优化

面向 BW/gfx936、单卡、BS=1 文本推理。通用引擎的可选优化需要显式开启；
机器 wrapper 的选项由其模型配置决定。最新测量见 [CPU 队列](DCU_HETERO_SCHEDULING.md)、
[图调度](DCU_GRAPH_SCHEDULING.md)和 [Q8 执行布局](DCU_Q8_MMAC_PLAN.md)。

## CPU Q4 专家

`STRATA_CPU_Q4_MT=1` 在 gate/up 与 down 中复用多个位置的 Q4_0 解包，
每片最多四个 token。需要 AVX-512 VL/BW/VNNI 和现有能力检查；不支持时回退原路径。

`STRATA_CPU_Q4_INPUT_BIAS=1` 依赖上述开关，并优化单位置任务。
对 Q4_0 整数 `u-8` 使用 `sum(u*y)-8*sum(y)`，按任务准备并复用输入修正，
保留块缩放和浮点累加顺序，包含 INT8 -128 边界。没有共享可写缓存或额外权重副本。
实际分发日志为 `strata cpu q4: shared input bias and single-position rows active`；
报告中的请求值不能单独证明路径已执行。

标量参考、完整专家及线程池检查覆盖极值、行切片、多位置和不支持 AVX-512 的回退。
2026-10-25 输入修正对照的英文/中文/代码中位数变化为 +2.24%/-1.71%/+3.06%，
属于单机描述性结果，不保证所有任务更快。原始材料位于本地
`profile_output/q4-input-bias-20261025/`。

## gfx936 整数点积和 Q8 读取

`strata_dp4a` 在 gfx936 使用 `__builtin_amdgcn_sdot4`，保留不饱和的 32 位累加。
Q8_0 块只有两字节对齐；使用 `memcpy` 连续读取相邻 64 位数据，避免未对齐强转。
Q6_K 缩放局部限制乘加合并，保持不同验证宽度的一致性。没有新运行开关或权重格式。

1,310,720 项整数参考检查通过。通用合成专家程序仍存在已用旧库复现的 FP16/GGML
转换差异，不能将该完整程序列为通过。只切换点积的初版端到端退步约 5.6%；
补齐连续读取后的匹配核心/NUMA 对照三域速度比几何平均 +1.80%。
原始记录位于本地 `profile_output/gfx936-dot4-20261025/`。

## BF16 投影和 DFlash2

`STRATA_BF16_Q8_VEC4=1` 启用向量读取；`STRATA_BF16_Q8_COLS=2|4`
在 gfx936/wave64 为多个验证列复用权重，要求输入/输出至少 2048、矩阵至少 24 MiB。
单列和小矩阵回退。保持每列求和顺序，不生成新权重。
早期 P1 吞吐使用 19,000 专家，超出当前半显存预算，不能作为当前性能基线。
QDEDUP 未建立净收益；QFUSE 出现输出差异，两者均未选为本机默认。
历史数据位于本地 `profile_output/p1/`。

`STRATA_DFLASH_SMALL_GEMV=1` 为原生 DFlash2 的 1–8 行投影复用 BF16 权重，
FP32 累加后写 BF16；较大 prefill 和三行以上完整词表头保留原 GEMM/BLAS。
104 组形状/guard/图检查通过；与 PyTorch 的隐藏状态差异仅作诊断，不声称数值等价。
使用方法见 [原生 DFlash2](DFLASH2_NATIVE.md)。

## CPU 输出直接合并

`STRATA_CPU_RESULT_DIRECT=1` 让 GPU 等待 CPU 完成后直接读取映射 FP32 输出并合并，
省去结果复制；保留回退缓冲区、第一层 PLE 完成条件及缓冲区复用协议。
要求 gfx936 兼容构建、单 GPU、原生专家、每位置 10 个专家、1–8 个位置及对齐输出。
不满足条件时使用原流程。

该路径未建立整体 TPS 收益，当前机器入口关闭。实际选择标记为
`strata verify: direct CPU result combine selected`。
分析逐层记录时应比较完整返回区间，不能把旧版复制时间与新版等待/读取/合并时间直接相除。
原始材料位于本地 `profile_output/cpu-result-direct-20261025/`。
