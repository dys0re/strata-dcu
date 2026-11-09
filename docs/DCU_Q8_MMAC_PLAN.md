# gfx936 Q8 MMAC 执行布局

`STRATA_Q8_MMAC=1` 为符合条件的 Q8 投影与输出头使用 gfx936 INT8 MMAC。
`STRATA_Q8_MMAC_STABLE=1` 按矩阵形状固定 split-K，使 T=1–5 的相同输入前缀
使用相同累加路径。两个选项在通用引擎中均为显式开关；已验证的机器入口开启二者。
依赖 [Q8 模型变体](DCU_MODEL_OPTIMIZATIONS.md)，不重新量化 AutoRound 专家。

每 32 个量化值先做精确整数点积，再乘该组 FP16 权重/激活缩放并累加到 FP32。
布局为 `Wq[k32_group][output_tile16][lane64][8 int8]`，缩放独立排列。
输出行尾部补零并计入权重账单。原始 INT8 编码及 FP16 scale 保持不变。

初版按窗口改变累加策略，虽然语义 QA 通过，同 seed 采样在第 22 token 分叉。
stable 模式修正后，六形状的 2,821,120 次跨宽度前缀比较差异为零；
相对原 Q8 算子仍允许约 1e-7 的舍入误差，不声称与旧算子逐位一致或模型质量无损。
孤立四位置算子约 1.9 倍的原型结果不能当作端到端加速。

## 单份执行权重（2026-11-03）

`ExecutionWeight` 将 GGUF 量化类型与设备布局分开描述，记录矩阵尺寸、设备数据、
权重字节和归约工作区。它是借用视图；`NativeDense` / `NativeHead` 持有分配，
必须在所有可能执行的 graph 结束使用后才释放；释放后不得重放旧图。
启动时选定布局，token 路径不分配或转换整份权重。

开启 `STRATA_Q8_MMAC=1` 时，支持的 Q8 投影直接从主机 GGUF 排列为唯一的 MMAC 设备副本。
原来的 canonical-pointer → MMAC-copy 全局表已移除。`STRATA_Q8_PACKED=1` 不再为这些矩阵另建副本。
不支持的矩阵仍使用 GGUF 布局；PLE key 的专用接口明确保留 GGUF 布局。

单位置、验证窗口与共享专家使用带布局的 MMVQ 接口。T=6–8 以及非 stable 模式的
小矩阵单位置使用读取同一 MMAC 布局的向量内核，不依赖隐藏的旧权重。
`STRATA_Q8_MMAC_STABLE=1` 保留 T=1–5 的原有固定分段策略。

prefill 从执行布局直接反量化到原有有界行块工作区，也支持带 padding 的输出行。
MTP 子词表抽行重建原始 Q8_0 编码和 FP16 scale，不反量化再量化，不还原完整输出头。
完整输出头的 MTP 和 draft proxy 也读取该执行视图；BF16 DFlash 接口仍明确只取 GGUF 数据。
`NativeHead::weights()` 对非 GGUF 布局返回空，避免新的调用者把执行字节误当连续 GGUF 行。

可构建的检查目标：

```bash
cmake --build <hip-build> --target q8_execution_weight_test -j4
STRATA_Q8_MMAC=1 STRATA_Q8_MMAC_STABLE=1 <hip-build>/q8_execution_weight_test --require-stable
STRATA_Q8_MMAC=1 STRATA_Q8_MMAC_STABLE=0 <hip-build>/q8_execution_weight_test
```

检查包含七种尺寸（完整词表与非 16 对齐尾行）、T=1–8、每宽度两次 graph 重放，
以及切片 FP16 反量化、原编码抽行和 width2560 的 proxy 打包/重评分。
数值参考在释放旧 GGUF 设备副本前生成；所有后续矩阵执行都只剩新布局。
性能、实际显存和模型 QA 的记录必须与具体节点/二进制绑定，不能用这个检查代替。

向量回退使用相邻 lane 读取相邻 MMAC 权重片，四个 lane 合算一行，保留原始量化组缩放。
单位置速度取决于尺寸；热 graph 微基准中部分尺寸仍慢于 GGUF 路径，不承诺所有形状都有加速。
该阶段完成执行权重统一；同输入投影组、跨算子融合与共享工作区调度仍属于后续工作。


### 单份布局的实测验证

2026-11-03，BW/gfx936 单卡 GPU0，12 workers + 主控，8500 GPU 专家，
context512、INT8 KV、Q8 dense/head + HC Q8、tuned adaptive MTP/maxT5、IQ4_NL PLE。
旧二进制 `6c700bc4` 与新二进制 `c9398e63`，两边显式启用 MMAC、stable 和 HC Q8。
每边一次进程启动，先预热128 token，再对每个领域各生成三次256 token；下表为中位 TPS。

| 项目 | 双份布局 | 单份布局 |
|---|---:|---:|
| GPU 权重账单（B，含相同 staging） |32,856,352,992|29,373,360,352|
| 英文 TPS |37.013|37.294|
| 中文 TPS |49.192|47.991|
| 代码 TPS |56.808|59.543|

释放3,482,992,640 B（3.24 GiB），专家数保持8500。三个领域速度比的几何平均为1.010；
样本只支持“这次测试速度接近”，不能声称统计显著加速，也不是显存峰值测量。

最终 HIP Release 构建和七形状 GPU 检查通过；最大相对误差1.84e-7，
FP16 行块、Q8 原编码抽行以及六种 width2560 形状的 proxy 打包/重评分通过。
T=1–5 stable 前缀差异为0；fast 模式允许浮点末位差异，未作为数值失败。
最终二进制通过18项语义/代码 QA、同 seed 重复、STOP/恢复和长度边界；
500-token 输入实际走过32-token 分块 prefill。

CUDA13.3 + GCC15、sm80 Release 编译通过，未做 NVIDIA 硬件运行。
SYCL/Metal 未构建或运行；fast 模式未跑整模型 QA；未做广泛模型质量基准。
原始记录保存在本地 `profile_output/bzy2008-execution-20261103/`，不随源码提交。
