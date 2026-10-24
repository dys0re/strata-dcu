# CPU INT4 专家的输入修正复用

`STRATA_CPU_Q4_INPUT_BIAS=1` 为 Q4_0 CPU 专家复用输入修正，并优化只服务一个位置的任务。
它需要同时启用 `STRATA_CPU_Q4_MT=1`，且 CPU 通过已有 AVX-512 VL/BW/VNNI 检查。
其他量化格式、没有编译该内核或 CPU 不支持时，继续走原有分发。
通用引擎默认关闭，设置为 `0` 保留此前单位置 ggml、多位置融合计算的路径。

本机增强MTP的模型配置已选择开启；原生DFlash2和联合模式保持显式选择。
本轮英文/中文/代码合并六次的中位数为37.90/48.35/58.61TPS，相对旧版为+2.24%/-1.71%/+3.06%。
收益较小且有波动，不代表每类任务都更快。`STRATA_CPU_Q4_INPUT_BIAS=0`可覆盖本机默认值。

显式启用本机入口：

```bash
STRATA_CPU_Q4_INPUT_BIAS=1 bash /root/strata-dcu/run-autoround.sh --prompt 'Reply with exactly OK.' --max-new 16
```

运行报告的 `kernel_options.cpu_q4_input_bias` 记录请求的开关；实际使用该路径时，
引擎打印 `strata cpu q4: shared input bias and single-position rows active`。
能力检查失败时不会打印该标记，不能仅凭报告字段判断实际执行。

Q4_0 权重整数是 `u-8`，其中存储的 nibble `u` 在0到15之间。
新内核使用 `sum(u*y)-8*sum(y)`：每四个Q8输入的修正先计算一次，各权重行复用。
它减少每行的符号转换和绝对值计算，包含输入为-128的情况。
块缩放、浮点累加顺序和最终输出处理沿用原实现。
修正在每个CPU任务内准备，没有引入共享可写缓存、GPU权重副本或专家搬运。

测试与正常速度记录见 [本轮材料](../profile_output/q4-input-bias-20261025/README.md)。
前期独立原型的 [约10.5%矩阵耗时减少](../profile_output/optimization-angle-20261025/README.md)
不等于完整专家池或decode的提速。
