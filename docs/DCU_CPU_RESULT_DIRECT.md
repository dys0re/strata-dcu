# CPU 专家输出直接合并

`STRATA_CPU_RESULT_DIRECT=1` 让 GPU 在 CPU 完成后，直接读取映射内存中的专家输出并加权合并。
省去将这些输出复制到显存临时区、再加入 GPU 专家结果的步骤。结果保持 FP32，路由、专家数、
量化权重和逐专家加权顺序不变。没有增加 GPU 权重副本。
原临时缓冲区仍保留供回退使用；新路径省去该阶段对它的读写，没有宣称释放这部分显存。

通用引擎默认关闭，`STRATA_CPU_RESULT_DIRECT=0` 使用原流程。
本机增强 MTP 是否默认启用，由持久盘模型配置的 `cpu_result_direct` 决定；显式环境变量优先。
原生 DFlash2 和联合模式仍需显式设置。
本轮四次交替冷启动未建立整体 TPS 收益，因此本机也保持默认关闭。
英文/中文/代码合并六次的中位数为旧版34.46/44.88/53.24、新路径34.03/43.89/54.81 TPS。
这些是描述性结果，没有统计显著性声明。

```bash
STRATA_CPU_RESULT_DIRECT=1 bash /root/strata-dcu/run-autoround.sh --prompt 'Reply with exactly OK.' --max-new 16
STRATA_CPU_RESULT_DIRECT=0 bash /root/strata-dcu/run-autoround.sh --prompt 'Reply with exactly OK.' --max-new 16
```

当前仅为 `STRATA_HIP_GFX906` 兼容构建选择这条路径。要求单 GPU、原生专家、每个位置10个专家、
1–8个位置，以及对齐的 FP32 输出。多 GPU、远端专家、设备自行跳过 CPU 的工作表、旧的
`STRATA_VERIFY_PROFILE` 及分流成本记录等特殊执行条件沿用原流程。
全常驻图也保留原有快速路径。其他后端没有开启这项分发。

共享专家的侧流先完成，合并算子再等待原有 CPU 完成信号并读取结果。
第一层的 PLE 磁盘读取仍在同一个完成条件内；主机输出区的复用仍由下一层路由的先后关系约束。
没有提前发布未完成的行，也没有增加异步缓冲区。

`generation.json` 的 `kernel_options.cpu_result_direct` 记录请求值。
满足分发条件并捕获相应图时，日志打印 `strata verify: direct CPU result combine selected`。
仅有请求值不能证明该硬件和执行方式选择了新路径。

完整层记录仍可开启。新路径中GPU标记23到34包含“等待+直接读取+合并”；旧路径该段只含结果拷贝。
两种路径应比较22到24的整个结果返回区间，不能把23到34直接当成同一项。
独立原型的GPU执行时间不等于完整decode提速。

接入后的实测、质量检查、默认选择及完整原始记录见
[本轮材料](../profile_output/cpu-result-direct-20261025/README.md)。
