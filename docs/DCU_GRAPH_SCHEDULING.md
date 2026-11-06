# DCU CPU miss 图安排

单卡、CPU 处理所有未常驻专家时，`STRATA_CPU_MISS_GRAPH=1` 捕获专用图，裁掉 PCIe 工作表追加、等待、搬运和零组专家计算。共享专家与路由专家复用输入量化，GPU 结果相加并入最终合并。

`STRATA_FUSE_ROUTE_PUBLISH=1` 在专用图内进一步合并 top-k、CPU 输入发布和 GPU 常驻专家归组。256 个线程先写完 CPU 输入并发布已知轮次，再继续生成 GPU 工作表。CPU 结果完成等待、第一层 PLE 完成条件和缓冲区复用条件仍保留。

原始引擎默认关闭实验入口。当前只在 gfx936 的兼容构建验证；需要启用 GPU 常驻专家独立归组。正 PCIe 分流、自动分流、多 GPU、批处理、路由调整等不满足条件的配置使用通用图。两类图分别缓存，请求切换后不会误用专用图。服务协议的 `cpu_miss_graph=0` 可对单个请求选择通用图。

## 2026-11-03 实测

Hygon BW/gfx936，GPU0，12 个 CPU workers，8500 GPU 专家，Q8 dense/head 与 HC Q8、IQ4_NL PLE、INT8 KV、上下文512，adaptive MTP/maxT5。MTP 设置没有改变。

同一已加载进程先预热两类图，每类任务进行三组 ABBA/BAAB，每组每种模式两次256-token生成。正式测速未加载捕获审计器。

| 任务 | 通用图 TPS 中位数 | 优化图 TPS 中位数 | 变化 |
|---|---:|---:|---:|
| 英文 |38.784|39.949|+3.00%|
| 中文 |49.040|51.198|+4.40%|
| 代码 |59.513|60.002|+0.82%|

三项速度比几何平均为1.02730。各任务输出 token 完全一致。这是单个进程、共享主机上的描述性结果，不声称统计显著性或其他模型/设备的收益。

单 token 图的 kernel 提交数2245→1717，2–5 token 分别减少576次；五 token 为2290→1714。DTK 的外层图节点是聚合 packet，不能直接当作 kernel 数。本次另用捕获 API 审计实际提交的 kernel。

## 验证与限制

- HIP 实机80次底层检查：1–8 token、不同驻留比例、重复图重放；CPU 收到信号后立即读取输入，随后独立校验 CPU top-k、权重和专家工作表。
- 最终引擎18项回答/格式/代码 QA、同 seed 采样、取消恢复、单 token 和长输入边界通过。
- 实际执行0→0.3→0→0.3→0 PCIe 请求，发生专家搬运，并确认通用图与专用图缓存复用正确。
- GPU 权重账单29,373,360,352 B，未改变专家数，仍在半显存上限内。
- HIP Release 与 CUDA sm80 Release 编译通过。未做 NVIDIA 硬件、SYCL/Metal 或多 GPU 验证。

构建底层检查：`cmake --build build-gfx936 --target route_publish_test -j4`；在已配置的单 GPU 运行环境执行 `build-gfx936/route_publish_test`。正常推理在现有启动命令前设置上述两个环境变量；`STRATA_CPU_MISS_GRAPH=0` 恢复通用图。

原始实验记录和独立入口验收输出保存在本地 `profile_output/bzy2008-graph-20261103/`，不随源码提交。最终测试引擎 SHA256：`e6ae02899c9888d7c3108d11fd8763a69121607a73a4de326679d2fe96d79a34`。
