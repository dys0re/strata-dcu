# 当前版本 decode 生命周期：2026-10-23

基准提交 `c4932aa022869e2ba8441a2665a5d1a130c30d1f`。本轮实际运行于 SCNet 的 BW/gfx936，单请求、12 CPU workers、512 上下文、INT8 KV、磁盘 PLE。增强 MTP 固定 8500 个驻留专家；原生 DFlash2 固定 8300 个。全部 GPU 权重分别为 48.822253% 和 47.781584%，含临时专家槽，没有扫描或增加专家数量。

结论：CPU 专家结果和执行计划的等待已经直接进入 GPU 关键路径。CPU Q4_0 专家的多 token 实现缺少融合，是本轮找到的具体实现缺口；原生 DFlash2 还增加了每轮约 20 ms 的特征处理与草稿计算。此前把主要嫌疑放在 GPU 大投影、或只按草稿 forward 计时判断 DFlash2 开销，都不完整。

## 1. 正常执行的结果

每个配置独立启动，预热 128 tokens，英文/中文/代码各生成 256 tokens 一次。正常运行仅输出已有主机计时，不启用 GPU 阶段插桩。下表是本轮测量，不是多次重复的中位数。

| 配置 | 英文 TPS | 中文 TPS | 代码 TPS |
| --- | ---: | ---: | ---: |
| 当前增强 MTP | 25.87 | 36.36 | 37.71 |
| 当前原生 DFlash2 | 21.90 | 24.54 | 34.67 |
| 增强 MTP，CPU 节点 7 对照 | 29.56 | 32.27 | 40.27 |

原始材料：`mtp-normal/`、`native-normal/`、`mtp-numa7/`。三类请求相同，关闭前缀缓存、驻留交换和 PCIe 专家分流。工作负载差别不能直接当作硬件加速比。

正常增强 MTP 的每窗口主机计时：

| 工作负载 | 输出 tokens/window | 验证 ms/window | CPU 专家运算 ms/window | 完整草稿阶段 ms/window |
| --- | ---: | ---: | ---: | ---: |
| 英文 | 1.37 | 51.33 | 21.19 | 1.47 |
| 中文 | 2.03 | 53.11 | 20.14 | 2.62 |
| 代码 | 2.37 | 59.29 | 26.12 | 3.42 |

CPU 运算与 GPU 计算有重叠，不能将 CPU 计时再加到 GPU 总时间上。主机的 GPU-reach wait 是 CPU 等待 GPU 推进的时间，也不能解释成 GPU 等待。

## 2. GPU 时间线：等 CPU 的部分有多大

诊断副本使用 GPU wall clock 时间戳，保留生产路径的共享专家跨流、融合和图执行。记录每层主流阶段、共享专家侧流起止及对应 CPU 时间，另记录原生草稿阶段。

增强 MTP 的诊断结果，单位均为 **ms/实际输出 token**：

| 主模型 GPU 主流阶段 | 英文 | 中文 | 代码 |
| --- | ---: | ---: | ---: |
| 等 CPU 专家结果 | 11.36 | 5.88 | 7.78 |
| 等计划并复制计划 | 2.71 | 2.21 | 1.69 |
| 大投影、HC 读投影及输出头合计 | 10.34 | 9.75 | 8.44 |
| GPU 驻留路由专家 | 2.28 | 3.16 | 2.45 |
| 其他算子、合并、层间间隔 | 7.33 | 7.94 | 5.35 |
| GPU 主流总跨度 | 34.02 | 28.94 | 25.71 |

英文中，计划和 CPU 结果等待合计约 14.07 ms/token，占诊断 GPU 主流跨度约 41%。中文约 28%，代码约 37%。等待不是全部可消除：CPU 专家仍必须完成；这些数值定位关键路径，不是承诺的优化收益。

共享专家侧流的实测跨度分别为 1.16/1.06/0.88 ms/token（精确值见 `analysis.json`）；它与主流重叠，未再次加入上表。`wait_plan` 包含计划复制及时间戳节点间隔；`wait_cpu` 是等待结果的核与时间戳间隔，包含少量固定节点成本。当前默认路径没有把这两个等待融合进复制核。

热态长请求的专家文件读取为零，PCIe 专家权重分流为零。CPU↔GPU 仍交换激活、专家 ID、计划、结果和握手标记。**这里的大段等待是 CPU 结果/计划尚未就绪，不是把专家权重搬进 GPU 的耗时。** PLE 和输出头 embedding 的访问另属各自路径。

完整 CSV、按窗口/验证宽度/层的统计在 `mtp-live/lifecycle.csv`、`mtp-live/window-summary.json` 和 `analysis.json`。

## 3. CPU Q4_0 的具体实现缺口

`src/kernels/cpu/native_expert.cpp` 中 Q4_0（类型 2）不属于专用 IQ/KQ/NVFP4 多 token 内核。gate/up 和 down 最终都进入 `for (row) -> for (token) -> vec_dot(..., 1)` 的通用路径。同一专家被多个候选使用时，点积及权重解码分别执行，未融合跨 token 的解码和计算。缓存可能复用权重字节，因此不能将其表述为每个 token 都重新从 DRAM 读取整份权重。

本轮使用引擎现有 `kq_cpu_bench`，调用实际 `native_gu_rows`/`native_down_rows`，Q4_0/Q4_0，形状 H=2560、FF=640，约 2 GiB 权重，12 线程，三次中位数：

| 同一专家服务的 token 数 | 每份专家权重的有效处理率 | 每专家、每线程耗时 |
| --- | ---: | ---: |
| 1 | 74.4 GB/s | 0.446 ms |
| 2 | 42.4 GB/s | 0.783 ms |
| 4 | 21.4 GB/s | 1.549 ms |

独立 AVX-512 顺序读、2 GiB、相同 12 核的读取上限约 75.6 GB/s。一个 token 时专家核接近内存读取上限；四个 token 时耗时约为一个 token 的 3.47 倍，多候选无法充分摊薄此项成本。

上表“有效处理率”是**唯一专家权重字节/耗时**，不是 PMU 测出的 DRAM 总线流量。CPU 微基准使用有效随机量化权重、固定预量化激活，不包含生产池的逐层握手、任务划分和中间激活量化；不将微基准收益直接当作 TPS。

正常推理中，以实际 distinct CPU expert 数、每专家 2,764,800 字节和 CPU 运算时间折算，约 45–52 GB/s，与混合单/多 token 工作相符。源码和此项微测量共同支持优先补 Q4_0 的多 token SIMD 内核。

原始证据：`cpu-q4-bench.csv`（程序文本输出）、`read-bandwidth-node0.csv`、`read-bandwidth-node7.csv`、`read_bandwidth.cpp`。

## 4. DFlash2 的额外成本

正常原生 DFlash2 的完整 draft 阶段包括特征 append 和 forward，分别为英文/中文/代码约 20.05/20.05/20.22 ms/window，占对应完整 decode 约 23.8%/27.3%/19.2%。旧 `draft_ms` 只覆盖 forward，不含 append；此前按它估计约 17% 不完整。

原生诊断运行中，英文每次调用的 GPU 时间约为：

| 草稿阶段 | ms/call |
| --- | ---: |
| 接受特征投影、归一化与五层 K/V 注入 | 5.01 |
| 五层网络及最终归一化 | 12.94 |
| 草稿输出头 | 1.77 |
| selector 投影、top-k 与路径选择合计 | 0.38 |

五层中每层 attention 路径约 1.58 ms，FFN 路径约 1.00 ms。attention 路径包含归一化、动态卷积、Q/K/V/O 投影和 attention 核，不能把该值全称为 attention 核时间。网络经 `Gemm::bf16` 执行多个小行数矩阵乘；本轮没有再把每个 GEMM 与 attention 核逐个拆开，不能声称已经证明其中某个具体库 kernel 是唯一原因。

英文/中文/代码的完整特征处理与草稿主机时间，按实际输出 token 约为 10.95/11.04/4.75 ms。代码每窗口提交更多 token，摊薄了这笔成本。原生功能已通过，本轮数据仍不支持它比增强 MTP 普遍更快。

原始证据：`native-live/draft.csv` 和 `analysis.json`。

## 5. NUMA 与已排除的解释

HIP 设备 0 的真实 PCI BDF 为 `0000:b5:00.0`，NUMA 节点 7，链路报告 32 GT/s ×16；生产默认工作线程在节点 0。`numactl --cpunodebind=7 --preferred=7` 对照不改变专家数、权重预算或线程数，但三类结果有升有降，未形成一致收益。

`mtp-numa7/numa_maps.txt` 还显示该次约 41 GiB 专家区大量落在节点 4/6，仅少量在节点 7；preferred 不保证全部本地分配。因此此项不能宣称解决了 NUMA，也不能用它证明 NUMA 完全没有影响。没有据此修改默认绑核。

未观察到热态专家读盘等待、OOM 或专家权重搬迁占主导。512-token 测试不能替代长上下文结论。本轮没有重启已删除的预测器或调度策略。

## 6. 计时方法与验证边界

- 各 `provenance.json` 的 `source_sha256` 是运行时部署目录快照，不能单独当作二进制构建来源：`mtp-numa7`、`native-normal` 在临时诊断源码仍在场时执行的是原始二进制。二进制来源以 cleanup 的既有审计、下列原始 SHA，以及 `instrumentation.patch` / `diagnostic-build.sha256` 区分。
- 原始引擎 SHA-256：`745c912b923cec9696517e9a898609c1e4483fb4baca3b411f1f52c0dc8a154a`。正式软链接始终指向 cleanup 版本。
- 外部 `hipprof --hip-trace` 试验在开启追踪后导致带握手的图停滞，引擎 watchdog 释放等待并退出。保留 `mtp-hipprof/` 失败记录；未使用它产生任何性能结果。
- 旧 `STRATA_VERIFY_PROFILE` 会关闭若干融合和跨流路径，本轮诊断补丁保留这些选择，另加共享侧流时间戳。计数器使用此前验证的 25 MHz wall clock（每 tick 40 ns）。
- 时间戳、输出与收集有观察开销，也会影响自适应窗口选择。诊断 MTP 为 28.19/32.79/36.64 TPS；诊断原生为 20.72/21.98/32.10 TPS。MTP 英文平均 T 从正常的 1.70 变成 2.09，不能把 profile on/off 的 TPS 差直接当作纯插桩开销。
- `analyze.py` 按主机请求边界及 position 排除 prefill，仅计算实际 decode 窗口。两组诊断的 11 个请求逐一核对，CSV 窗口数全部与引擎统计一致；各窗口主流阶段之和与 GPU 总跨度一致。CPU 与共享侧流单列，避免重复相加。
- 诊断执行中的 OK/4/北京、采样请求、单 token、上下文边界以及 STOP/恢复检查通过。没有要求长文本逐 token 相同，也没有将短 QA 写成全面质量评测。
- 本轮仅验证 HIP/gfx936。未将诊断补丁作为新功能提交或推送。

## 7. 插桩与产物清单

| 文件 | 操作 | 内容 |
| --- | --- | --- |
| `run.py` | 新增 | 基于既有功能驱动，增加请求/输出时间、主机计时、外部追踪入口 |
| `instrumentation.patch` | 新增 | 诊断补丁，包含下两项改动，默认不开启 |
| `src/core/verify.cpp` | 仅远程诊断构建中临时修改，已还原 | `STRATA_LIFECYCLE_CSV` 保留融合/跨流，记录 GPU 分段及 CPU 逐层时间 |
| `src/prefill/dflash.cu` | 仅远程诊断构建中临时修改，已还原 | `STRATA_DFLASH_LIFECYCLE_CSV` 记录 append、五层网络、head、selector |
| `read_bandwidth.cpp` | 新增 | AVX-512、12 核顺序读上限；所有轮次校验和均正确 |
| `analyze.py` | 新增 | CSV 阶段归因、按请求/窗口宽度/层汇总，避免 CPU/GPU 重复计时 |
| 各运行目录、`analysis.json`、本文件 | 新增 | 命令、配置、回答、GPU 权重账单、原始日志、CSV 和分析 |

诊断构建存于持久盘 `/root/private_data/strata-dcu/lifecycle-20261023/strata-profile`。补丁已从远程生产源码反向应用；恢复后的两个源码 SHA 与本地当前提交相同。所有推理和 profiler 进程已退出，OOM 计数为零，`/root` 系统层约 954 MiB，本轮持久盘产物约 146 MiB。正常运行不依赖这些诊断文件。
