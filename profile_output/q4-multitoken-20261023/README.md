# Q4_0 CPU 与原生 DFlash2 投影优化 — 2026-10-23

实现和运行方式见 [内核文档](../../docs/DCU_KERNEL_OPTIMIZATIONS.md)。本目录保存本轮实际命令、模型输出、原始日志、权重账单、哈希及汇总。

- `mtp-off/`、`mtp-on/`：同一个 CPU 候选引擎，分别关闭/开启 `STRATA_CPU_Q4_MT`；三类请求各三次。
- `native-off/`、`native-both/`：原生 DFlash2 的旧路径与两项优化；三类请求各三次。`native-both` 另含完整 14 个 QA、198 tokens 及 STOP/恢复。
- `summary.json`、`analyze.py`：核验权重预算、固定专家数和请求数量，再计算中位数与范围。分析按完整用例名匹配，不将 `code-function` QA 混入代码吞吐统计。
- `cpu-*.txt`、`parity-*.log`、`ctest-*.log`：开发迭代与最终 CPU 实测。第一版两个块拼成 512-bit 的两候选性能回退，未采用；最终使用 256-bit EVEX/VNNI。
- `bf16-small-final.csv`：104 个 GPU 算子用例及 graph 内时间，最后一行必须为 `PASS BF16_SMALL_MM`。生产对完整词表头的三行以上采用 BLAS，不将未选中的慢内核算作收益。
- `witness/`、`long/`：真实权重缓存/图检查和 2100-token 长输入后恢复。
- `entry/`：按文档正常入口的独立执行结果。
- `audit.json`：最终二进制、源码、预算、进程退出及系统盘审计。

每个性能配置只有一次引擎启动，内部每类三个请求；范围不是独立硬件重复的置信区间。所有推理均为 BS=1，固定 8500（MTP）或 8300（DFlash2）专家，权重占用不超过显存一半。没有比较不同专家数来声明加速。

CPU 微基准的 GB/s 按每份专家权重字节计，不是硬件测出的 DRAM 总线带宽。随机量化权重测试和真实模型 TPS 分开记录；算子速度不能直接外推到整机。

完整构建日志和二进制保留于远程持久盘 `/root/private_data/strata-dcu/q4-multitoken-20261023/`。本地不保存模型或大二进制。
