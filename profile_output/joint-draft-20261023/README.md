# MTP + native DFlash2 联合路径验收

2026-10-23，BW/gfx936，Qwen3.8-F AutoRound，BS=1。联合路径实现为 MTP 短前缀 → 前缀条件化 DFlash2 后续候选 → 主模型统一验证。不是在两种独立草稿中二选一。未修改原权重、量化、PLE 存放和专家调度。

入口及限制见 [使用文档](../../docs/JOINT_DRAFT.md)。默认仍为增强 MTP；显式 `--dflash-mtp-prefix 2` 才启用组合。候选预算联合调整前缀长度及候选总数，测量成本包含两阶段草稿、特征追加和主模型验证。

## 功能

- `qa-prob-v2/`：14 项回答、数学、JSON、Python 函数执行、多轮、采样、单 token 和上下文尾部，共 198 输出 tokens；另测 STOP 与恢复。全部通过。60 个窗口实际使用概率验证，MTP 前缀用 q 分布，DFlash2 后续候选用点质量分布。采样复现也通过，但长文本逐 token 一致不作为质量验收条件。
- `qa-reject/`：同组 14 项 QA，以及 STOP 与恢复，通过。人为修改每一个候选；217 个 MTP 和 397 个 DFlash2 候选均未被接受，验证了连续拒绝后的追赶和重置。
- `witness/`：真实权重、固定随机特征，15 项缓存/前缀用例，eager 和 graph 通过且输出逐字节相同。覆盖前缀确实改变网络状态、前缀 ID 不被覆盖、重复、重置、增量/重算、短候选、完整七 token 前缀、2048 窗口边界和环形覆写。两个非法前缀在 embedding 访问前被拒绝。
- `legacy-cache/`：原独立 DFlash2 九项真实权重缓存用例，两种执行模式均与本日上一版保存的 hidden/IDs 逐字节一致。
- `long/`：4096 上下文、2100-token 输入正确返回 ORBIT，下一请求返回 OK。
- `ctest.log`：概率验证器混合 q 前缀/确定性后续候选的独立参考检查、原 MTP 预算和新增联合预算，3/3 通过。联合预算单元覆盖 36,000 个决策及五种非法观察；其中成本数据是控制器合成测试，不是推理测速。

以上是有限的功能验收，没有声称完整任务质量不变。完整 HIP 引擎构建并运行。修改的 `dflash.cu` 用 CUDA 13.3、g++-15、sm_80 编译通过；未验证完整 CUDA、SYCL、Metal 引擎。

## 速度

`bench/`：预热 128 tokens，然后英文、中文、代码各三次 256-token 请求，同一次引擎启动。512 上下文、INT8 KV、12 workers、PLE direct IO；CPU Q4 多 token 与 GPU 小投影优化均开，专家交换和 PCIe 专家分流关闭。只开主机计时，没有 GPU 逐层 profiling。记录见 `results.json`、`command.json` 和 `summary.json`。

| 路径 | GPU 专家数 | 英文 TPS | 中文 TPS | 代码 TPS |
| --- | ---: | ---: | ---: | ---: |
| 本日较早的增强 MTP | 8500 | 31.59 | 38.88 | 40.29 |
| 本日较早的独立 DFlash2 | 8300 | 29.89 | 33.19 | 43.58 |
| 联合 MTP + DFlash2 | 8300 | 23.78 | 27.85 | 35.28 |

基线来自 [前一轮测量](../q4-multitoken-20261023/README.md)，与本轮使用不同冻结二进制；不宣称同一二进制对照或统计显著性。每条路径只有一次引擎启动。三类任务均未观察到联合收益，因此保留为显式选项。

联合测速含预热共 2432 输出 tokens，接受了 846 个 MTP 和 539 个 DFlash2 候选，证明两条来源实际参与生成。累计特征追加约 498 ms、MTP 2101 ms、DFlash2 5073 ms；这些是整个测试的累计主机时间，不是单 token 耗时。候选计数包含前部拒绝后未到达的后续候选，不能直接解释为预测准确率。联合路径新增串行草稿成本，主模型验证仍占主要耗时；此测试没有把吞吐下降全部归因到某一项。

全部 GPU 模型权重 **34,195,377,152 字节 / 49.772975%**，上限 **34,351,349,760**。含 native draft 1,206,131,200、MTP 1,368,138,752，以及 staging 44,236,800 字节。GPU 固定 8300 个专家，其余 16276 个在 RAM；PLE 保持磁盘。工作区不保存大权重、构建或二进制。

## 构建与复核

正式二进制位于 `/root/private_data/strata-dcu/joint-draft-20261023/strata-final`，SHA-256 `1551c2b1cc1624a29ab9b17f2276599c2a35648c58babc417e2c6ad98947b1b1`。`build-provenance.json` 记录实际源码及 witness 哈希，`compile-commands.txt` 记录实际 HIP/gfx936 编译命令。QA 的 `strata-prob` 与测速的 `strata-final` 内容相同。

`analyze.py` 读取真实结果并检查权重账单、专家数、两来源接受计数及缓存验收。执行 `python3 profile_output/joint-draft-20261023/analyze.py` 应打印 `PASS JOINT_RESULT_AUDIT`。`check_legacy.py` 与 `capture_provenance.py` 是在远程实际执行的复核脚本。

完整日志与大文件保存在远程持久盘同名目录。测试关闭 core dump，无依赖安装或升级。标准入口的独立复验记录存于 `entry/`。
