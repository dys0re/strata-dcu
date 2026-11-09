# DCU 运行选项

适用范围为 Qwen3.8-F、BW/gfx936、单卡、BS=1。
通用引擎不会自动选择机器上的模型变体；wrapper 的默认值取决于部署配置。
目前验证 worker 选择 Q8 dense/head、HC Q8、stable MMAC、蒸馏 adaptive MTP/maxT5，
8500 GPU 专家、12 workers、IQ4_NL PLE、INT8 KV、context512。
CPU 队列优化开启，PCIe 分流关闭，全部 GPU 权重受半显存预算约束。

## 模型和草稿

| 选项 | 用途 |
|---|---|
| `--dense-gguf FILE` | 选择预先生成的非专家矩阵变体 |
| `--mtp-runtime DIR` | 选择准备并验证过的 MTP runtime |
| `--mtp-mode adaptive` | 自适应 MTP；与实际窗口上限一起记录 |
| `--dflash2 DIR` | 显式选择原生 DFlash2；不能搭配要求 BF16 共享头却已量化为 Q8 的配置 |
| `--suffix-draft N` | prompt lookup 最小匹配长度；当前机器入口为 0 |

lookup 草稿仍由目标模型验证；MTP/DFlash 缓存按实际提交状态更新。
增强 MTP 神经草稿最多验证五行，lookup 可扩展到七行；DFlash2/联合总上限为八行。
剩余输出长度和上下文会进一步限制窗口。当前通用任务未建立 lookup 的稳定收益。

模型准备见 [模型变体](DCU_MODEL_OPTIMIZATIONS.md)，草稿使用见
[原生 DFlash2](DFLASH2_NATIVE.md)和[联合模式](JOINT_DRAFT.md)。

## 内核和调度

| 环境变量 | 用途 |
|---|---|
| `STRATA_Q8_MMAC=1`、`STRATA_Q8_MMAC_STABLE=1` | 单份 Q8 MMAC 执行权重，固定 T1–5 累加策略 |
| `STRATA_HC_Q8=1` | 载入 HC Q8 权重，区别于 HC_REQ8 |
| `STRATA_HC_SPLIT=2` | gfx936 staged HC；0/1 分别选择 plain/split |
| `STRATA_CPU_Q4_MT=1`、`STRATA_CPU_Q4_INPUT_BIAS=1` | CPU Q4 多位置计算与输入修正复用 |
| `STRATA_VERIFY_HIT_PLAN=1` | GPU 独立安排常驻专家 |
| `STRATA_CPU_EXPERT_PIPELINE=1`、`STRATA_CPU_EARLY_RESULT=1` | 按专家推进，输出完成后提前通知 GPU |
| `STRATA_CPU_MISS_GRAPH=1`、`STRATA_FUSE_ROUTE_PUBLISH=1` | 无 PCIe 分流条件下的专用图和融合路由发布 |
| `STRATA_CPU_POOL_COST=1` | 长任务优先、按服务 token 数分配行块；依赖专家流水 |
| `STRATA_CPU_RESULT_DIRECT=1` | 显式实验：GPU 直接读取 CPU 输出；当前机器入口关闭 |
| `STRATA_POOL_IDLE_US=N` | 层末自旋后休眠；没有稳定吞吐收益，当前入口不启用 |

详见 [内核参考](DCU_KERNEL_OPTIMIZATIONS.md)、[MMAC](DCU_Q8_MMAC_PLAN.md)、
[层内交接](DCU_LAYER_SCHEDULING.md)、[专用图](DCU_GRAPH_SCHEDULING.md)和
[CPU 队列](DCU_HETERO_SCHEDULING.md)。环境值只表示请求，必须核对实际分发日志。

## 专家交换和 PCIe

`--adapt-every 0` 关闭交换，`--adapt-swaps` 控制每轮交换上限。
`--pcie-frac 0` 关闭普通分流，`-1` 请求自动策略；`--pcie-mode auto|kernel|dma|direct`
选择传输方式。`--resident-pin` 只请求锁页映射，成功与否要检查日志。
`STRATA_PCIE_STAGED_MAX` 使用小型锁页中转区，不要求锁页整个 RAM 专家副本。

当前机器入口保留 CPU 队列优化并关闭 PCIe 分流，未删除上游通用 PCIe 支持。
不要把启动 DMA 带宽、请求的非零比例或自动策略开关当成实际搬运的证据；应记录真实 PCIe 专家数。

逐层 trace/profile 用于诊断，会影响耗时；正式 TPS 使用不带这些记录的运行。
完整测量要求见 [测试标准](DCU_TEST_STANDARD.md)。
