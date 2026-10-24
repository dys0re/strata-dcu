# 上游缓存交换与 PCIe 分流实测

2026-10-23，固定增强 MTP、8500 个 GPU 专家，全部 GPU 权重 48.822253%。
原模型、量化和磁盘 PLE 不变。每组新进程，预热后英文故事、中文解释、Python LRU
各三次、最多 256 tokens，随后运行 14 项功能 QA 和取消恢复。

## 已完成的三次吞吐中位数（token/s）

| 组 | RAM | 缓存交换 | PCIe 分流 | 英文 | 中文 | 代码 |
|---|---|---|---|---:|---:|---:|
| baseline-clean | 普通 | 关 | 关 | **31.74** | **38.74** | **40.75** |
| cache | 普通 | 4 窗口/96 上限 | 关 | 28.50 | 29.71 | 36.75 |
| cache-pinned | 锁页 | 4 窗口/96 上限 | 关 | 13.79 | 13.72 | 20.27 |
| pcie-pinned | 锁页 | 关 | 自动比例，GPU copy kernel | 8.81 | 16.85 | 12.68 |
| both | 锁页 | 4 窗口/96 上限 | 自动比例，GPU copy kernel | 13.65 | 10.47 | 11.92 |
| dma（10-24 重启后） | 锁页 | 关 | 自动比例，DMA | 12.25 | 12.58 | 22.62 |

以上完整 QA 均通过。实际回答已抽查；同种子输出不同没有作为失败条件。
交换保持 GPU 槽数不变，只替换其中的专家身份；没有恢复之前删除的预测调度器。
CPU 专家工作量下降，但交换等待及 GPU 等待增大，未形成净收益。
部分锁页配置的长输入 prefill 也明显变慢。尚未把底层原因归结为单一带宽或驱动因素。

初始 `baseline/` 为 31.76 / 39.32 / 41.16。因编译与其启动阶段重叠，
另跑无编译争用的 `baseline-clean/` 作为正式对照；保留两份数据。

## 两个容易混淆的前提

初始 `pcie/` 虽然设置 `pcie-frac=-1`，实际 PCIe 专家数始终为零。
普通 `--resident-cpu-experts` 不锁页，现有 planner 要求专家具有锁页映射，
所以该组只是路径未生效的诊断，32.29 / 39.95 / 42.46 TPS 不代表分流收益。

`pcie-mode=auto` 当前固定选择 GPU copy kernel，并非 DMA。
启动探测器报告的约 48.5 GB/s 属于 DMA，不能替代 `fetch_blobs_kernel` 的测量。
`pcie_experts` 是实际分流专家数，不是 DMA 调用数。
已生效的锁页组确认 41.39 GiB 专家 RAM 全部锁页，PCIe 专家计数非零。
观测链路为 32 GT/s ×16；一次运行中的 GPU 温度约 57°C，OOM 计数未增加。

入口现在暴露 `--adapt-every`、`--adapt-swaps`、`--pcie-frac`、
`--resident-pin` / `--no-resident-pin` 和 `--pcie-mode auto|kernel|dma|direct`。
显式非零 pcie-frac 默认尝试锁页，仍需核对实际计数。
DMA 补测由重启后的 `resume_20261024.py` 顺序执行，smoke 和完整 QA 均通过，实际 PCIe 计数非零。
DMA 可以运行，没有出现测试超时，但三项中位数均低于关闭分流的基线，默认仍不启用。
DMA 的数据来自重启后，未用它与重启前 GPU copy kernel 的差值宣称稳定加速。
direct 未测试，不作为推荐配置。

原四组执行脚本为 `run_matrix.py`，后续补测见相邻目录的执行脚本与日志。
原引擎 SHA256：`1551c2b1cc1624a29ab9b17f2276599c2a35648c58babc417e2c6ad98947b1b1`。
全部二进制、模型和大文件位于远程 `/root/private_data/strata-dcu/`。
