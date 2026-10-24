# DCU 层内交接与 CPU 专家执行

这组开关用于 Qwen3.8-F、单 GPU、单请求的实际执行链。常驻专家数不变，
全部 GPU 模型权重仍不超过物理显存的 50%；PLE 继续读取持久化磁盘。
引擎原默认仍关闭这些选项。本机 `run-autoround.sh` 的增强 MTP 入口默认开启前三项；
原生 DFlash2、联合模式以及其他入口仍显式选择。分流与休眠默认关闭。性能选择和完整证据记录在
[本轮记录](../profile_output/layer-scheduling-20261024/README.md)。

| 环境变量 | 含义 |
|---|---|
| `STRATA_VERIFY_HIT_PLAN=1` | GPU 根据 router 的真实结果自行安排显存专家，在 CPU 工作安排到达前启动它们 |
| `STRATA_CPU_EXPERT_PIPELINE=1` | 一个专家的 gate/up 行完成后即可量化并发布其 down 行任务，不等整批专家全部完成 gate/up |
| `STRATA_CPU_EARLY_RESULT=1` | 输出齐全后先通知 GPU，再等待工作线程回到待命位置和完成主机侧统计 |
| `STRATA_PCIE_STAGED_MAX=2` | 按实测成本决定少量 miss 分流，每个层内 token 组最多搬两个专家 |
| `STRATA_POOL_IDLE_US=500` | 一层专家计算结束后，工作线程最多自旋约 500 微秒再休眠；同层两个旧计算阶段之间仍用原等待设置 |

`STRATA_POOL_IDLE_US` 的时钟检查有间隔，因此不是严格的微秒截止时间。
休眠使用已有的通知协议；主控线程仍轮询 GPU，不能把工作线程占用下降解释为主控占用也下降。
`STRATA_POOL_SPIN_US` 保留原来对所有待命阶段的设置。

## GPU 先计算显存专家

原路径由 CPU 同时整理显存专家和分流专家的工作安排。新路径给显存专家分配独立 GPU 工作表，
保持专家分组顺序、输入行和输出行顺序。CPU 不再重复生成显存专家描述，
只记录分流专家；GPU 等到这部分描述后，将它接在自己生成的显存专家描述后面。

显存专家的参数不搬运，也不替换。路由不改、不丢专家、不预测未来路由。
设备表可能暂时落后于主机表的 `always_publish` 路径会关闭此选项；异步专家交换 gate 不与它并用。
使用额外 swap-space 专家的 Foresight 配置继续走原工作安排。当前测试范围为本机单 GPU。

## CPU 按专家推进

保留每个专家按行拆分的并行度。所有 gate/up 行任务先入队；一个专家最后一份 gate/up 任务完成后，
由完成它的线程量化中间结果，并把这个专家的 down 行任务加入同一个队列。
因此 down 可以与尚未结束的其他专家 gate/up 重叠。队列按先进先出执行，并非优先抢占 gate/up。

批次完成后显式关闭原任务领取计数，晚醒线程必须先检查批次编号及是否结束，才能读本批任务。
没有这个检查，晚醒线程可能在主线程切回其他执行方式后领取已经结束的任务。
测试用 `STRATA_POOL_TEST_DELAY_US=2000` 人为制造晚醒；正常运行不设置它。

结果提前发布使用全部任务完成后的 acquire，再向 GPU 发布完成信号。
缓冲区复用仍等工作线程全部回到待命位置。第一层保留 PLE 数据收集 gate；
存在其他 GPU 专家来源或文件读取分批时，不提前发布。
CPU 的专家查找清理及路由记录在发布前处理，防止下一层覆盖相同的映射 ID 缓冲。

## 小批 DMA 分流

复用原有 16 个 GPU 临时专家槽位，不增加常驻专家数。这些槽位已计入权重账单。
只申请约 44 MB 的锁页主机中转区，不把四十多 GB 的整个专家 RAM 副本全部锁页。
CPU 将选中专家复制到中转区，DMA 搬到 GPU；这次 CPU 复制的成本同样包含在测量中。
该模式实际使用 DMA，覆盖普通 `--pcie-mode` 的选择。

使用已完成层的 CPU 和 GPU 时间估计两边完成时间。不同 token 组宽度分别拟合，
避免把单位置与多个候选位置的成本混用；在数量相同时优先搬当前组中复用次数多的专家。
只考虑 router 已经给出的真实名单。少量探测用于补充搬运成本样本，
每 64 次有 miss 的选择可能探测一个专家，因此不是零探索的最优调度器。

## 验证

底层检查为 `resident_plan_parity` 和 `pool_tasks_test`。
后者覆盖原路径、专家流水执行、立即休眠、延迟醒来、12 个工作线程、变化的批次大小和 1–8 个位置，
并检查提前通知发生时所有输出已经正确。相应 CTest 项为 `pool_tasks_test`、
`pool_expert_pipeline`、`pool_expert_pipeline_sleep`。

真实模型的回答、取消恢复、边界及吞吐分别记录。逐线程占用使用可选的
`tools/bench_dflash2_native.py --thread-usage`，读取 Linux `/proc` 的线程运行计数。
该占用包含忙等，不是有效计算时间。新功能的验证不能替代其他 GPU 平台的运行检查。

2026-10-24 本机正常测速，每类三次 256-token 生成的中位数：

| 配置 | 英文 TPS | 中文 TPS | 代码 TPS |
|---|---:|---:|---:|
| a6b66f29 原二进制 | 33.27 | 41.23 | 40.63 |
| 前三项一起开启 | 34.65 | 42.33 | 45.03 |
| 再加层末 500 微秒休眠 | 33.51 | 42.43 | 44.44 |

休眠把 12 个工作线程的平均占用中位数从约 100% 降到 79.53%，主控线程仍接近 100%。
它没有提升吞吐，因此不默认开启。小批搬运也未体现稳定收益，保留显式选项。
增强 MTP、强制搬运、原生 DFlash2、联合草稿强制拒绝的功能 QA、取消恢复均通过；
公开入口三种模式由独立会话原样执行一次通过。没有新增 IPC 运行验证。

默认入口直接运行即可。复现修改前的三项关闭状态，或选择休眠：

```bash
STRATA_DCU_SCHEDULING_DEFAULTS=0 bash /root/strata-dcu/run-autoround.sh --prompt 'Reply with exactly OK.' --max-new 16
STRATA_POOL_IDLE_US=500 bash /root/strata-dcu/run-autoround.sh --prompt 'Reply with exactly OK.' --max-new 16
```

`STRATA_VERIFY_TRACE=1 STRATA_VERIFY_TRACE_FULL=1 STRATA_POOL_TRACE=1` 可记录每个窗口全部 48 层。
`STRATA_VERIFY_PROFILE` 不参与这组记录，避免它改变原有融合和流选择。
记录本身会增加开销；CPU 和 GPU 时间轴各自独立，不能跨轴相减，也不能用这些运行计算正式 TPS。
正常运行不设置这些记录开关。
