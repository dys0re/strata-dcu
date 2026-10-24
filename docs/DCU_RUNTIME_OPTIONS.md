# DCU 原生选项与 prompt lookup

本机 Qwen3.8-F、BS=1 的入口为 `/root/strata-dcu/run-autoround.sh`。
全部 GPU 模型权重仍限制在物理显存的 50% 以内；增强 MTP 固定 8500 个 GPU 专家，
DFlash2 和联合模式固定 8300 个。PLE 继续从持久化磁盘读取。

本轮修复了 HC 两个归一化函数中的隐式浮点乘加合并差异，保留严格自检。
原先 split/staged 因数值差异回退到 plain；修复后两者在 gfx936 上通过逐位检查。
实测 staged 有收益，入口选择 `STRATA_HC_SPLIT=2`；可设为 0 或 1 复现 plain 或 split。
这项修复仅约束 DTK wave64 兼容构建中的两个函数，没有全局关闭 FMA。

入口与基准脚本均提供以下选项：

| 选项 | 默认 | 含义 |
|---|---:|---|
| `--adapt-every` | 0 | 每隔多少验证窗口交换缓存专家；0 关闭，上游值为 4 |
| `--adapt-swaps` | 96 | 每轮交换上限；不增加 GPU 专家槽数 |
| `--pcie-frac` | 0 | 未命中专家的 GPU 分流比例；-1 使用引擎自动策略 |
| `--pcie-mode` | auto | auto/kernel 使用 GPU 拷贝 kernel；dma 使用拷贝引擎；direct 直接读取映射内存 |
| `--resident-pin` / `--no-resident-pin` | 自动 | 显式非零 pcie-frac 时尝试锁页；分流实际生效还需要锁页映射成功 |
| `--suffix-draft` | 0 | prompt lookup 的最小匹配长度；3 为上游值 |

本机对缓存交换和 GPU 拷贝分流的测量均出现退步，因此保持可配置、默认关闭。
2026-10-24 增加的 [层内交接选项](DCU_LAYER_SCHEDULING.md) 在本机增强 MTP 入口默认开启
GPU 自行安排常驻专家、CPU 按专家推进和结果提前交回。原生 DFlash2/联合模式仍为显式选择。
`STRATA_PCIE_STAGED_MAX=2` 使用约 44 MB 锁页中转区，可不锁页整个 RAM 专家副本；
它实际使用 DMA 并覆盖普通 pcie-mode。此分流及层末休眠仍默认关闭。
`pcie-frac=-1` 本身不能证明发生了搬运；应检查实际 PCIe 专家数。
48.5 GB/s 左右的启动探测值来自 DMA，不能作为 GPU 拷贝 kernel 的速度。
`direct` 仅暴露原有接口，本轮没有测试，不作为推荐配置。

开启 prompt lookup：

```bash
bash /root/strata-dcu/run-autoround.sh --suffix-draft 3 --prompt 'Reply with exactly OK.' --max-new 16
bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --suffix-draft 3 --prompt 'Reply with exactly OK.' --max-new 16
bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --dflash-mtp-prefix 2 --suffix-draft 3 --prompt 'Reply with exactly OK.' --max-new 16
```

lookup 替代神经网络草稿时，目标模型照常验证。MTP 和 DFlash 的缓存按实际提交的目标状态补齐，
被替代的草稿不参与各自接受率的更新。联合模式把整个 lookup 窗口作为查找草稿验证，
不会把其中一段误当成 MTP 前缀。IPC 后端同步增加了丢弃反馈命令，须使用同版本 worker。

增强 MTP 开启 lookup 时，MTP 自身仍最多验证 5 行；lookup 可以扩展到 7 行。
DFlash2 与联合模式总上限仍为 8 行。剩余输出长度和上下文会进一步限制窗口。
本轮通用三项任务没有观察到 suffix 的稳定净收益，故默认关闭。

性能与完整验证记录见 [本轮记录](../profile_output/lookup-compat-20261023/README.md)
和 [原生选项测量](../profile_output/upstream-options-20261023/README.md)。


2026-10-24 的 [模型权重优化](DCU_MODEL_OPTIMIZATIONS.md) 已加入本机增强 MTP 默认入口：
HC Q8、Q8 非专家投影/输出头、发布的蒸馏 MTP。草稿筛选仍默认关闭。
`--dense-gguf FILE`、`--mtp-runtime DIR` 显式选择准备好的变体；原生 DFlash2 的共享输出头须保留 BF16。
`STRATA_DCU_MODEL_DEFAULTS=0` 恢复此前权重选择，原有层内交接默认仍保留。
配置路径由 `STRATA_DCU_MODEL_CONFIG` 指定，当前文件在持久盘 `model-specific-20261024/selected-model.json`。
