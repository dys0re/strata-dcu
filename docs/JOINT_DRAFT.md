# MTP 与原生 DFlash2 联合草稿

这是显式启用的联合路径。默认仍为增强 MTP；只传 `--dflash2` 仍为独立 DFlash2。

原来两条路径各自产生完整候选窗口，没有合并候选和同步缓存的逻辑，所以入口禁止同时开启。联合路径让 MTP 产生短前缀，把前缀 token 的真实 embedding 送入 DFlash2 的对应位置，余下位置继续使用 mask。DFlash2 的 selector 固定这些前缀 ID，再根据前缀选择后续候选。主模型验证整个窗口，只有接受的 token 推进两个草稿模型的缓存。

## 使用

SCNet 主机已有目标权重、MTP 与原生 DFlash2 pack。在 GPU 空闲时运行：

```bash
ulimit -c 0
bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --dflash-mtp-prefix 2 --prompt 'Reply with exactly OK.' --max-new 16
```

`--dflash-mtp-prefix N` 允许 1–3 个 MTP 前缀候选；0 关闭联合路径。必须为原生 DFlash2、单请求服务模式，并为 DFlash2 后续候选保留至少一个位置。IPC、批处理、流水线和提示缓存不支持此组合。提示查找现可通过 `--suffix-draft 3` 开启，见 [DCU 运行选项](DCU_RUNTIME_OPTIONS.md)。接近输出或上下文上限时，候选缩短到剩余空间。

`--mtp-mode adaptive` 调整前缀长度，DFlash2 的默认 adaptive 选项调整整个候选数；联合模式使用一个控制器，根据已完成窗口的实际输出 token 数和总耗时选择组合。总耗时包括目标特征追加、MTP、DFlash2 和验证。使用 `--mtp-mode fixed --dflash-fixed` 固定两者。它不调整专家数或专家分流。

普通贪心请求保留主模型验证。开启 `STRATA_SPEC_PROB=1` 时，采样请求对 MTP 前缀使用其真实 q 分布，对确定性的 DFlash2 后续候选使用点质量分布。混合分布使用现有统一验证器；没有跳过后续候选的验证。

`generation.json` 应显示 `draft_method=mtp+dflash2`、`joint_mtp_prefix=2`、原生后端及两项 CPU/GPU 内核选项。引擎日志 `strata joint draft:` 分别报告两种来源的候选数、接受数和耗时，`joint budget:` 报告实际尝试过的前缀/候选组合。

## 验证记录

测试配置为 Qwen3.8-F、BW/gfx936、BS=1、8300 个固定 GPU 专家、12 workers、PLE direct IO。全部 GPU 模型权重继续受物理显存 50% 上限约束。没有恢复专家路由预测器或此前删除的调度策略。

本轮原始材料保存在持久盘 `/root/private_data/strata-dcu/joint-draft-20261023/`；可提交的小文件及构建、QA、缓存和性能记录位于 `profile_output/joint-draft-20261023/`。

联合路径已通过 14 项语义/格式/代码执行/采样 QA、STOP 与恢复、强制拒绝后的同组 QA、15 项 eager/graph 缓存检查，以及 2100-token 输入与后续请求恢复。旧独立 DFlash2 的九个缓存用例与之前保存的输出逐字节一致。测试是有限的功能验收，不代表完整质量评测。

每类三次 256-token 生成的中位数，联合模式为英文 **23.78**、中文 **27.85**、代码 **35.28 TPS**。本日之前的增强 MTP 为 31.59/38.88/40.29，独立 DFlash2 为 29.89/33.19/43.58。联合模式本轮更慢，没有改成默认。前两条基线使用本日较早的二进制；增强 MTP 固定 8500 个专家，独立 DFlash2 和联合模式固定 8300 个，不作同一二进制或统计显著性的宣称。详情见 [本轮结果](../profile_output/joint-draft-20261023/README.md)。

当前引擎为持久盘 `/root/private_data/strata-dcu/joint-draft-20261023/strata-final`，SHA-256 `1551c2b1cc1624a29ab9b17f2276599c2a35648c58babc417e2c6ad98947b1b1`。全部 GPU 权重为 **34,195,377,152 字节 / 49.772975%**，包含两个草稿模型及已有临时专家槽。HIP/gfx936 完整引擎运行及修改的 DFlash CUDA 翻译单元编译通过；完整 CUDA、SYCL、Metal 引擎未运行验证。
