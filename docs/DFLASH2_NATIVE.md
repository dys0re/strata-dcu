# Qwen3.8-F 原生 DFlash2

`--dflash2` 现在可在 Strata 内执行已训练的 Flash-Next DFlash2，不需要独立 Python 草稿进程。Python 入口只负责准备文件、分词和发请求。未开启 DFlash2 时，AutoRound 入口继续使用增强 MTP。

## 执行流程

使用 `incoai/Qwen3.8-Flash-Next-DFlash2` 的固定版本，原始权重 SHA-256 为 `d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`。`tools/dflash2_pack.py` 检查配置、大小和源 SHA，按张量原样保存 BF16 数据及 manifest；不量化、不改主模型、不复制 embedding 或输出头。派生文件必须保存在持久盘。

主模型完成的第 `[5,15,25,35,45]` 层四路 HC 残差在 GPU 上取均值，拼接为 12800 维特征。草稿的输入投影、归一化、逐层 K/V 注入均在 GPU 执行。短提示词经过 verifier 的路径和批量 prefill 路径都提供这些特征，不经过主机或 socket。

原生网络包含完整的五层 GQA Transformer、attention/MLP 两侧的动态因果卷积、SwiGLU、RoPE、BF16 输出投影和 top-16 候选选择器。草稿直接借用目标模型已有的 BF16 输出头和主机映射 embedding。每个已知 anchor 后最多起草七个候选，主模型的验证宽度为 2–8。

每层草稿 K/V 使用 2056 行环形缓冲区：2048 的滑动窗口加最多八行临时候选。位置采用绝对位置，attention 只访问当前块和滑动窗口内的已提交上下文。新的已提交特征覆盖相应位置，拒绝的 noise 行不会成为后续上下文。每个新请求逻辑清空缓存；主模型 GDN、QSA、卷积和 PLE 状态仍由原验证器提交、回滚。

为 1–7 个候选分别捕获 GPU graph，共七个。上下文位置和 anchor 为动态设备输入，图不依赖 512 的固定上下文长度。`--dflash-eager` 可关闭图执行。`--dflash-adaptive` 根据已经完成轮次的提交 token 数和墙钟耗时选择下一轮候选数；没有使用 DSpark confidence head。

主模型仍按请求采样参数和位置 RNG 做精确匹配验证。草稿本身使用 greedy 候选。尚未接入基于草稿概率的 p/q 拒绝采样。

## 使用

已准备 AutoRound 环境的远程机器可直接运行：

```bash
ulimit -c 0
bash /root/strata-dcu/run-autoround.sh \
  --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint \
  --prompt 'Reply with exactly OK.' --max-new 16
```

该入口默认选择 native、graph、自适应窗口、最大验证宽度 8，主模型 GPU 专家固定为 8300，不扫描专家数。`--dflash-fixed` 固定使用最大窗口；`--dflash-max-t 4` 可限制为一个 anchor 加三个候选。

自行准备原生草稿包：

```bash
python3 tools/dflash2_pack.py \
  --checkpoint /root/private_data/strata-dcu/dflash2/checkpoint \
  --out /root/private_data/strata-dcu/dflash-native/pack
```

原生引擎在已有主模型参数后添加：

```text
--dflash2 /root/private_data/strata-dcu/dflash-native/pack
--dflash-adaptive --spec 8 --mtp-max-t 8
--serve --prompt-cache 0 --conversation-cache-mib 0
--suffix-draft 0 --adapt-every 0 --pcie-frac 0 --adapt-swaps 0
```

不得同时开启 MTP、lookup drafting、批量请求、窗口流水或输出头借用。当前原生草稿只接受该固定 checkpoint 和 BF16 目标 head/embedding。位置上限为 checkpoint 的 262144；实际验证过的上下文范围以测试报告为准，不能将接口上限当作已测试上限。HIP/gfx936 是运行验证平台。

旧的独立 PyTorch 路径保留为显式诊断入口：`--dflash-backend ipc --dflash-max-t 4`，其上下文仍限制为 512，见 [IPC 实现](DFLASH2_IPC.md)。

## 验收与权重

验收以端到端 QA 为主：问答答案、JSON、生成代码的执行、多轮消息、固定种子采样、取消恢复、边界和最终目标输出。草稿自身与 PyTorch 的隐藏状态距离及候选差异单独记录，不要求两个后端的草稿逐位相同，也不据此声称算子数值等价。

在固定 8300 GPU 专家下，原生草稿 GPU 参数为 1206131200 字节，复用已有 1271398400 字节目标头。全部 GPU 权重包含已有的 16 个临时专家槽（44,236,800 字节），合计 32827238400 字节，占物理显存 47.781584%，满足 [50% 标准](DCU_TEST_STANDARD.md)。相对 IPC 路径省去 1271398400 字节的重复输出头；未将省下的空间用于增加专家数。KV、工作区、图和激活单独计量，PLE 保留在磁盘。

功能 QA 驱动是 `tools/bench_dflash2_native.py --qa-only`；完整候选、增量缓存、重置、环形缓存和 graph/eager 的独立诊断使用 `tools/test_dflash2_native.py`。测试数据保存在远程 `/root/private_data/strata-dcu/dflash-native/`，小型结果归档于 `profile_output/dflash-native/`。
