# Qwen3.8-F 的 DFlash2 实验接入

这是供单请求验证的 Linux IPC 实现。主模型、CPU 专家、PLE、采样和拒绝回滚仍由 Strata 执行；独立 PyTorch 进程运行已训练的 DFlash2，并通过 socket 返回候选。功能由 `tools/strata_dflash2.py` 显式启用，原启动入口继续使用增强 MTP。

权重使用 [Inco AI 的 Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)，HF revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6`。ModelScope 同名仓库的权重 SHA-256 为 `d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`，与 HF 发布值一致。经 ModelScope 下载并校验（HF Mirror 大文件传输过慢后切换），使用 `tools/download_dflash2.py`，缓存均位于远程 `/root/private_data/strata-dcu/dflash2/`。下载依次支持 HF Mirror、ModelScope；网络异常日志隐藏可能包含代理认证的异常字符串。

## 特征与状态

按发布者的 Flash-Next 补丁，提取完成的 0-based 第 `[5,15,25,35,45]` 层的四路 HyperConnection 残差均值，拼为 12,800 维特征。验证器的 FFN 写入延迟到下一层的 GR read，因此在该写入完成后抓取前一层状态。prefill 批处理和通过验证窗口读取短提示词两条路径都提供特征。

仅把实际提交的 token 对应的特征交给草稿端；被拒绝的候选行不进入草稿上下文。草稿 K/V 只保留已提交的主模型上下文，候选自身的 K/V 在本轮起草后裁掉。每个新请求清空草稿状态，首个输出 token 来自主模型。候选由原验证器检查并提交，GDN、卷积、PLE 历史和 QSA 的回滚由既有逻辑处理。

草稿端使用 `third_party/dflash/model.py` 的原始参考实现，固定在 upstream revision `07ebd93db9f472af339b644bb70221ad8428328a`，保留 MIT 许可证。适配器只负责权重绑定、上下文缓存和通信。候选采用确定性 greedy proposal；主模型继续用请求自身的采样参数和逐位置 RNG 做精确匹配验证。没有实现基于完整草稿概率分布的拒绝采样加速。

## 显存与支持范围

执行 [GPU 全部权重不超过 50% 的标准](DCU_TEST_STANDARD.md)。C++ 启动时接收草稿端实际参数字节数，把外部进程的权重副本一并计入预算，在开始生成前核验。

首版草稿权重占 1,206,131,200 字节，草稿进程另持有 1,271,398,400 字节的 BF16 输出头，共 2,477,529,600 字节。embedding 留在主机，按需读取输入行。主模型默认试验配置为 8,300 个 GPU 路由专家，16,276 个 RAM 专家。合计 GPU 权重 34,054,400,000 字节，占物理显存 49.5678%，低于 34,351,349,760 字节上限。

当前只支持：Linux、单 GPU、单请求、文本、最多 512 上下文、不复用前缀/会话、不与原生 MTP、lookup drafting 或窗口流水并用。试验块长为 4，即一个已知 anchor 加三个候选。未验证其他硬件、其他量化、长上下文或完整任务质量。

特征通过 GPU→CPU→进程间传输→GPU 交给草稿端，使用 PyTorch SDPA 的现有可用后端；这部分耗时计入 decode。该实现用于确认真实权重、接受率和端到端行为，性能以实际测量为准。

## HIP Graph 草稿执行

`tools/strata_dflash2.py --graph` 为每个增量上下文长度 1–4 和候选数 1–3 捕获一个图，共 12 个图。K/V 保存在地址稳定的 520-row 缓冲区，按绝对位置写入实际提交的上下文及当前噪声块；动态 mask 屏蔽所有未用行。新的噪声块覆盖上一次候选行。首轮较长上下文直接走相同的固定缓存前向。

图中包含草稿五层、输出头、top-k 和候选选择。实际输入和位置写入固定 GPU 缓冲区后 replay；输出仍由 Strata 验证。种子 forward 检查覆盖重复运行和增量缓存与完整重算的一致性。运行耗时按整轮记录，包含进程通信、特征拷贝和草稿执行。

`--adaptive` 根据已完成轮次的实际提交 token 数和 draft＋verify 墙钟耗时，在 1–3 个候选之间选择下一轮长度，定期重新探测。选窗不查看尚未生成的候选；每次请求重置任务统计。该策略直接复用 IPC 返回可变候选数的接口。

## 运行

远程需先 `source /opt/dtk/env.sh` 和 `source /opt/hyhal/env.sh`，设置 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false`。无需更换系统 PyTorch 或安装其他框架。

先验证草稿算子、权重和缓存：

```bash
python3 /root/strata-dcu/Strata/tools/dflash2_worker.py \
  --checkpoint /root/private_data/strata-dcu/dflash2/checkpoint \
  --target /root/private_data/Qwen3.8-F \
  --out /root/private_data/strata-dcu/dflash2/witness
```

复现完整请求时使用 `profile_output/dflash2/benchmark.py`，其 `command.json` 保存实际的 wrapper 和引擎参数。原生接口可直接执行：

```text
python3 tools/strata_dflash2.py --checkpoint DRAFT_DIR --model TARGET_DIR --stats-file WORKER_JSON --graph --adaptive -- STRATA_BINARY <已准备的 native/pack/PLE 参数> --serve --spec 4 --mtp-max-t 4 --expert-cache 8300 --prefill 32 --max-context 512 --prompt-cache 0 --conversation-cache-mib 0 --suffix-draft 0 --adapt-every 0
```

不得同时传 `--mtp` 或 `--mtp-adaptive`。所有生成结果必须附带权重账单；旧二进制不能用于此实验。

已有 AutoRound 入口也支持显式开启，默认选择 8,300 个 GPU 专家，并启用 graph 与成本选窗：

```bash
bash /root/strata-dcu/run-autoround.sh \
  --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint \
  --prompt 'Write a Python LRU cache class using OrderedDict.' --max-new 256
```

`run-command.json` 保留原生引擎命令，`dflash2-command.json` 保存实际 wrapper 命令；`generation.json` 记录 `draft_method=dflash2`、权重账单及草稿统计。未传 `--dflash2` 时继续使用原增强 MTP 配置。

## 记录

原始请求、输出 token IDs、权重账单、日志及草稿耗时存放在 `profile_output/dflash2/`。首次 smoke 因短提示词特征缺失失败，保留日志，修复后单独重跑，不将失败运行作为有效性能结果。
