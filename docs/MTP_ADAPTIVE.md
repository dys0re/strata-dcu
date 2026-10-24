# 单请求成本感知 MTP

本次将 DeepInfer 的“按接受率与实际成本选择验证长度”设计用于 Strata 的 Qwen3.8-F 单请求路径。实现使用本地 Qwen MTP 权重及历史接受统计，没有引入 DeepSeek 草稿权重或训练 Confidence Head。主模型仍使用原 AutoRound INT4 缓存，PLE 仍从磁盘读取。

功能默认关闭。以下结果只验证了当前 BW/gfx936、DTK 26.04、INT8 KV、512 上下文上限、一个并发请求；没有宣称其他 GPU、Metal、长上下文、并发批处理或完整模型质量基准通过。

## 当前机器直接运行

草稿缓存已经准备在 `/root/private_data/strata-dcu/autoround/mtp`，旧路径 `/root/strata-dcu/autoround` 通过软链接指向该持久化缓存。主模型转换缓存也在同一持久化目录中，避免占用平台限额为 15GB 的可保存镜像层。

```bash
bash /root/strata-dcu/run-autoround.sh \
  --mtp-mode adaptive --mtp-max-t 5 \
  --prompt '请详细解释 prefill 和 decode 的区别。' --max-new 256
```

`--mtp-max-t 5` 包含一个真实输入位置，即最多验证四个草稿。接口支持 2–8，实际生成测试也覆盖了上限 8。对照选项：`--mtp-mode off` 不加载草稿；`--mtp-mode fixed --mtp-max-t 3` 使用固定三位置窗口。

重新准备独立草稿缓存：

```bash
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
python3 /root/strata-dcu/Strata/tools/autoround_pack.py mtp \
  --model /root/private_data/Qwen3.8-F \
  --out /root/private_data/strata-dcu/autoround --llama /root/strata-dcu/llama.cpp
```

源文件中的 1,565 个 BF16 MTP 张量约 4.856GiB，转换时先逐位保留并融合成现有 ABI 的 31 个张量，再派生草稿缓存。现有 MTP 运行时要求 routed experts 为 Q2_0、主要投影为 Q8_0；其余 BF16/F32 按原运行时规则处理。这是草稿专用的派生副本，原 Safetensors、主模型专家和 PLE 文件均不写入。

缓存通过临时目录生成，成功后原子发布；`ready.json` 绑定原 checkpoint inventory、配置与索引哈希，运行时检查四个运行文件的大小及修改时间。准备时另记录 SHA-256 供审计。缓存缺失或不匹配会报错，不会静默切换模型。

## 策略与状态

每个候选窗口 T 的收益估计为 `1 + Σ A_j`，其中 A_j 是前 j 个草稿都被接受的历史频率。窗口成本包含主模型验证、commit/输出以及生成该窗口草稿的时间。选择依据是收益/耗时，同时有少量探测、短窗口偏好和随探测次数衰减的探索奖励。

实现要点：

- 决策只读取已完成轮次的反馈，不读取本轮正在生成的草稿 token 或其置信度。
- 当前窗口的成本配对它之前的草稿准备时间；不会错配为下一轮不同长度的草稿成本。
- 在 decode 计时前准备候选验证图、MTP round/step 图和前端 KV 图，避免把懒捕获作为稳态成本。
- 成本使用最多九个样本的中位数，首次观测不计入成本。新请求保留一个成本先验，再交错刷新各尺寸；接受统计也向新请求倾斜。
- 候选包含 T=1，也就是不验证草稿。此时仍运行 MTP 的前端来补齐已提交位置的 KV，以后可以恢复投机；它不等于卸载 MTP，仍有驻留显存和维护开销。
- EOS 终止轮不参与收益学习，避免把没有发出的 bonus token 计作收益。窗口受剩余输出和上下文预算约束。

本版使用原 exact-match verifier，支持 greedy 和普通采样，包括 coupled draft 模式。实验性的 `STRATA_SPEC_PROB=1` 与自适应模式互斥；suffix prompt lookup 现可通过 `--suffix-draft 3` 与增强 MTP 并用，查找窗口不会更新 MTP 的接受率统计，见 [DCU 运行选项](DCU_RUNTIME_OPTIONS.md)。固定置信度阈值、lookup chain、batch slots、流水线窗口及多 GPU 仍不与这个单请求策略混用。冲突会明确报错。

## 常驻服务配置

原服务配置的 `args` / `user_args` 可以透传新引擎参数。将下列选项合并进当前单 GPU、单请求的配置：

```json
{
  "mtp": true,
  "user_args": [
    "--mtp", "/root/strata-dcu/autoround/mtp",
    "--spec", "5", "--mtp-max-t", "5", "--mtp-adaptive",
    "--suffix-draft", "0", "--lookup-chain", "0", "--spec-min-p", "0"
  ]
}
```

原有 `"mtp": false` 开关会同时移除 `--mtp-adaptive`，可继续正常关闭草稿。原生协议的 EOS、STOP 和输出预算语义不变。这里没有启动常驻 HTTP 服务。

## 本机验证结果

2026-10-22，保持同一主模型、19,000 个静态专家缓存槽、相同 tokenizer 和提示、INT8 KV、512 上限、12 个 CPU workers、bs=1；关闭 route-tail skip、PCIe 专家分流和自适应专家交换，以便逐 token 比较。每个引擎先热身，再生成 256 tokens。decode 不含加载和 prefill。每种工作负载/模式只有一次正式测量，不是统计显著性结论。

| 工作负载 | MTP off | 固定 T=2 | 固定 T=3 | 自适应 |
| --- | ---: | ---: | ---: | ---: |
| 英文故事 | 34.01 | 36.54 | 34.07 | **35.92** |
| 中文解释 | 34.85 | 40.43 | 39.65 | **40.72** |
| Python 代码与说明 | 35.88 | 45.67 | 49.97 | **50.31** |

单位均为 token/s。自适应相对 off 的本组提升约 5.6%、16.9%、40.2%；不保证每次都超过最佳固定窗口。固定 T=4/5 的早期对照也已执行，未混入上述最终表格。

正式启动脚本另按默认自动缓存复测英文 256 tokens，off 为 34.39、自适应为 35.78 token/s。自动缓存下 MTP 的额外显存会减少主模型驻留专家数量，因此这组不是固定缓存预算实验，也不与表格求平均。实际 MTP 显存约 1.3GiB，主要文件约 0.77GiB。

| 检查 | 证据 |
| --- | --- |
| 本地 MTP 布局与缺失/错误形状拒绝 | Python 打包测试 5/5 通过，含原 INT4 测试 |
| 选窗、零收益回退、草稿成本归属、初始/新任务采样、漂移、尾部和非法观测 | 本地 g++ 与远程 DTK 编译的 `mtp_budget_test` 通过 |
| 最终固定 T=2/3 和自适应的 24 个请求 | 与 off 对照逐 token 相同，24 组首 token logits 逐位相同 |
| `OK`、`2+2`、中国首都 | 各模式均正确，EOS 停止 |
| EOS 收益反馈 | 热身与三个短问答后的观测次数不变，终止窗口未被学习 |
| 零草稿 KV | 在五处实际位置对照前端-only 与完整草稿路径，新写入 KV 逐字节相同；含两行补齐和后续恢复投机 |
| 原生 STOP | 取消长请求，随后新请求正确回复 OK |
| 固定 seed 普通采样 | 跨模式一致；同一引擎内重复请求也一致 |
| 输出/上下文上限 | 单 token 请求及 501-token 输入 + 3-token 输出通过；保留原服务入口的 8-token 余量 |
| 验证上限 T=8 | 实际执行 T=8 窗口，64 个输出 IDs 与 off 前缀相同 |
| 非法配置 | 缺 MTP、启用 lookup、实验性概率 verifier 均退出 2；请求级 spec_min_p 被拒绝且服务保持可用 |
| 默认 off 兼容性 | 正式脚本英文 256 IDs 与提交 `6ec15a8` 的记录完全一致 |
| API stop strings | 现有 mock 单元测试 7/7 通过，单列记录，不当作模型推理测试或性能数据 |
| 服务/Control 开关 | 5/5 单元测试通过，含关闭 MTP 时移除新参数 |

原始数据、自动核验和审计入口在 [p0-test](../p0-test/README.md)。先前边界测试曾使用 509+3 tokens，被原服务的额外八 token admission 余量拒绝；更正为最后合法边界 501+3 后通过，失败记录保留。早期选窗的冷成本和任务切换滞后也保留为探索记录，没有作为最终性能结果。

开发设计参考 DeepInfer [MTP 成本选择](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp)。这里是面向本机 Qwen/bs=1 的独立实现，不是直接复制 SM100、DeepSeek 模型或其 Confidence Head。
