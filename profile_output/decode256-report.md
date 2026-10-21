# Decode 测试

单次远程测试：BW/gfx936 单卡，Qwen3.8-F AutoRound INT4，单请求，42-token 输入，512-token 上下文上限，12 个 CPU workers，MTP 关闭，PLE 从磁盘读取。

| 指标 | 实测 |
| --- | --- |
| 输出长度 | 256 tokens |
| Decode 墙钟耗时 | 13,637.1ms |
| Decode 吞吐 | 18.77 token/s |
| 平均每 token 耗时 | 53.27ms |
| 引擎 prefill（不含最后一个输入 token） | 41 tokens / 3,069.0ms |
| 引擎报告首 token 等待 | 29.92s |
| 退出码 / 停止原因 | 0 / length |

Decode 计时排除模型加载与 prefill。生成文本为连贯英文故事；原始文本、token IDs 及计时见同目录日志和 JSON。仅测一次短上下文，不代表长上下文或并发吞吐。

## 测试文件变更

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| profile_output/autoround-decode256.log | 新增 | 远程完整运行输出 |
| profile_output/autoround-decode256.json | 新增 | 输入、生成文本、token IDs、退出码 |
| profile_output/decode256-report.md | 新增 | 本报告 |
| envs.md | 修改 | 追加 decode 实测记录 |

远程新增 `/root/strata-dcu/logs/autoround-decode256.log` 和同名 `.json`；复用入口会覆盖其工作目录中的最新生成结果及运行日志。未修改推理代码，未增加计时插桩。清理上述新增测试文件即可移除本次测试材料。
