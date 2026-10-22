# P0 MTP 移植验证材料

实现和使用说明：[单请求成本感知 MTP](../docs/MTP_ADAPTIVE.md)。基线提交为 `6ec15a8`，本目录属于后续 P0 改动。验证机器为 BW/gfx936；模型固定 Qwen3.8-F AutoRound INT4，PLE 在磁盘。

## 复核

```bash
python3 p0-test/verify_results.py
g++ -std=c++17 -O2 -Iinclude src/spec/mtp_budget_test.cpp -o /tmp/mtp-budget-test
/tmp/mtp-budget-test
```

`verify_results.py` 会直接读取保存的 token IDs、首 token logits、控制协议结果和 KV 诊断记录，不以汇总表代替原始结果。

| 材料 | 用途 |
| --- | --- |
| p0-final-matrix/ | 同一固定缓存预算的 off、固定 T=2/3 控制；其中旧自适应数据保留作迭代记录 |
| p0-final-adaptive/ | 最终 EOS-censored 自适应实现；含热身、三个短问答、英文/中文/代码、采样和 admission 尾部 |
| verified-results.json、verification.log | 自动核验结果和最终吞吐表 |
| p0-protocol/ | 原生 STOP/恢复、同 seed 重复、单 token 上限、非法请求和三项非法引擎配置 |
| p0-front-check.* | 实际 KV 逐字节对照；诊断有额外计算，不能当作吞吐基准 |
| p0-cap8/ | 最大验证窗口的实际生成检查 |
| ready.json、source-tensors.json、quantization-report.json | 独立草稿缓存的源绑定、BF16 融合 manifest 和派生量化报告，不含大权重文件 |
| benchmark.py、protocol_checks.py | 远程复现脚本，保存所有输入、命令和原始输出 |
| budget-unit-test.log、p0-pack-tests.log、control-tests.log | 原生策略、布局和服务开关测试 |
| api-stop-unit-tests.log | 使用 mock engine 的 API stop 单元测试，日志中的模拟速率不是模型 TPS |

计时脚本运行多个独立引擎，每个引擎串行处理请求，不复用提示前缀。表中数字均为热身后的单次测量，不包含模型加载和 prefill，不合并不同实验条件。

原生 CLI：`--mtp-adaptive` 必须同时有 `--mtp`、`--spec 2..8`，禁用 suffix/lookup、batch、流水线、多卡和固定 spec-min-p；`STRATA_SPEC_PROB=1` 不与本版组合。入口包装器用 `--mtp-mode off|fixed|adaptive` 设置这些参数，默认 off。

## KV 诊断

```bash
STRATA_MTP_FRONT_CHECK=1 bash /root/strata-dcu/run-autoround.sh \
  --mtp-mode adaptive --max-new 96 --prompt 'Write a long story about an observatory.'
```

诊断只用于当前 fully resident INT8 KV：零草稿时先写入 KV，再对同一批输入运行原完整草稿路径，逐字节比较相同位置的 K/V 及量化 scale。出现差异会使推理失败；正常基准不设置该变量。

固定 T=4/5 和旧的自适应试验保留在远程 `/root/strata-dcu/profiles/p0-matrix`。该早期驱动最后因 509+3 未计入八 token admission 余量而退出 1；不是一个全绿的测试批次。最终合法边界测试为 501+3。之前的自适应 v2/v3 也未删除，不作为最终表格使用。

大模型原文件和草稿 runtime 二进制只留在远程，不写入 Git。最终源码/二进制和原始模型不变检查见本目录 `audit.json`；所有材料的路径及 SHA-256 见 `artifacts.tsv`。
