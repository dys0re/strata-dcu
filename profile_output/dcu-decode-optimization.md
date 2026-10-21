# gfx936 AutoRound decode 优化实测

2026-10-21，单卡 BW/gfx936 64GB，15 核 CPU 配额、59GiB 内存，DTK 26.04，Strata engine 0.1.46.3。代码基于本地提交 `d2cf641` 加本次改动。原始 Qwen3.8-F AutoRound INT4 权重与准备好的缓存保持不变，PLE 从原始磁盘文件读取，MTP 关闭，上下文上限 512，12 个 CPU workers。

## 完整生成结果

英文故事使用相同的 42-token 提示、相同二进制和权重，每次生成 256 tokens，顺序为基线、新内核、新内核、基线。两组均使用之前修正的实际单 token 验证窗口。decode 计时不含模型加载、prefill。

| 配置 | 两次 decode / token/s | 中位数 / token/s |
| --- | --- | --- |
| 原 BF16/Q8_1 内核 | 24.76、24.49 | 24.63 |
| 新向量化内核 | 33.56、32.48 | 33.02 |

相同工作负载下中位数提高 **34.09%**。这是两次一组的交错测试，未完成大规模统计。

常驻 stdin 引擎另外逐一运行五个请求，不监听 HTTP 端口，不复用提示前缀。长请求的结果：

| 请求 | 原内核 / token/s | 新内核 / token/s | 输出 tokens |
| --- | --- | --- | --- |
| Python LRU 类与说明 | 25.94 | 36.16 | 256 |
| 中文解释 prefill/decode | 26.06 | 36.87 | 256 |

两种入口的样本不能合并算平均速度。正式 `run-autoround.sh` 默认配置再次运行相同中文请求：**256 tokens / 7,200.4ms = 35.55 token/s**，退出 0；prefill 38 tokens / 2,741.0ms，引擎报告首 token 等待 27.75s；这不包含启动脚本的完整冷启动过程。

最早 18.77 token/s 的测试还多验证一个无效 token。之前加上 `--mtp-max-t 1` 后为 25.53 / 25.72 token/s。上述 34.09% 的对照双方都已修正窗口，因此不把窗口修正重复计算为向量化的收益。

## 改动与数值检查

新内核每个线程合并读取四个 BF16 权重和四个 Q8 code，使用四个累加器；宽归约使用 64 lanes，短输入使用 32 lanes。没有重新量化或修改权重，Q8_1 激活格式也不变。独立算子比较在七种实际矩阵形状上测得约 1.4–3.2 倍速度，完整生成收益以上表为准。

FP32 求和顺序会改变。`STRATA_BF16_Q8_VEC4=1` 仅在 gfx936、waveSize 64 上生效；引擎默认关闭，本机的启动脚本默认开启。`STRATA_BF16_Q8_VEC4=0` 保留原求和顺序。

| 检查 | 实测 |
| --- | --- |
| 新旧路径各 60 种矩阵形状，对照独立 CPU double 点积 | 两组通过；输出有限，误差在原有求和容差内 |
| 多输入列、32/64/640/2560/6144 输入宽度、1/5/33 输出行 | 通过，含尾行和短输入 |
| BF16 转 F32/F16 行切片 | 通过 |
| `OK`、`2+2`、中国首都 | 新旧路径均为 `OK`、`4`、`北京`，EOS 停止 |
| 同一路径重复两次英文生成 | 各自全部 token IDs 相同，首 token logits 逐位相同 |
| 新旧英文首 token 分布 KL | 0.00003048；首个不同输出 token 的零基下标为 180 |
| 另外五个请求首 token 分布 KL | 0.00007629–0.00047987，argmax 全部相同 |
| 内核启动拒绝、非有限 logits、子进程失败 | 本组测试未出现 |

长输出不保证与旧路径逐 token 相同。首 token KL 和少量文本检查不等于全面质量评测。代码请求达到长度上限时类尚未完成，因此没有执行正确性结论。未验证其他硬件、CUDA/SYCL 编译、视觉、MTP、长上下文或并发。

## 定位与未采用的设置

内置 GPU 计时以前只在 serve 路径输出，本次为单次生成补上同一张阶段表，仍由 `STRATA_VERIFY_PROFILE=1` 控制，正常运行不插桩。实际查询 gfx936 墙钟为 25,000kHz，即 40ns/tick，与兼容路径的时间换算一致。

插桩运行中，GDN QKV、门投影、输出投影分别约 3.60、2.74、3.43ms/window，共享专家阶段合计约 4.38ms，输出头约 2.29ms。插桩增加内核节点、同步，且会关闭一些融合或跨流路径；这些时间仅用于定位，不能直接预测正常吞吐。

PLE 累计阻塞约 174ms 的统计包含启动探测、prefill 和 128-token decode，不能直接除以 128 当作每 token 读盘成本。专家文件读取为 0；该计数不含 PLE。

| 额外对照 | 结果 | 选择 |
| --- | --- | --- |
| 关闭路由预取 | 24.96 → 25.18 token/s，单次 | 差距小，保留默认 |
| 新内核关闭共享专家跨流 | 33.60 → 27.71 token/s，单次 | 明显变慢，保留默认跨流 |

## 复现

默认优化入口：

```bash
bash /root/strata-dcu/run-autoround.sh --prompt '请详细解释大语言模型的 prefill 和 decode 阶段有什么区别，以及它们各自的性能瓶颈。请分段说明并举例。' --max-new 256 --context 512
```

相同入口关闭新内核：

```bash
STRATA_BF16_Q8_VEC4=0 bash /root/strata-dcu/run-autoround.sh --prompt 'Reply with exactly OK.'
```

完整命令与运行输出见 [交错对照](vec4-story-abba/results.json)、[多请求检查](vec4-quality-suite/results.json)、[正式入口日志](final-wrapper.log)、[独立算子测速](bf16-layout2.csv) 和 [数值结果汇总](optimization-results.json)。

## 文件变更与诊断清理

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| src/kernels/cuda/native_mmvq.cu | 修改 | 有开关和 gfx936 运行时保护的 BF16/Q8_1 向量化路径 |
| src/kernels/bf16_q8_mmvq_parity.cpp | 修改 | 扩展为 60 种形状，独立数值检查 |
| src/program/generate.cpp | 修改 | 标记 `[PROFILE]` 的三行，输出已有 GPU 阶段表 |
| tools/autoround_pack.py | 修改 | 无 MTP 时限制实际验证窗口为 1 |
| strata-test/run-autoround.sh | 修改 | 本机默认开启新内核，可用环境变量关闭 |
| envs.md | 修改 | 追加窗口修正及性能记录 |
| profile_output/bf16-layout-bench.hip | 新增 | 独立矩阵布局比较程序 |
| profile_output/decode-matrix.py | 新增 | 串行、可交错的完整生成对照 |
| profile_output/vec4-serving-suite.py | 新增 | 不复用前缀的多请求文本和 logits 检查 |
| profile_output 下的原始数据和报告 | 新增 | 逐文件清单见 [optimization-artifacts.tsv](optimization-artifacts.tsv) |

远程另有 `profiles/clock-rate.cpp`、其查询程序、`profiles/bf16-layout-bench`，以及构建目录中的 `strata-before-profile`、`strata-before-vec4` 备份；未写入模型目录。源码和二进制哈希见 [本次审计](optimization-audit.json)，增量代码见 [补丁](decode-optimization.patch)。原路径的 256 个英文输出 IDs 也与初始版本完全一致。

正常启动不启用计时插桩。诊断脚本和原始数据可以单独清理；需要时也可移除标记的三行阶段表输出，而不影响优化内核。
