# 无 MTP 时的验证窗口对照

相同 BW/gfx936、AutoRound INT4 权重、42-token 提示词、256-token 输出、512 上下文上限、12 个 CPU workers。原始权重及引擎二进制未改变，PLE 始终从原始磁盘文件读取。仅将实际验证窗口限制为 1。

| 运行 | 参数 | decode | 耗时 | 结果 |
| --- | --- | --- | --- | --- |
| 初始入口 | --spec 2 | 18.77 token/s | 13,637.1ms | 255 个无效候选均拒绝 |
| 直接对照 | --spec 2 --mtp-max-t 1 | 25.53 token/s | 10,029.1ms | 输出 256 个 IDs 与初始逐一相同 |
| 修正后的正式入口 | 同上 | 25.72 token/s | 9,954.7ms | 输出 256 个 IDs 与初始逐一相同 |

三次均退出 0。吞吐排除加载和 prefill。本组按顺序运行，未做交错多轮，约 36–37% 的改善仅代表此工作负载。尝试 --spec 1 被 native-pack 入口校验拒绝，退出 2，因此没有该参数的 TPS 测量。

## 已有时序与解释边界

初始每轮 host 等同步为 37.259ms，CPU pool 为 7.818ms；修正后两次分别为 27.101/27.022ms 和 6.085/6.013ms。同步等待包含设备执行及依赖，不能当作 GPU 纯计算时间。专家文件读取为 0；这项计数不包含 PLE 读盘。

上游 README 的 RTX 5070 / Q2_0 为 94 token/s、RX 9070 XT / Q2_0 为 60 token/s。DETAILS 明确其 RTX 主表开启 MTP。当前为不同量化、不同硬件，MTP 关闭，不能把倍率差直接归因于硬件性能。当前 gfx936 仍使用 wave64 兼容路径，BF16 原生投影采用新增通用 kernel；这些是待测优化对象，未证明具体占比。

## 文件变更

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| tools/autoround_pack.py | 修改 | 无 drafter 的入口增加 --mtp-max-t 1 |
| envs.md | 修改 | 保留初始数据，追加窗口修正和复跑结果 |
| profile_output/autoround-window1*.log | 新增 | 对照及正式入口日志 |
| profile_output/autoround-window1*.json | 新增 | 命令、生成文本及 IDs |
| profile_output/autoround-spec1.stderr.log | 新增 | 不支持 --spec 1 的真实失败记录 |
| profile_output/window1-comparison.md | 新增 | 本报告 |

没有增加计时插桩。旧补丁及旧源文件哈希属于初始已提交版本；新配置只改一个 Python 文件，远程已同步。新增日志可以单独清理。
