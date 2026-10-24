# 公开入口独立复核

仅在主会话已完成部署、确认GPU为2MiB且没有其他推理/构建任务后执行。
从本地Strata仓库原样运行一次，不修改脚本、入口或配置，不重试：

```bash
ssh -S /tmp/strata-q4-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; python3 -u /root/private_data/strata-dcu/q4-input-bias-20261025/validate_entry.py'
```

预期退出0，依次输出：

```text
PASS ENTRY mtp-default
PASS ENTRY mtp-override
PASS ENTRY native-opt-in
PASS ENTRY joint-opt-in
PASS Q4 INPUT BIAS PUBLIC ENTRIES; GPU idle
```

生产链接`/root/strata-dcu/build-gfx936/strata`应指向持久盘
`/root/private_data/strata-dcu/q4-input-bias-20261025/strata-final`，SHA256为
`4fc8304ec17c7e6b0df4a843f7f60ebe21934e6a7e88004a358e6b2356edb8cd`。
`deployment.json`记录已核验的部署目标及本机默认值。

脚本通过真实公开wrapper做四次冷启动：

1. 增强MTP默认值：本机模型配置`cpu_q4_input_bias=true`，Q8非专家权重、HC Q8、蒸馏MTP，CPU新算式实际启用。
2. 增强MTP显式覆盖：`STRATA_CPU_Q4_INPUT_BIAS=0`，实际新算式标记不出现，其他默认值相同。
3. 原生DFlash2显式启用：`STRATA_CPU_Q4_INPUT_BIAS=1`；原BF16非专家权重，原模式的层内交接默认关闭。
4. 联合模式显式启用同一开关，MTP前缀为2；原BF16非专家权重，原模式的层内交接默认关闭。

每次要求准确复制八句文本，启用suffix lookup=3并实际命中。
检查generation.json的请求开关和stderr实际分发标记，核验模型/专家数、完整显存账单与其他默认开关。
增强MTP固定8500个GPU专家，总权重29373360352B；原生/联合固定8300个，总权重32827238400/34195377152B。
均不超过34351349760B。12workers加主线程，PLE继续在持久盘。
该检查不做TPS评测，CPU绑定沿用生产入口原有设置。

原始命令、生成JSON、stdout/stderr与checks.json写入远程本目录的entry/。
最后要求所有引擎退出，GPU回到2MiB，公开链接及二进制摘要不变。
独立会话只报告实际退出码、上述标记和文档/实际差异，
将完整命令输出及结论写到本地`profile_output/q4-input-bias-20261025/entry-validation.txt`。
