# 当前公开入口独立复核

仅当主会话确认 GPU 空闲（2 MiB）且没有其他推理任务时执行。不要修改脚本、入口或配置，也不要重试。
从本地 Strata 仓库运行一次：

```bash
ssh -S /tmp/strata-dcu-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; python3 -u /root/private_data/strata-dcu/model-specific-20261024/validate_entry.py'
```

期望退出 0，依次输出：

```text
PASS ENTRY mtp
PASS ENTRY native
PASS ENTRY joint
PASS ALL PUBLIC ENTRIES; GPU idle; no remaining engine
```

脚本做三次真实冷启动，原样调用 `/root/strata-dcu/run-autoround.sh`：增强 MTP、原生 DFlash2、联合模式。
每次复制相同的八句文本并显式使用 suffix lookup=3，检查实际回答及 lookup 已使用。
它核对生产链接 `/root/strata-dcu/build-gfx936/strata` 指向
`/root/private_data/strata-dcu/model-specific-20261024/final-v1/strata-final`，SHA256 为
`3ae138ead7fdbab0324bb17b423022d32c497eb18e3fb5aa82901011b98c9450`。

MTP 默认配置来自 `/root/private_data/strata-dcu/model-specific-20261024/selected-model.json`：
Q8 投影和输出头、HC Q8、蒸馏 MTP，筛选头关闭；原有三项层内交接选项开启。
GPU 专家固定 8500 个，专家权重 23500800000 B，全部模型权重 29373360352 B。
原生和联合默认仍用原 BF16 非专家权重及原始 MTP（联合时），层内三项交接选项仍默认关闭；
GPU 专家各 8300 个，全部模型权重分别为 32827238400 / 34195377152 B。
三种模式均使用 HC staged 配置值 2、CPU Q4 多位置和 DFlash 小投影开关；
增强 MTP 实际 HC 路径由 HC Q8 接管。交换、PCIe 专家分流、层末休眠均关闭。
12 workers 加主控，共 13 个参与线程。PLE 继续读取持久盘。
物理显存 68702699520 B，权重上限 34351349760 B。

原始输出、生成 JSON、权重账单和前后 GPU 状态保存到：
`/root/private_data/strata-dcu/model-specific-20261024/final-v1/entry/`。
最后所有引擎退出、GPU 回到 2 MiB。独立会话仅报告实际退出码、标记及差异，
将完整 stdout/stderr 与结论写入本地同目录 `entry-validation.txt`。
