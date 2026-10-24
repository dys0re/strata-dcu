# 当前公开入口的独立复核

仅在父代理确认 GPU 空闲后运行。环境沿用 `.aris/compute/scnet-dflash2.md` 的 dflash2@52f8bc84，
无需安装或更新依赖。SSH 控制连接 `/tmp/strata-dcu-control` 已由父代理建立。

从本地 `/home/dry/data/strara-dcu/Strata` 原样运行一次：

```bash
ssh -S /tmp/strata-dcu-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; python3 -u /root/private_data/strata-dcu/layer-scheduling-20261024/validate_entry.py'
```

预期退出 0，依次打印 `PASS ENTRY mtp`、`PASS ENTRY native`、`PASS ENTRY joint`，
最后打印 `PASS ALL PUBLIC ENTRIES; GPU idle; no remaining engine`。
证据保存在远程 `/root/private_data/strata-dcu/layer-scheduling-20261024/v9/entry/`。
这是三个公开入口的冷启动、原样复制回答、lookup、默认选项和权重账单检查，不是吞吐测量。
不要在失败后自行修改命令或脚本；报告实际分歧，由父代理处理。

当前公开引擎软链接为 `/root/strata-dcu/build-gfx936/strata`，目标为
`/root/private_data/strata-dcu/layer-scheduling-20261024/v9/strata-final`。
SHA256 必须为 `a2ac48e5fe51fb18d93db50de4bfdb9fdf1d3682877284b1a7b2d4ff217336c2`。

默认增强 MTP 应开启 GPU 自行安排显存专家、CPU 按专家推进和提前发布结果；
原生 DFlash2、联合模式的这些新选项仍关闭。三种入口均使用 HC staged、CPU Q4 多位置内核和
原生 DFlash 小投影选项。分流、交换、层末休眠保持关闭；lookup 在本检查中显式设置为 3。
MTP 固定 8500 个 GPU 专家，总权重 33542205952 字节；原生 DFlash2 固定 8300 个，
总权重 32827238400；联合模式固定 8300 个，总权重 34195377152。
全部不得超过半显存上限 34351349760 字节。CPU 池 12 个工作线程加主线程，PLE 在持久化磁盘。
