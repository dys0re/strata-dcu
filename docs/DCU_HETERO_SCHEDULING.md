# CPU 专家队列优化

保留 `STRATA_CPU_POOL_COST=1`：长任务优先，并按实际服务token数分配专家分块。
需要已有的 `STRATA_CPU_EXPERT_PIPELINE=1`。每批总任务数、数值计算、输出位置及完成同步协议不变。
通用引擎仍为显式开关；当前BW/gfx936机器入口默认开启队列优化，PCIe分流关闭。

## 单卡验证

GPU0，8500 GPU专家，12 workers，Q8 dense/head和HC、stable MMAC、IQ4_NL PLE、INT8 KV，
context512，MTP上限5，权重仍受半显存限制。
每模式每领域六次256-token生成，四进程ABBA，排除载入与prefill。

|任务|固定窗口原队列/优化 TPS|adaptive原队列/优化 TPS|
|---|---:|---:|
|英文|34.210 / 33.506|37.420 / 37.302|
|中文|53.904 / 52.210|51.333 / 51.212|
|代码|67.452 / 70.016|62.626 / 66.722|

速度比几何平均：固定窗口-0.51%，adaptive +1.95%；本轮adaptive代码提示+6.54%。
共享主机上的描述性样本，不推广为所有任务的稳定收益。

552项线程池逐位检查、常规及延迟唤醒CTest、18项整模型QA、生成代码执行、取消/恢复均通过。
HIP及CUDA sm80构建通过；没有NVIDIA设备吞吐验证。
固定窗口输出token一致，adaptive窗口选择变化时不声称逐位一致。
GPU权重账单29373360352 B，驻留专家数保持8500。

## 当前入口

```bash
bash /public/home/bzy2008/strata-dcu/run-autoround.sh --prompt 'Write Python code for binary search.' --max-new 256
```

该机器入口开启队列优化并固定 `--pcie-frac 0`。
`STRATA_CPU_POOL_COST=0` 可临时恢复原队列对照。
冻结引擎SHA256：`9f1d7edfec17b18623c93deb350daa0ce6cf66532ab94fbbeaeee96937dc931e`。
原始试验记录保存在 `profile_output/cost-schedule-20261107/`。
