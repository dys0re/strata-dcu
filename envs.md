# DCU 开发环境

最后整理：2026-11-09。运行记录截至 2026-11-07；本文不表示远程节点当前在线。
构建与基本运行见 [README](README.zh-CN.md)，测量约束见 [测试标准](docs/DCU_TEST_STANDARD.md)。

## 已验证环境

| 项目 | 当前 worker 的验证配置 |
|---|---|
| 系统 | Linux x86_64，Hygon CPU |
| 可用资源 | 32 个 CPU（0–15、80–95），121.5 GiB 内存 |
| DCU | 物理 GPU0，BW/gfx936，65,520 MiB 显存 |
| 软件 | DTK 26.04、HYHAL、Python 3.11.9 |
| 转换环境 | NumPy 1.25.0、Transformers 4.57.6、sentencepiece 0.2.1 |
| 草稿参考环境 | 海光适配 PyTorch 2.9.0；原生引擎不启动 PyTorch 草稿进程 |
| GGML | 固定提交见 [VERSION.txt](third_party/ggml/VERSION.txt) |

早期实例只有 15 核配额和 59 GiB 内存；两实例数据不能直接作为同条件对照。
容器的实际资源要读取 cgroup 和 CPU affinity，不能仅看宿主 `free`/`lscpu`。

在同一个 shell 加载动态库环境，并限制为单卡：

```bash
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
hy-smi
```

## 模型准备

将模型、运行缓存、PLE 和大型构建放在平台持久化目录；路径由部署环境决定。
基础转换不改写源 checkpoint：INT4 专家 code/scale 重排为 Q4_0，dense 保留 BF16/F32，
外部 PLE 按描述文件读取。准备前核对源索引与文件完整性。

```bash
python3 tools/autoround_pack.py prepare \
  --model /persistent/Qwen3.8-F --out /persistent/autoround \
  --llama /path/to/llama.cpp --workers 4
```

进一步的 Q8 dense/head、HC Q8、蒸馏 MTP 与 IQ4_NL PLE 见
[模型变体](docs/DCU_MODEL_OPTIMIZATIONS.md)。它们需要独立准备和显式选择，
不能把基础转换命令视为已经启用全部优化。量化会改变模型数值。

`tools/recover_autoround_shards.py --help` 提供源 PLE 缺失、外部 PLE 仍可用时的专家恢复入口。
2026-11-06 的恢复记录包含 48 层、432 项参考比较、逐分片摘要及迁移前后生成检查。
该工具要求其余文本权重完整，不能恢复缺失的任意源张量。已发布的分片应位于持久盘。

## 当前测量配置

GPU0、8500 GPU 专家、12 workers 加主控、INT8 KV、context512、adaptive MTP/maxT5；
Q8 dense/head、HC Q8、stable MMAC、蒸馏 MTP、磁盘 IQ4_NL PLE。
GPU 权重 29,373,360,352 B，保持半显存预算。当前机器入口开启 CPU 队列优化，关闭 PCIe 分流。
具体对照和验证边界见 [CPU 队列](docs/DCU_HETERO_SCHEDULING.md)、
[图调度](docs/DCU_GRAPH_SCHEDULING.md)、[MMAC](docs/DCU_Q8_MMAC_PLAN.md)。

原始运行数据保存在本地 `profile_output/`，不随源码提交。
本文件不再重复历史逐轮日志、登录入口、旧引擎摘要及瞬时资源状态。
