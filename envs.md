# SCNet DCU 开发环境

当前速度测试遵循[GPU 权重占用不超过 50% 的标准](docs/DCU_TEST_STANDARD.md)。主模型、输出头和 MTP 一起计入；此前满显存/19,000 专家配置的速度保留为历史记录。

最后更新：2026-10-25。本文记录远程实例的配置、环境加载和联网测试结果；未包含登录密码或代理认证凭据。

## 连接

```bash
ssh -p 11754 root@ssh.zzai.scnet.cn
```

这是 SCNet Notebook 容器实例，用户为 `root`。SSH 密码单独保管。

## 实例资源

| 项目 | 实测结果 |
| --- | --- |
| 系统 | Ubuntu 22.04.5 LTS，x86_64 |
| CPU | 海光 Hygon C86，实例配额 15 核 |
| 内存 | cgroup 上限 59 GiB，无 Swap |
| 加速卡 | 1 × 海光 BW DCU |
| 显存 | 63.98 GiB，平台标称 64GB |
| 架构 | `gfx936:sramecc+:xnack-` |
| 计算单元 | PyTorch 报告 `multi_processor_count=80` |
| 共享内存 | `/dev/shm` 约 59 GiB |

`lscpu` 和 `free -h` 展示的是宿主机的 128 个逻辑 CPU、503 GiB 内存，不能据此判断容器可用资源。cgroup 实测为：

```text
cpu.max:    1500000 100000
memory.max: 63350767616
```

首次检查时卡处于空闲状态，利用率 0%，温度约 51°C；这些是检查时的瞬时状态。

## 软件环境

| 项目 | 版本或位置 |
| --- | --- |
| Python | 3.11.9，`/usr/local/bin/python3` |
| Notebook 内核 Python | `/usr/local/bin/python3.11` |
| PyTorch 安装包 | `2.9.0+das.opt1.dtk2604` |
| `torch.__version__` | `2.9.0` |
| DTK | 26.04，`/opt/dtk -> /opt/dtk-26.04` |
| `torch.version.hip` | `6.3.26093` |
| `torch.version.cuda` | `None` |
| Transformers | 4.57.6 |
| NumPy | 1.25.0 |

这是海光适配的 PyTorch/HIP 环境，设备仍通过 `torch.cuda` 接口使用。未加载动态库路径的 SSH 会话中，`import torch` 曾报错：

```text
ImportError: libgalaxyhip.so.5: cannot open shared object file
```

在执行程序的同一个 Shell 中先加载：

```bash
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
```

查看 DCU 状态：

```bash
hy-smi
```

已通过的最小计算检查：

```bash
python3 - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("Device count:", torch.cuda.device_count())
print("Device:", torch.cuda.get_device_properties(0))
x = torch.randn(256, 256, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("FP32 matmul finite:", torch.isfinite(y).all().item())
print("BF16 supported:", torch.cuda.is_bf16_supported())
PY
```

实测识别到 1 张 BW 卡，FP32 矩阵运算结果有限，BF16 支持查询返回 `True`。本次未进行 BF16 数值测试或训练性能测试。

## 存储

| 路径 | 用途与检查结果 |
| --- | --- |
| `/` | 容器系统盘；底层文件系统显示总量约 1.8T，检查时剩余约 1.1T |
| `/root/private_data` | 平台个人持久化数据目录 |
| `/root/group_data` | 平台团队共享数据目录 |
| `/root/public_data` | 平台公共共享数据目录，平台说明为只读 |

模型、转换缓存等大文件必须放入 `/root/private_data`。平台保存环境时要求镜像层不超过 15GB；`df` 显示的底层剩余空间不代表镜像层配额。NFS 显示的 PB 级容量是共享存储容量，不等于个人账号配额；本次未核实账号存储配额。

## 对外联网

SSH 交互登录会话和临时 Notebook 内核均已实测能够通过平台代理访问公网。

- 代理地址：`10.16.1.51:3128`，需要认证。
- 检查时两边已有带认证的 `http_proxy`、`https_proxy` 环境变量。
- 绕过代理进行直连时，请求报网络不可达或超时。
- 百度和 PyPI 返回过 HTTP 200；PyPI 偶有超时。
- `raw.githubusercontent.com` 可下载文件。
- GitHub、Hugging Face 主站在本次测试中多次超时。

脚本和子进程需要继承代理环境。不要仅把代理变量改成无认证的 `http://10.16.1.51:3128`，这不等价于平台现有配置。

可用下列方式检查代理是否存在，避免输出认证凭据：

```bash
python3 - <<'PY'
import os
from urllib.parse import urlsplit

for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    value = os.environ.get(key)
    if value:
        p = urlsplit(value)
        print(key, {"host": p.hostname, "port": p.port,
                    "authenticated": bool(p.username)})
PY
```

## GitHub 与 Hugging Face 镜像测速

以下为此实例经平台代理发起的短下载实测，不能代表其他机器或长期带宽。

- GitHub：下载同一个 ripgrep 14.1.0 Release 压缩包，大小 2,527,296 字节，约 2.41 MiB。
- HF：下载 `google-bert/bert-base-uncased` 的 `model.safetensors` 前 8 MiB。
- 每个镜像测试 3 次；第一轮最多 3 个请求并发筛选，后两轮逐站串行复测。
- 速度包含连接、重定向及接收数据的耗时，单位为 MiB/s。
- 下表中位数和范围仅统计完成请求。各镜像完成下载的同一文件或片段 SHA-256 一致。

| 类型 | 镜像地址 | 中位速度 | 速度范围 | 完成次数 |
| --- | --- | ---: | ---: | ---: |
| GitHub | https://gh-proxy.org | 1.22 | 1.16–2.73 | 3/3 |
| GitHub | https://gh-proxy.com | 0.83 | 0.76–0.88 | 3/3 |
| GitHub | https://ghfast.top | 0.53 | 0.51–0.56 | 3/3 |
| GitHub | https://ghproxy.net | 0.32 | 0.27–0.38 | 2/3 |
| HF | https://alpha.hf-mirror.com | 2.53 | 1.83–3.43 | 3/3 |
| HF | https://hf-mirror.com | 1.30 | 0.72–1.52 | 3/3 |
| HF | https://hf-mirror.net | 1.26 | 1.15–1.85 | 3/3 |

`ghproxy.net` 另一次仅收到约 0.75 MiB，约 32 秒后结束，未完成下载。GitHub、Hugging Face 原站在测速阶段各测试 1 次，均超时。

三个 HF 镜像的以下接口也均返回 HTTP 200 和正确类型的 JSON：

```text
/api/models/google-bert/bert-base-uncased
/google-bert/bert-base-uncased/resolve/main/config.json
```

`hf-mirror.com` 的权重请求最终跳转至 `cas-bridge.xethub.hf.co`，alpha 节点跳转至 `us.aws.cdn.hf.co`；`hf-mirror.net` 在本次请求中保持该镜像域名。

### 使用方式

根据本次结果，GitHub 文件下载优先试 `gh-proxy.org`，备用 `gh-proxy.com`。在原始下载链接前加镜像前缀，例如：

```bash
wget 'https://gh-proxy.org/https://github.com/BurntSushi/ripgrep/releases/download/14.1.0/ripgrep-14.1.0-x86_64-unknown-linux-musl.tar.gz'
```

GitHub 测试覆盖 Release 文件下载，未验证 Git clone。

HF 优先试 alpha 节点，备用 `hf-mirror.com` 或 `hf-mirror.net`：

```bash
export HF_ENDPOINT=https://alpha.hf-mirror.com
```

在 Notebook 中，应在导入 `huggingface_hub` 或 `transformers` 前设置：

```python
import os
os.environ["HF_ENDPOINT"] = "https://alpha.hf-mirror.com"
```

以上配置示例仅在执行后生效，本次检查未修改远程默认代理、Shell 或 HF 配置。

## 本地测试记录

以下文件位于本仓库，测速脚本应在远程实例中、继承平台代理环境后运行：

- [测速脚本](network-check/mirror_benchmark.py)
- [下载测速原始结果](network-check/mirror_results.json)
- [HF API 检查结果](network-check/hf_api_results.json)

## Strata / gfx936 与 AutoRound 模型

实测日期：2026-10-21。源码基于 Niko1221/Strata 提交 `9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`，加上本地 DCU/AutoRound 修改；二进制报告版本 `0.1.46.3`。llama.cpp 固定为 `3cf03257f219afbe7334045ff7c6a06ac68c627d`。

### 导入和运行方式

已支持这台机器上的 `/root/private_data/Qwen3.8-F`：Qwen4Exp 架构、AutoRound INT4、`auto_round:auto_gptq` 打包、对称零点 8、group size 128。

- 专家：将原 INT4 码和 FP16 scale 无损重排为 Q4_0 块，生成 Strata 运行缓存，没有再次量化专家权重。不同或非对称零点会被拒绝。
- 非专家投影：保留 BF16/F32 表示；新增 BF16 原生投影及提示词阶段的类型转换支持。解码仍使用 Strata 的 Q8_1 激活约定，不承诺与原 HF 后端的输出逐位相同。
- PLE：直接从原 Safetensors 文件的已验证偏移读取 BF16 行。128 段按编号连续，仅生成 318 字节描述文件，不复制或压缩约 102.4GB 的表，也不将整表装入内存或显存。
- DCU：使用 wave64 兼容路径编译到 gfx936，修正 HIP 对齐声明和大线程块内核的启动上限。
- 计算图：gfx936 上跳过会在 DTK 中崩溃的可选 hipGraphUpload 预热；保留实例化、执行和同步。其他硬件仍调用原来的上传接口。

原模型索引引用的 17 个文件都存在，文件大小与 Safetensors 头部声明一致，总大小 181,196,007,016 字节；未对全量源文件做哈希校验。原始模型未修改。

远程主要路径：

~~~text
/root/strata-dcu/Strata/                  修改后的源码
/root/strata-dcu/llama.cpp/               固定提交的转换/量化依赖
/root/strata-dcu/build-gfx936/strata      已编译的引擎
/root/private_data/strata-dcu/autoround/ 持久化运行缓存，含 MTP 约 83GiB
/root/strata-dcu/autoround/              指向上述持久化目录的软链接
/root/strata-dcu/autoround/dense.gguf    约 9.9GB 的非专家权重
/root/strata-dcu/autoround/pack/         专家、索引、tokenizer 等
/root/strata-dcu/autoround/ple.gguf      引用原始 PLE 的小描述文件
/root/strata-dcu/run-autoround.sh        单次文本生成入口
/root/strata-dcu/logs/                   测试和运行日志
~~~

直接运行现有缓存：

~~~bash
bash /root/strata-dcu/run-autoround.sh \
  --prompt '中国的首都是哪座城市？只回答城市名称。' \
  --max-new 32
~~~

入口会加载 DTK/HYHAL 环境，用原模型的聊天模板和结束 token，输出 token IDs、文本及统计。默认上下文 512、12 个 CPU workers、单卡、关闭 MTP；本次未启动常驻 HTTP 服务。

重建缓存：

~~~bash
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 /root/strata-dcu/Strata/tools/autoround_pack.py prepare \
  --model /root/private_data/Qwen3.8-F \
  --out /root/private_data/strata-dcu/autoround \
  --llama /root/strata-dcu/llama.cpp --workers 4
~~~

准备过程会记录源文件大小、修改时间和配置/索引哈希；源清单改变时拒绝复用缓存。专家打包使用临时文件并按层记录进度，成功后发布完成标记。运行缓存及源模型均位于个人持久化存储；启动脚本仍可通过原缓存路径的软链接访问。

2026-10-22 修正了此前将缓存放在系统盘的问题：迁移 28 个文件、88,593,893,750 字节，完整 rsync 内容校验通过后删除系统盘副本。迁移后 `/root/strata-dcu` 为 939MiB，`/root` 的本地文件占用为 998MiB（均不跟随持久化挂载及缓存软链接）。持久盘只保留整秒修改时间，因此在核对四个 MTP 运行文件 SHA-256 不变后更新了 `ready.json` 的时间字段；checkpoint inventory 和 MTP 缓存检查均通过。迁移记录位于 `/root/private_data/strata-dcu/storage-migration.json`。未重新运行推理性能测试；此前性能数值对应迁移前的缓存存储位置。

### 验证结果与范围

| 检查 | 结果 |
| --- | --- |
| 12 个跨层、跨专家、跨 gate/up/down 投影，约 1966 万权重 | 与 GPTQ FP32 参考解码逐位一致 |
| 合成 INT4 布局与独立 gguf 解码器交叉检查；错误格式拒绝 | 通过 |
| BF16/Q8_1 投影，12 种矩阵/多 token 形状；BF16/F16 行切片转换 | 通过 |
| 外部 BF16 PLE：跨页、EOF、缓存、相对路径、文件大小及边界拒绝 | 通过 |
| gfx936 计算图实际执行、1024 线程 RMSNorm、512 线程 PLE gate 和残差 | 通过，与独立主机参考一致 |
| 默认、one-block、旧版三种采样路径回归 | 3/3 通过 |
| `Reply with exactly OK.` | `OK`，随后 EOS |
| `What is 2+2? Answer only the number.` | `4`，随后 EOS |
| `中国的首都是哪座城市？只回答城市名称。` | `北京`，随后 EOS |

三条完整生成均退出 0，日志无被拒绝的内核启动；cgroup 未记录 OOM。实测约 19,883 个专家驻留显存，其余约 12.08GiB 留在内存，PLE 从磁盘读取。

运行脚本会将 DTK 的 `Launch params` / `larger than launch bounds` 消息视为失败，即使引擎退出码为 0，也不会发布有效生成报告。结束 token 来自原模型的 generation_config，成功结果写入 generation.json。

验证范围为这张 BW/gfx936 上的文本推理及上述算子。未验证视觉、MTP、长上下文、其他硬件或完整质量基准。早期强制 `STRATA_MMVQ_WAVE=0` 的上游逐位一致性诊断未通过；本路径保留默认 wave64 布局。

本地材料：[完整修改补丁](strata-test/autoround.patch)、[启动脚本](strata-test/run-autoround.sh)、[端到端结果](strata-test/autoround-smoke-results.json)、[模型结构检查](strata-test/model-integrity.json)、[采样回归](strata-test/sampler-regression.log)。

### Decode 速度补测（2026-10-21）

单卡 BW/gfx936，现有 AutoRound 缓存，单请求、上下文上限 512、12 个 CPU workers、关闭 MTP、PLE 从磁盘读取。英文故事实际生成 256 tokens，decode 墙钟时间 13,637.1ms，平均 **18.77 token/s**（53.27ms/token）。生成文本连贯，以长度上限结束，退出码 0。此数值排除模型加载和 prefill；本次 prefill 为 41 tokens / 3,069.0ms，首 token 等待 29.92 秒。

这是单次短上下文测试；之前 2-token 回答的 9–11.5 token/s 受启动摊销影响，不能当作持续生成速度。

原始证据：[运行日志](profile_output/autoround-decode256.log)、[生成结果](profile_output/autoround-decode256.json)。仅新增测试日志及本文档记录，未修改推理源码或增加计时插桩。


### 单 token 验证窗口修正（2026-10-21）

初始 18.77 token/s 的入口没有 MTP，却使用默认两 token 验证窗口；日志为 255 个候选全部拒绝，每轮仅提交 1 个 token。相同输入、相同引擎和权重，仅增加 `--mtp-max-t 1`，连续生成 256 tokens 耗时 10,029.1ms，达到 **25.53 token/s**，比初始单次测试提高约 36%。全部 256 个输出 token IDs 与初始测试一致。

`--spec 1` 被 native-pack 入口校验拒绝（退出 2）；因此保留其要求的 `--spec 2`，用 `--mtp-max-t 1` 限制实际运行窗口。`tools/autoround_pack.py` 已采用该配置并同步远程。MTP 仍关闭，PLE 仍从原始磁盘文件读取。

原始对照证据：[单窗口日志](profile_output/autoround-window1.log)、[生成结果](profile_output/autoround-window1.json)、[实际命令](profile_output/autoround-window1-command.json)。初始实测与先前源文件哈希记录保留；本次仅改变入口的窗口配置。

正式启动脚本再次复跑：256 tokens / 9,954.7ms = **25.72 token/s**，退出 0，全部输出 token IDs 仍与初始测试一致。两次单窗口测试为 25.53 / 25.72 token/s；尚未完成交错多轮基准。日志：[启动脚本复跑](profile_output/autoround-window1-wrapper.log)。


### gfx936 BF16/Q8_1 decode 优化（2026-10-21）

在实际单 token 验证窗口下，英文 256-token 生成按基线/新内核/新内核/基线交错实测：基线 24.76、24.49 token/s，新内核 33.56、32.48 token/s，中位数 **24.63 → 33.02 token/s（+34.09%）**。常驻进程的代码/中文请求分别为 36.16、36.87 token/s；正式启动脚本中文 256-token 复跑为 **35.55 token/s**。

本机启动脚本默认设置 `STRATA_BF16_Q8_VEC4=1`。用 `STRATA_BF16_Q8_VEC4=0 bash /root/strata-dcu/run-autoround.sh ...` 切回原内核。新路径仅对 gfx936/wave64 生效；引擎本身仍需显式开启。原权重及 INT4 code/scale 未改变，PLE 仍放磁盘，MTP 仍关闭。变化是 BF16 投影的向量读取与浮点归约顺序，长回答不保证逐 token 相同。

新旧内核各 60 种形状的独立 CPU double 参考检查通过；三个短问答均正确。六个提示的首 token KL 为 3.05e-5–4.80e-4，未完成全面质量或长上下文评测。关闭路由预取没有稳定收益；关闭共享专家跨流反而变慢，均保留原默认。

完整方法、结果、验证边界和文件清单：[decode 优化报告](profile_output/dcu-decode-optimization.md)。


### P1：增强 MTP 下的多列权重复用（2026-10-22）

远程入口已选择持久盘 `/root/private_data/strata-dcu/p1/build/strata`。`STRATA_BF16_Q8_COLS=4` 为 gfx936 的大 BF16 投影复用验证列的权重读取；单列和小矩阵保留原路径，不重新转换权重。测试双方始终开启 `--mtp-mode adaptive`，固定 19,000 专家槽、INT8 KV、512 上下文、12 workers，旧/新/新/旧交错，每个方案两次启动、每种工作负载六个 256-token 请求。

英文、中文、代码 decode 中位数分别由 34.97/40.17/46.46 提升至 37.08/41.34/48.54 token/s（+6.04%/+2.92%/+4.49%）。54 个主对照请求的输出一致。自适应与固定 T=8 的独立检查各有 11 个请求及 13 组首 token logits 与 P0 一致，STOP/恢复、seed、单 token 和合法边界检查通过。

量化去重需要关闭共享跨流，在本组英文/中文上没有净收益；QFUSE 出现非等价输出，因此二者均保持关闭。适用范围、原始数据、开关与回退方法见 [P1 报告](docs/P1_BF16_MULTICOL.md)。


### GPU 权重 ≤50% 的新测试标准（2026-10-22）

后续速度测试统一采用[当前标准](docs/DCU_TEST_STANDARD.md)，主模型 pack、dense、输出头、GPU 专家和 MTP 合计不得超过物理总显存的一半。启动入口和报告均强制核验。当前正式入口选择持久盘 `/root/private_data/strata-dcu/weight50/strata`，默认固定 8,500 个 GPU 主模型专家、16,076 个 RAM 专家（41.394 GiB），增强 MTP adaptive 开启，VEC4/COLS=4。

实测 GPU 权重 33,497,969,152 字节 = 31.197 GiB，占 63.984 GiB 显存的 **48.758%**，低于 31.992 GiB 上限，包含全部 MTP 权重。初始化设备总占用 32.383 GiB；其中权重之外的 1.185 GiB 含 KV/状态/工作区/驱动等，不能当作单独的 KV 用量或峰值。

两次独立启动、每类六次 256-token 生成，BS=1、512 上下文、INT8 KV、12 workers，decode 中位数：**英文 29.48、中文 37.66、代码 38.51 token/s**。短问答、seed、上下文边界、STOP/恢复通过，没有 OOM。旧 19,000 专家配置被新预算拒绝；测试后推理进程均已退出。完整账单、范围和原始材料见[重测报告](profile_output/weight50/README.md)。旧满显存结果仅保留历史条件。


### DFlash2 IPC 接入记录（2026-10-22，历史）

以下是首次 IPC 接入时的环境和结果；当前原生入口见下一节。

已接入 `incoai/Qwen3.8-Flash-Next-DFlash2`，权重通过 ModelScope 下载到持久盘 `/root/private_data/strata-dcu/dflash2/checkpoint`，1,206,140,008 字节，SHA-256 与 HF 发布值一致。HF Mirror 大文件传输过慢后切换；ModelScope 有效传输阶段约 23 MiB/s。没有把大权重或构建放进系统盘，结束时 `/root` 本地占用 954MiB。

正式引擎链接已选用 `/root/private_data/strata-dcu/dflash2/strata-final`，旧 weight50 二进制仍保留。默认草稿方法继续为增强 MTP；用 `bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --prompt 'Reply with exactly OK.' --max-new 16` 显式启用 DFlash2 graph＋自适应候选数。入口禁用 route-tail-skip 和未验证融合；单次生成入口已实际返回 OK。

DFlash2 目前为独立 PyTorch 草稿进程＋Strata 验证器，支持单卡、BS=1、文本、上下文最多 512。主模型 GPU 专家固定 8,300，其余 16,276 在 RAM，全部 GPU 权重 **31.715631 GiB / 49.5678%**（包含草稿输出头副本）。MTP 基线为 8,500 个 GPU 专家、权重 48.758%。

各类三次 256-token 生成的中位数：增强 MTP 英文/中文/代码 **29.47 / 37.52 / 38.26 TPS**；自适应 DFlash2 **27.25 / 31.93 / 41.81 TPS**，分别 **−7.54% / −14.90% / +9.30%**。只在本轮代码任务观察到收益，保留显式开关。

34 个请求、5,140 个输出 token IDs 与同一 8,300 专家布局的逐 token 参考一致；采样、STOP/恢复、单 token 和上下文边界通过，没有 OOM。增强 MTP 的 10 个回归请求与此前 weight50 输出一致。fresh agent 原样复跑真实草稿 witness 通过。测试及草稿进程均已退出。实现、范围、原始数据见 [接入文档](docs/DFLASH2_IPC.md) 和 [测试报告](profile_output/dflash2/README.md)。


### 原生 DFlash2（2026-10-22）

已补齐 Strata 内的原生 CUDA/HIP 草稿路径：完整五层网络、动态因果卷积、top-16 选择器、1–7 候选、七个 GPU graph、按已完成轮次成本选窗，以及 2048 滑动窗口的环形 K/V。特征在 GPU 取 HC 均值并直接注入草稿，复用已有 BF16 输出头和主机映射 embedding，不启动 PyTorch 草稿进程。`--dflash2` 默认使用该路径，未传该开关时默认仍是增强 MTP。

正式二进制 `/root/private_data/strata-dcu/dflash-native/strata-final`，SHA-256 `2aba1cff1291d1af0479188e50050a7b78ff4b619fd90ab5f0e155e4e8f5c42d`，`/root/strata-dcu/build-gfx936/strata` 已指向它。原 IPC、weight50 二进制均保留。草稿原始权重不变，原生派生包在 `/root/private_data/strata-dcu/dflash-native/pack`，标准 AutoRound 缓存通过 `dflash2-native` 符号链接复用它。运行时 Python/DTK/PyTorch 环境未更换。

固定 8300 GPU 专家、16276 RAM 专家，GPU 草稿参数 1206131200 字节，全部 GPU 权重 **32783001600 字节 / 47.717196%**。省去 1271398400 字节重复输出头，未增加专家数；PLE 仍在磁盘。

验收先看 QA：固定七候选的 13 个请求、自适应的 14 个请求，最终 token IDs 全部与关闭草稿的逐 token 参考一致。覆盖问答、算术、格式/JSON、生成 Python 函数的执行、多轮消息、采样复现、单 token、上下文尾部及 STOP/恢复。2100-token 提示越过草稿窗口后返回 ORBIT，后续请求返回 OK。graph/eager、重复/重置、增量/完整重算的独立缓存检查通过。草稿隐藏状态与 PyTorch 的差异仍记录为诊断，未声称两种实现数值等价。

原生自适应、最大七候选，每类单次 256-token 生成的速度记录：英文 **22.26**、中文 **24.33**、代码 **33.06 TPS**。四个请求（包括预热）的最终输出与此前逐 token 参考一致。本轮没有超过增强 MTP；该单轮记录不作统计显著性结论。只验证了 HIP/gfx936 的实际运行，CUDA 对新增 `.cu` 做了编译检查，完整 CUDA 引擎与其他后端未运行验证。实现与范围见 [原生文档](docs/DFLASH2_NATIVE.md)，原始结果见 [验收记录](profile_output/dflash-native/README.md)。


### 原生 DFlash2 清理与重新验证（2026-10-23）

工作区已移除未提交的专家预测、训练/TTT、临时分流和生命周期插桩，保留原生 DFlash2。HIP 完整构建通过；新增 DFlash CUDA 翻译单元使用 CUDA 13.3 / g++-15、sm_80 编译通过。重新执行 14 项端到端 QA、2100-token 长提示及请求恢复，以及原生 graph/eager、重复/重置、增量和环形缓存检查，均通过。增强 MTP 的 4 项回归通过。采样 token 完全一致只作为记录，答案、格式和代码执行作为功能验收依据。

清理后二进制：`/root/private_data/strata-dcu/dflash-native/cleanup-20261023/strata`，SHA-256 `745c912b923cec9696517e9a898609c1e4483fb4baca3b411f1f52c0dc8a154a`；`/root/strata-dcu/build-gfx936/strata` 已指向它。原生 DFlash2 的 8300 常驻专家及草稿权重加上已有的 16 个临时专家槽后，完整 GPU 权重为 32827238400 字节（47.781584%）。原始结果见 `profile_output/dflash-native/cleanup-20261023/`。本轮没有重新进行性能对比，不新增速度提升结论。

### Q4_0 多 token CPU 与 DFlash2 小 batch 投影（2026-10-23）

实现与验收见 [DCU 内核优化](docs/DCU_KERNEL_OPTIMIZATIONS.md)。本机入口启用 `STRATA_CPU_Q4_MT=1` 和 `STRATA_DFLASH_SMALL_GEMV=1`，可分别设为 0 对照。固定原有 8500/8300 个专家与 50% 权重预算：增强 MTP 英文/中文/代码三次中位数为 31.59/38.88/40.29 TPS，原生 DFlash2 为 29.89/33.19/43.58 TPS。GPU 权重账单不变。14 个真实 QA、代码执行、STOP/恢复、九个 eager/graph 缓存用例及 2100-token 输入后恢复通过。正式二进制与源码哈希、测量范围和原始数据见文档及 `profile_output/q4-multitoken-20261023/`。

### MTP 与 DFlash2 联合草稿（2026-10-23）

标准引擎更新到 `/root/private_data/strata-dcu/joint-draft-20261023/strata-final`，SHA-256 `1551c2b1cc1624a29ab9b17f2276599c2a35648c58babc417e2c6ad98947b1b1`。默认仍是增强 MTP；`--dflash2 ... --dflash-mtp-prefix 2` 才联合使用。MTP 产生短前缀，原生 DFlash2 在该前缀条件下补后续候选，统一验证并按接受结果推进两个缓存。固定 8300 GPU 专家、其余 16276 在 RAM；全部 GPU 权重 34,195,377,152 字节（49.772975%），PLE 仍在盘上。

14 项语义/代码/采样 QA、强制拒绝后的同组 QA、STOP/恢复、15 项 eager/graph 前缀和缓存用例、2100-token 输入及后续恢复通过。独立 DFlash2 九项原用例输出与前版逐字节相同。联合模式三次中位数为英文/中文/代码 23.78/27.85/35.28 TPS，均低于本日之前的单独路径；没有切换默认草稿。使用、测量限制与原始记录见 [联合文档](docs/JOINT_DRAFT.md) 和 [验收结果](profile_output/joint-draft-20261023/README.md)。

## HC 修复与 lookup 兼容（2026-10-24）

当前候选已部署到 `/root/private_data/strata-dcu/lookup-compat-20261023/strata-final`，
公开引擎链接指向此文件；SHA256 为 `1c5b8a268585ab62bfe53a098c114b7e645c8befa91384d58c645fec9a8ff396`。
严格 HC 自检恢复通过，入口默认 staged。增强 MTP 的英文/中文/代码三次中位数为
33.55 / 42.21 / 43.13 TPS；原生 DFlash2 为 30.13 / 35.12 / 46.21 TPS。
这些是 2026-10-23 的固定专家数对照结果；重启后接续完成长上下文、拒绝和 IPC 功能检查。

缓存交换、PCIe 比例和搬运方式、锁页内存、suffix lookup 都已开放配置。
默认仍是增强 MTP，交换/分流/lookup 关闭；权重限制和专家槽数保持不变。
具体用法与结果见 [DCU 运行选项](docs/DCU_RUNTIME_OPTIONS.md)；公开入口独立复核已通过，三种模式均正确执行查找草稿；进程全部退出，GPU 回到 2 MiB。

## 层内交接更新（2026-10-24）

公开引擎链接现指向持久盘 `/root/private_data/strata-dcu/layer-scheduling-20261024/v9/strata-final`，
SHA256 `a2ac48e5fe51fb18d93db50de4bfdb9fdf1d3682877284b1a7b2d4ff217336c2`。
环境依赖未变。增强 MTP 入口默认开启 GPU 常驻专家独立安排、CPU 按专家推进、结果提前交回；
原生 DFlash2/联合模式仍显式选择这些选项。临时搬运及层末休眠默认关闭。

固定 8500 个常驻专家、全部 GPU 权重 33542205952 字节，三次中位数英文/中文/代码为
34.65 / 42.33 / 45.03 TPS；同日原二进制为 33.27 / 41.23 / 40.63。
原生及联合模式仍固定 8300 个专家，三种模式都满足权重 ≤50%。PLE、模型、构建和原始结果继续放持久盘。
功能 QA、强制搬运、联合草稿拒绝后恢复、底层并发检查和三种公开入口独立复核通过。
结束时 GPU 2 MiB，无推理进程；系统目录 `/root/strata-dcu` 895 MiB、`/root/.cache` 59 MiB。
实现及完整范围见 [层内交接](docs/DCU_LAYER_SCHEDULING.md) 和 [本轮记录](profile_output/layer-scheduling-20261024/README.md)。


## 模型权重与 MTP 更新（2026-10-24）

本机公开链接现指向 `/root/private_data/strata-dcu/model-specific-20261024/final-v1/strata-final`，
SHA256 `3ae138ead7fdbab0324bb17b423022d32c497eb18e3fb5aa82901011b98c9450`。
环境依赖沿用 dflash2@52f8bc84，无新增安装。增强 MTP 默认 HC Q8＋Q8非专家投影/输出头＋发布的蒸馏MTP，
筛选头保留开关、默认关闭；原生DFlash2/联合默认权重继续沿用此前配置。
选择文件在持久盘 `model-specific-20261024/selected-model.json`，设 `STRATA_DCU_MODEL_DEFAULTS=0` 恢复此前权重选择。

专家仍固定8500/8300，主专家INT4不改；选定MTP全部模型权重29373360352B，低于34351349760B上限。
三任务各三次decode中位数：英文36.20、中文48.51、代码56.98TPS；同组旧二进制33.72、42.01、44.62。
五配置18项QA及恢复通过，原生Q8投影、联合强制拒绝QA通过，三个公开入口独立复核通过。
后续诊断期间旧版和新版都出现CPU专家明显减速，短英文约9TPS，原因未定位；不能保证当前持续达到前表速度。

PLE、Q8派生权重、MTP、构建和日志均在持久盘。结束时GPU2MiB，无推理/构建进程，OOM0；
系统目录/root/strata-dcu为896MiB、/root/.cache为59MiB。
完整实现、失败记录和测量范围见 [专项文档](docs/DCU_MODEL_OPTIMIZATIONS.md) 和 [本轮报告](profile_output/model-specific-20261024/README.md)。


2026-10-25 已定位上述后段骤降的主要原因：默认固定CPU0–12遭遇宿主用户态争用。
同一默认权重进程的256-token英文生成，原核心9.72TPS，移到CPU16–28为33.24TPS，移回12.18TPS；
全部decode阶段无限流，专家大区页分布保持相同，线程排队等待随换核消失/恢复。
本轮未改生产默认绑核或引擎；最终GPU2MiB、所有推理进程退出。见 [原因与原始证据](profile_output/cpu-slowdown-20261025/README.md)。


2026-10-25 CPU独占权限检查：容器cgroup2只读，cpuset为member且不可写，
无CAP_SYS_NICE/CAP_SYS_ADMIN，RLIMIT_RTPRIO/NICE为0，自有子进程nice(-1)实测EPERM。
因此当前无法在容器内申请宿主独占核心，需要平台侧分配；未修改生产入口或调度优先级。
证据见 [权限检查](profile_output/cpu-reservation-20261025/README.md)。


## 自动选空闲核心重测（2026-10-25）

在模型载入后、首次请求前，按内核同胞掩码和三秒CPU负载采样选13个核心，
优先同节点及靠近专家内存；后续配置重新检查负载，安静才沿用原组。
增强MTP各任务三次256token中位数：原权重32.90/41.29/44.32TPS；
原权重加筛选29.53/41.77/44.26；当前Q8+HCQ8+蒸馏组合33.84/43.27/53.88；
当前组合加筛选31.97/44.23/51.25。四配置18项QA、代码执行及STOP/恢复全部通过。
专家仍8500/16076，权重均<=50%，筛选继续默认关闭。
这是外部bench的自动选核重测，生产wrapper默认未切换；不构成独占。
结束GPU2MiB、无推理进程、OOM0，系统目录896MiB/cache59MiB。
原始记录及方法见 [重测报告](profile_output/idle-core-retest-20261025/README.md)。

### CPU INT4输入修正复用（2026-10-25）

环境仍为dflash2@52f8bc84，没有安装或升级依赖。
当前生产引擎改为持久盘`/root/private_data/strata-dcu/q4-input-bias-20261025/strata-final`，
SHA256为`4fc8304ec17c7e6b0df4a843f7f60ebe21934e6a7e88004a358e6b2356edb8cd`。
CPU Q4_0按输入复用整数修正，并支持单位置；已有ISA检查及`STRATA_CPU_Q4_MT=1`仍是前提。
本机增强MTP的selected-model.json选择`cpu_q4_input_bias=true`，
可用`STRATA_CPU_Q4_INPUT_BIAS=0`覆盖；原生/联合模式保持显式选择。
`STRATA_DCU_MODEL_DEFAULTS=0`会同时停用模型配置中的该默认值。

固定8500/16076专家与原Q8/蒸馏MTP权重，权重29373360352B，仍不超过50%。
空闲选核下四次ABBA冷启动，每类每版合并六次256token样本：
旧版37.07/49.18/56.87TPS，新版37.90/48.35/58.61TPS（英文/中文/代码），
分别+2.24%/-1.71%/+3.06%，小幅混合收益，不声称统计显著。
47124项标量参考、完整专家、合成及真实专家线程池552项各自通过；
四轮增强MTP18QA+STOP/恢复及原生/联合强制拒绝的18QA+lookup/恢复通过。
自动选核仍是外部测速方法，生产CPU绑定没有在本轮改动。
细节、全部单次值和公开入口复核记录见[本轮材料](profile_output/q4-input-bias-20261025/README.md)。
独立公开入口复核已原样执行一次、退出0，四项默认/覆盖/原生/联合检查通过。
配置中的当前引擎摘要已更新，旧权重选择来源另存；结束GPU2MiB，无引擎/编译/队列进程，OOM0。

### CPU专家输出直接合并（2026-10-25）

环境仍为dflash2@52f8bc84，没有安装或升级依赖。公开引擎现指向持久盘
`/root/private_data/strata-dcu/cpu-result-direct-20261025/strata-final`，SHA256为
`81f79177f7e9b15a1d4506bed0211db95b369594868f51bcc19e64fa4017ce9b`。
新开关`STRATA_CPU_RESULT_DIRECT=1`直接读取CPU的FP32专家输出并与GPU输出合并；
原完成信号、第一层PLE条件、路由和专家数保持不变。其他执行条件不满足时回退。

固定增强MTP、8500/16076专家、Q8非专家矩阵与HC、蒸馏MTP、CPU Q4输入修正，
四次ABBA冷启动、每类每版六次256token样本，旧版英文/中文/代码中位数
34.46/44.88/53.24TPS，新版34.03/43.89/54.81TPS；三项比值几何平均约−0.19%。
未建立整体提速，机器配置选择`cpu_result_direct=false`，显式环境值1可覆盖。
CPU Q4输入修正仍默认开启，模型权重与生产选核不变。

四轮18项QA+STOP/恢复、原生/联合强制拒绝/显式关闭/设备工作表回退各20项QA通过；
生产接口960组数值与延迟握手、原路由/合并和1601组工作表检查通过。
新路径全48层记录已保存；该运行不用来报告正常TPS。
说明与原始材料见[本轮材料](profile_output/cpu-result-direct-20261025/README.md)。

独立公开入口修正版本原样执行一次通过：默认/覆盖/原生/联合四项及GPU idle均通过，
此前一次验证脚本日志字符串误报已修正并保留失败材料。最终GPU2MiB，无本轮引擎/编译/测试进程，
OOM和OOM kill均0；系统目录896MiB/cache59MiB，模型及构建仍在持久盘。

### gfx936 原生整数点积与 Q8 连续读取（2026-10-25）

当前入口指向持久盘 `/root/private_data/strata-dcu/gfx936-dot4-20261025/load64/strata-final`，
SHA256 `ac63eb49bfa614aed8365e63a1d7c7af85ef015f4d3a681ed029477aac29ee42`。
补齐gfx936原生有符号整数点积分发，并明确以64位复制读取Q8相邻权重字；Q6局部缩放固定乘加舍入。
无新运行参数。模型、量化格式、8500/8300专家数、半显存规则、增强MTP和生产绑核均保持原配置。

同一组空闲物理核心、同近端专家页的补充对照：英文36.81→38.01、中文47.84→47.86、代码57.52→58.74TPS，
三类比例的几何平均+1.80%。每项每版本6次256-token生成，无全层标记；不是统计显著性结论。
原始ABBA中一轮新版落在非空闲/远端NUMA节点、一轮旧版更换了核组，均保留，另补B3/A3匹配核组。
只改整数指令的初版反而约慢5.6%，未启用；完整记录没有删除。

增强MTP、原生DFlash2和强制拒绝联合模式的完整回答/可执行代码QA通过，包含取消恢复与lookup。
1310720点积边界/随机检查和8项GPU测试通过。通用专家检查中的旧FP16转换逐位差异已用旧库复现，
输出与新库完全一致；未宣称该完整测试命令通过。CUDA投影编译单元通过，其他硬件未运行。
实现说明见 [DCU_GFX936_DOT4.md](docs/DCU_GFX936_DOT4.md)，
证据见 `profile_output/gfx936-dot4-20261025/`，其中 `load64/` 为最终版本。

新鲜独立检查者按文档执行一次，增强MTP、原生DFlash2、联合模式公开入口均通过，退出0。
最终审计通过；GPU2MiB、无剩余推理/测试进程，OOM/kill为0，系统目录892MiB、cache57MiB。
