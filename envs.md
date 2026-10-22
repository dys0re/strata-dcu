# SCNet DCU 开发环境

当前速度测试遵循[GPU 权重占用不超过 50% 的标准](docs/DCU_TEST_STANDARD.md)。主模型、输出头和 MTP 一起计入；此前满显存/19,000 专家配置的速度保留为历史记录。

最后更新：2026-10-22。本文记录远程实例的配置、环境加载和联网测试结果；未包含登录密码或代理认证凭据。

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


### DFlash2 实验接入（2026-10-22）

已接入 `incoai/Qwen3.8-Flash-Next-DFlash2`，权重通过 ModelScope 下载到持久盘 `/root/private_data/strata-dcu/dflash2/checkpoint`，1,206,140,008 字节，SHA-256 与 HF 发布值一致。HF Mirror 大文件传输过慢后切换；ModelScope 有效传输阶段约 23 MiB/s。没有把大权重或构建放进系统盘，结束时 `/root` 本地占用 954MiB。

正式引擎链接已选用 `/root/private_data/strata-dcu/dflash2/strata-final`，旧 weight50 二进制仍保留。默认草稿方法继续为增强 MTP；用 `bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --prompt 'Reply with exactly OK.' --max-new 16` 显式启用 DFlash2 graph＋自适应候选数。入口禁用 route-tail-skip 和未验证融合；单次生成入口已实际返回 OK。

DFlash2 目前为独立 PyTorch 草稿进程＋Strata 验证器，支持单卡、BS=1、文本、上下文最多 512。主模型 GPU 专家固定 8,300，其余 16,276 在 RAM，全部 GPU 权重 **31.715631 GiB / 49.5678%**（包含草稿输出头副本）。MTP 基线为 8,500 个 GPU 专家、权重 48.758%。

各类三次 256-token 生成的中位数：增强 MTP 英文/中文/代码 **29.47 / 37.52 / 38.26 TPS**；自适应 DFlash2 **27.25 / 31.93 / 41.81 TPS**，分别 **−7.54% / −14.90% / +9.30%**。只在本轮代码任务观察到收益，保留显式开关。

34 个请求、5,140 个输出 token IDs 与同一 8,300 专家布局的逐 token 参考一致；采样、STOP/恢复、单 token 和上下文边界通过，没有 OOM。增强 MTP 的 10 个回归请求与此前 weight50 输出一致。fresh agent 原样复跑真实草稿 witness 通过。测试及草稿进程均已退出。实现、范围、原始数据见 [接入文档](docs/DFLASH2_IPC.md) 和 [测试报告](profile_output/dflash2/README.md)。
