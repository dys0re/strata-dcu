# strata-dcu

[English](README.md)

这是 **[Niko1221/Strata](https://github.com/Niko1221/Strata)** 的 DCU 分支，面向
**海光 BW / gfx936**，研究和优化 Qwen3.8-Flash-Next 在 GPU 与 CPU 专家混合执行时的推理性能。

原始引擎、桌面应用和模型支持来自 Strata。上游安装包和应用说明见
[原项目](https://github.com/Niko1221/Strata)；使用本分支的 DCU 修改需要从源码构建。

## 本分支的工作

| 方向 | 实现 |
| --- | --- |
| DCU 运行环境 | DTK/HYHAL 集成、gfx936 wave64 兼容与计算图执行修正 |
| 模型准备 | AutoRound INT4 导入、逐层专家恢复校验、磁盘 BF16/IQ4_NL PLE |
| 草稿生成 | 增强自适应 MTP、原生 DFlash2、MTP + DFlash2 联合模式 |
| CPU 专家 | AVX-512/VNNI Q4 计算、多位置复用权重解包与输入修正 |
| GPU 投影 | gfx936 原生整数点积和 Q8 连续权重读取 |
| Q8 执行 | decode、prefill、MTP 共用一份 MMAC 权重，跨验证窗口固定累加策略 |
| 调度 | 融合路由发布、CPU miss 专用图、按工作量分块的 CPU 专家队列 |

MMAC 覆盖符合条件的 Q8 投影和输出头，不再常驻完整的第二份 GPU 权重。
通用引擎的可选优化仍需显式开启；验证机器的 wrapper 单独选择模型和内核配置。

## 已验证的环境

最近 worker 测试使用 Linux、DTK 26.04/HYHAL，单张 BW/gfx936（65,520 MiB 显存），
32 个可用 CPU、121.5 GiB 内存。早期 15 核配额/59 GiB 实例的结果属于不同环境。
本分支的实机验证范围为 gfx936 单请求文本推理。

全部 GPU 模型权重（含草稿和执行布局）限制在**物理显存的 50%**以内。
增强 MTP 固定 8,500 个 GPU 专家，原生 DFlash2/联合实验固定 8,300 个。
详见[测试标准](docs/DCU_TEST_STANDARD.md)和[环境与准备说明](envs.md)。

## 构建

克隆本分支：

```bash
git clone https://github.com/dys0re/strata-dcu.git
cd strata-dcu
```

安装好 DTK/HYHAL，并准备兼容的 llama.cpp 源码后：

```bash
source /opt/dtk/env.sh
source /opt/hyhal/env.sh

cmake -S . -B build-gfx936 -G Ninja \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx936 \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_GGML_DIR=/path/to/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++
cmake --build build-gfx936 --target strata -j4
```

`STRATA_HIP_GFX906` 是 wave64 兼容后端的名称，这里实际编译目标是 `gfx936`。
已验证的 llama.cpp 版本记录在 [third_party/ggml/VERSION.txt](third_party/ggml/VERSION.txt)。
Python、转换工具依赖和模型缓存准备过程见 [envs.md](envs.md)。

## 使用已准备的模型

`tools/autoround_pack.py` 提供模型准备与运行入口。模型和缓存准备完成后：

```bash
python3 tools/autoround_pack.py run \
  --model /path/to/Qwen3.8-F \
  --out /persistent/autoround \
  --llama /path/to/llama.cpp \
  --engine ./build-gfx936/strata \
  --expert-cache 8500 \
  --mtp-mode adaptive \
  --prompt 'Reply with exactly OK.' --max-new 32
```

[历史测试机 wrapper](strata-test/run-autoround.sh) 包含实测使用的环境选项，其路径需要按自己的机器调整。
DFlash2 的检查点、准备过程和运行方式见[原生 DFlash2](docs/DFLASH2_NATIVE.md)与[联合模式](docs/JOINT_DRAFT.md)。

MMAC 需要准备好的 **Q8 非专家投影权重**，通过 `--dense-gguf` 选择，并使用包含 gfx936 MMAC 实现的引擎。
权重变体准备方法见 [DCU 模型优化](docs/DCU_MODEL_OPTIMIZATIONS.md)。运行时设置 `STRATA_Q8_MMAC=1 STRATA_Q8_MMAC_STABLE=1`；
该实验路径默认关闭。

## 实测与当前状态

最近队列对照（2026-11-07）：GPU0、12 workers 加主控、8,500 GPU 专家，Q8 dense/head 与 HC、
stable MMAC、蒸馏 adaptive MTP/maxT5、磁盘 IQ4_NL PLE、INT8 KV、context512。
四次独立启动按 ABBA 交错，预热后每模式每领域六次 256-token 请求；decode 排除加载和 prefill。

| 工作负载 | 原 CPU 队列 | 优化 CPU 队列 |
| --- | ---: | ---: |
| 英文 | 37.42 tokens/s | 37.30 tokens/s |
| 中文 | 51.33 tokens/s | 51.21 tokens/s |
| 代码 | 62.63 tokens/s | 66.72 tokens/s |

表内为单机描述性中位数：代码 +6.54%，三域速度比几何平均 +1.95%，不代表通用或统计显著提速。
当前机器入口开启 CPU 队列优化，关闭 PCIe 分流；通用引擎保持显式选择。
全部 GPU 模型权重为 29,373,360,352 B（27.36 GiB）；与旧双份 MMAC 实现相比，
统一 Q8 布局节省 3.24 GiB，专家数不变。

已有验收记录包括 HIP/CUDA sm80 构建、552 项线程池比较、延迟唤醒测试、18 项模型 QA、
生成代码执行与取消恢复。CUDA 仅编译；未完成 NVIDIA 实机、全面模型质量或长上下文评测。
专家恢复另完成 48 层、432 项参考比较及迁移前后生成检查。
原始实验材料保存在本地 `profile_output/`，不随源码提交。

## 文档导航

- [环境与模型准备](envs.md)
- [模型变体：Q8、蒸馏 MTP、IQ4_NL PLE](docs/DCU_MODEL_OPTIMIZATIONS.md)
- [Q8 MMAC 布局与复现性](docs/DCU_Q8_MMAC_PLAN.md)
- [CPU/GPU 内核参考](docs/DCU_KERNEL_OPTIMIZATIONS.md)
- [CPU 专家队列](docs/DCU_HETERO_SCHEDULING.md)与[图调度](docs/DCU_GRAPH_SCHEDULING.md)
- [运行选项](docs/DCU_RUNTIME_OPTIONS.md)与[测试标准](docs/DCU_TEST_STANDARD.md)
- [原生 DFlash2](docs/DFLASH2_NATIVE.md)与[联合草稿](docs/JOINT_DRAFT.md)

## 许可与署名

采用 [MIT 许可证](LICENSE)。Strata 由 Niko1221 和 Strata contributors 创建。
本分支保留上游许可证和署名；第三方许可说明见 [third_party](third_party/)。
