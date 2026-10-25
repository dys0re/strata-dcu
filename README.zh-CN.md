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
| 模型准备 | AutoRound INT4 专家导入、从磁盘读取 PLE |
| 草稿生成 | 增强自适应 MTP、原生 DFlash2、MTP + DFlash2 联合模式 |
| CPU 专家 | AVX-512/VNNI Q4 计算、多位置复用权重解包与输入修正 |
| GPU 投影 | gfx936 原生整数点积和 Q8 连续权重读取 |
| 实验性 MMAC | 分块 Q8 权重、DCU 矩阵指令和计算维分段合并，通过 `STRATA_Q8_MMAC=1` 开启 |

当前 MMAC 接入覆盖符合条件的大 Q8 投影和主模型输出头。这个实验版本保留执行权重副本，
并将副本完整计入 GPU 权重账单。投影融合和单一执行权重布局仍在开发计划中。

## 已验证的环境

运行验证使用单张 BW/gfx936 DCU，显存 65,520 MiB；CPU 为海光处理器，配额相当于 15 核计算时间，
内存上限 59 GiB。软件环境为 Linux、DTK 26.04 和 HYHAL。
本分支的实测范围是文本推理，没有对上游支持的所有显卡、操作系统和模型重新验证。

测试时，主模型、草稿模型和执行副本等全部 GPU 模型权重合计不超过**物理显存的 50%**。
增强 MTP 固定 8,500 个 GPU 专家，原生 DFlash2 和联合模式固定 8,300 个；
PLE、模型、缓存和大型构建文件放在持久化存储。
详见[测试标准](docs/DCU_TEST_STANDARD.md)和[环境记录](envs.md)。

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

[测试机 wrapper](strata-test/run-autoround.sh) 包含实测使用的环境选项，其路径需要按自己的机器调整。
DFlash2 的检查点、准备过程和运行方式见[原生 DFlash2](docs/DFLASH2_NATIVE.md)与[联合模式](docs/JOINT_DRAFT.md)。

MMAC 需要准备好的 **Q8 非专家投影权重**，通过 `--dense-gguf` 选择，并使用包含 gfx936 MMAC 实现的引擎。
权重变体准备方法见 [DCU 模型优化](docs/DCU_MODEL_OPTIMIZATIONS.md)。运行时设置 `STRATA_Q8_MMAC=1`；
该实验路径默认关闭。

## 实测结果

一条预热后的 LRU Cache 生成样例：每种模式生成 384 tokens，增强 MTP、上下文 512、
12 个 CPU workers 加主控线程，GPU 专家均为 8,500 个。

| 指标 | MMAC 关闭 | MMAC 开启 |
| --- | ---: | ---: |
| Decode 速度 | 58.95 tokens/s | 61.60 tokens/s |
| 含 prefill 的请求耗时 | 6.99 秒 | 6.69 秒 |
| 含执行副本的 GPU 模型权重 | 29.37 GB | 32.86 GB |

这是**单样例观察**，不是平均性能提升结论。两次生成均达到 token 上限；这条样例不算完整的可执行代码 QA。
独立的四位置 Q8 算子原型测得约 1.9 倍速度，不等于端到端推理达到 1.9 倍。
原始实验输出保存在本地，不加入新的源码提交。

## 开发说明

- [gfx936 整数点积与 Q8 读取实现](docs/DCU_GFX936_DOT4.md)
- [Q8 MMAC 方案与原型结果](docs/DCU_Q8_MMAC_PLAN.md)
- [运行选项](docs/DCU_RUNTIME_OPTIONS.md)
- [原生 DFlash2](docs/DFLASH2_NATIVE.md)
- [测试标准](docs/DCU_TEST_STANDARD.md)

## 许可与署名

采用 [MIT 许可证](LICENSE)。Strata 由 Niko1221 和 Strata contributors 创建。
本分支保留上游许可证和署名；第三方许可说明见 [third_party](third_party/)。
