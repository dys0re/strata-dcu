# strata-dcu

[简体中文](README.zh-CN.md)

A DCU-focused fork of **[Niko1221/Strata](https://github.com/Niko1221/Strata)** for running
Qwen3.8-Flash-Next with GPU-resident and CPU-resident experts. This branch develops and tests
inference paths for the **Hygon BW / gfx936** platform.

The original engine, desktop application and model support come from Strata. For upstream
installers and application documentation, see the [upstream project](https://github.com/Niko1221/Strata).
Build this fork from source to use its DCU changes.

## What this fork adds

| Area | Work in this branch |
| --- | --- |
| DCU runtime | DTK/HYHAL integration, gfx936 wave64 compatibility and graph execution fixes |
| Model preparation | AutoRound INT4 expert import and disk-backed PLE access |
| Drafting | Enhanced adaptive MTP, native DFlash2 and joint MTP + DFlash2 |
| CPU experts | AVX-512/VNNI Q4 computation, multi-position weight decoding reuse and input correction reuse |
| GPU projections | Native gfx936 integer dot products and coalesced Q8 weight reads |
| Experimental MMAC | Tiled Q8 weights, DCU matrix instructions and split-K reduction, enabled with `STRATA_Q8_MMAC=1` |

The MMAC path currently covers eligible large Q8 projections and the main output head. It retains
execution weight copies for this experimental integration and counts them in the GPU weight budget.
Projection fusion and a single execution weight layout remain development work.

## Validated environment

Runtime testing has used one BW/gfx936 DCU with 65,520 MiB VRAM, a Hygon CPU allocation of 15 CPU
time equivalents, and a 59 GiB RAM limit. The software stack is Linux with DTK 26.04 and HYHAL.
These fork-specific measurements cover text inference; support inherited from upstream has not
been revalidated across every GPU, operating system or model.

Test settings keep all GPU model weights, including draft weights and execution copies, within
**50% of physical VRAM**. Expert counts are fixed at 8,500 GPU experts for enhanced MTP and 8,300
for native DFlash2 or joint mode. PLE and large model/cache/build files use persistent storage.
See [the test standard](docs/DCU_TEST_STANDARD.md) and [environment notes](envs.md).

## Build

Clone this fork:

```bash
git clone https://github.com/dys0re/strata-dcu.git
cd strata-dcu
```

With DTK/HYHAL installed and a compatible llama.cpp checkout available:

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

`STRATA_HIP_GFX906` names the wave64 compatibility backend; the target architecture here is
`gfx936`. The validated llama.cpp revision is recorded in [third_party/ggml/VERSION.txt](third_party/ggml/VERSION.txt).
Python/conversion dependencies and the prepared-model workflow are described in [envs.md](envs.md).

## Run with a prepared model

`tools/autoround_pack.py` provides model preparation and inference. After preparing the model/cache:

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

The [host wrapper](strata-test/run-autoround.sh) documents the settings used on the validation
machine; its paths are specific to that machine. See [native DFlash2](docs/DFLASH2_NATIVE.md) and
[joint drafting](docs/JOINT_DRAFT.md) for their checkpoints, preparation and invocation.

MMAC requires prepared **Q8 non-expert projections**, selected with `--dense-gguf`, and a build
containing the gfx936 MMAC implementation. Set `STRATA_Q8_MMAC=1` for that run. Preparing those
weight variants is documented in [DCU model optimizations](docs/DCU_MODEL_OPTIMIZATIONS.md).
MMAC is disabled by default.

## Measurements

One warmed LRU Cache generation sample, 384 output tokens per mode, enhanced MTP, context 512,
12 CPU workers plus the host thread, and the same 8,500 GPU experts:

| Metric | MMAC off | MMAC on |
| --- | ---: | ---: |
| Decode throughput | 58.95 tokens/s | 61.60 tokens/s |
| Request time, including prefill | 6.99 s | 6.69 s |
| GPU model weights, including execution copies | 29.37 GB | 32.86 GB |

This is a **single-sample observation**, not an average speedup claim. Both generations reached
the token limit; this sample is not a complete executable-code QA result. Isolated four-position
Q8 operator prototypes measured about 1.9× speed, which does not imply 1.9× end-to-end throughput.
Raw experiment outputs are kept locally and are not included in new commits.

## Development notes

- [gfx936 dot4 and Q8 load implementation](docs/DCU_GFX936_DOT4.md)
- [Q8 MMAC design and prototype results](docs/DCU_Q8_MMAC_PLAN.md)
- [Runtime options](docs/DCU_RUNTIME_OPTIONS.md)
- [Native DFlash2](docs/DFLASH2_NATIVE.md)
- [Test standard](docs/DCU_TEST_STANDARD.md)

## License and credit

[MIT](LICENSE). Strata was created by Niko1221 and the Strata contributors. This fork retains
the upstream license and attribution. Third-party notices remain under [third_party](third_party/).
