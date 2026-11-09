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
| Model preparation | AutoRound INT4 import, verified per-layer expert recovery and disk-backed BF16/IQ4_NL PLE |
| Drafting | Enhanced adaptive MTP, native DFlash2 and joint MTP + DFlash2 |
| CPU experts | AVX-512/VNNI Q4 computation, multi-position weight decoding reuse and input correction reuse |
| GPU projections | Native gfx936 integer dot products and coalesced Q8 weight reads |
| Q8 execution | One MMAC weight allocation shared by decode, prefill and MTP; stable split-K across verification widths |
| Scheduling | Fused route publication, CPU miss graphs and cost-weighted CPU expert queues |

The MMAC path covers eligible Q8 projections and the output head without retaining a full second
GPU weight copy. Optional paths remain opt-in in the generic engine; the validated worker wrapper
selects its own model and kernel settings.

## Validated environment

The latest worker tests used Linux, DTK 26.04/HYHAL, one BW/gfx936 DCU with 65,520 MiB VRAM,
32 available CPUs and 121.5 GiB RAM. Earlier tests used a different 15-CPU / 59 GiB instance.
The fork-specific runtime evidence covers single-request text inference on gfx936.

All GPU model weights, including drafting and execution layouts, stay within **50% of physical VRAM**.
Enhanced MTP uses 8,500 GPU experts; native DFlash2/joint experiments use 8,300.
See [the test standard](docs/DCU_TEST_STANDARD.md) and [environment/setup notes](envs.md).

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

The historical [host wrapper](strata-test/run-autoround.sh) documents the settings used on the validation
machine; its paths are specific to that machine. See [native DFlash2](docs/DFLASH2_NATIVE.md) and
[joint drafting](docs/JOINT_DRAFT.md) for their checkpoints, preparation and invocation.

MMAC requires prepared **Q8 non-expert projections**, selected with `--dense-gguf`, and a build
containing the gfx936 MMAC implementation. Set `STRATA_Q8_MMAC=1 STRATA_Q8_MMAC_STABLE=1` for that run. Preparing those
weight variants is documented in [DCU model optimizations](docs/DCU_MODEL_OPTIMIZATIONS.md).
MMAC is disabled by default.

## Measurements and current status

Latest queue comparison (2026-11-07): GPU0, 12 workers plus the host thread, 8,500 GPU experts,
Q8 dense/head and HC, stable MMAC, distilled adaptive MTP/maxT5, disk-backed IQ4_NL PLE,
INT8 KV and context512. Four fresh processes in ABBA order; six 256-token requests per domain
and mode after warmup. Decode excludes model loading and prefill.

| Workload | Original CPU queue | Optimized CPU queue |
| --- | ---: | ---: |
| English | 37.42 tokens/s | 37.30 tokens/s |
| Chinese | 51.33 tokens/s | 51.21 tokens/s |
| Code | 62.63 tokens/s | 66.72 tokens/s |

These are medians from descriptive single-host measurements: code +6.54%, three-domain geometric
mean +1.95%. They do not establish a general or statistically significant speedup. The worker
wrapper selects the CPU queue optimization and disables PCIe offload; generic engine defaults stay opt-in.
GPU model weights total 29,373,360,352 B (27.36 GiB). The unified Q8 layout saves 3.24 GiB at the same
expert count compared with the previous two-copy MMAC implementation.

Recorded checks passed HIP and CUDA sm80 builds, 552 CPU pool comparisons, delayed-wakeup tests,
18 model QA cases, generated-code execution and cancellation/recovery. CUDA was compile-only;
no NVIDIA runtime, broad model-quality or long-context qualification is claimed. AutoRound recovery
also verified 48 expert layers with 432 reference comparisons and generation before/after migration.
Raw experiment outputs remain local under `profile_output/`.

## Documentation

- [Environment and model preparation](envs.md)
- [Model variants: Q8, distilled MTP and IQ4_NL PLE](docs/DCU_MODEL_OPTIMIZATIONS.md)
- [Q8 MMAC layout and reproducibility](docs/DCU_Q8_MMAC_PLAN.md)
- [CPU/GPU kernel reference](docs/DCU_KERNEL_OPTIMIZATIONS.md)
- [CPU expert queues](docs/DCU_HETERO_SCHEDULING.md) and [graph scheduling](docs/DCU_GRAPH_SCHEDULING.md)
- [Runtime options](docs/DCU_RUNTIME_OPTIONS.md) and [test standard](docs/DCU_TEST_STANDARD.md)
- [Native DFlash2](docs/DFLASH2_NATIVE.md) and [joint drafting](docs/JOINT_DRAFT.md)

## License and credit

[MIT](LICENSE). Strata was created by Niko1221 and the Strata contributors. This fork retains
the upstream license and attribution. Third-party notices remain under [third_party](third_party/).
