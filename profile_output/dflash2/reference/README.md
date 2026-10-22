---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
  - Qwen/Qwen3.8-Flash-Next
inference: false
tags:
  - dflash2
  - speculative-decoding
  - block-diffusion
  - draft-model
  - sglang
---

# Qwen3.8-Flash-Next-DFlash2

[Blog](https://inco.ai/blog/dflash2/) | [GitHub](https://github.com/z-lab/dflash)

This repository contains the DFlash 2 draft model for
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
It is not a standalone language model: it runs inside a speculative
decoding server and drafts tokens for the target model to verify.

DFlash 2 is a block-diffusion drafter for speculative decoding. It predicts
a whole block of tokens in a single pass and keeps the top candidates at
every position. A lightweight selector then traces one coherent path through them.
Two-tap dynamic convolutions in the backbone keep the draft from decaying
toward the end of the block. Decoding is lossless: greedy output
matches the target model exactly, and sampling preserves its distribution.

<div align="center">
  <img src="assets/dflash2-figure.png" alt="DFlash 2: parallel block drafting with a candidate path selector" width="100%">
</div>

## Quick Start

Serve with [SGLang](https://github.com/sgl-project/sglang). Flash-Next needs a small
target-feature capture patch, [included in this repository](patches/qwen38-dflash2-capture.patch)
and validated at TP1, until support lands upstream:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout 3c560089e1818a5217a73f2168273485a5350341
git apply "$(hf download incoai/Qwen3.8-Flash-Next-DFlash2 \
  patches/qwen38-dflash2-capture.patch --quiet)"
pip install -e "python[all]"

python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-Flash-Next-FP8 \
  --linear-attn-verify-backend triton \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path incoai/Qwen3.8-Flash-Next-DFlash2 \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens 4
```

See the [blog post](https://inco.ai/blog/dflash2/) for more details.

## Evaluation

- Target: the official FP8 checkpoint, [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- Runtime: SGLang on one NVIDIA GB300 (TP1)
- Target attention: Qwen Sparse Attention (QSA) for sparse-attention layers; Gated DeltaNet with FlashInfer prefill/decode and Triton verification
- DFlash 2 draft attention: `trtllm_mha`
- Speculation block size: 4 (3 draft tokens per verification step), limited by the current QSA implementation
- Sampling: Flash-Next's officially recommended parameters (temperature 1.0, top-p 0.95, top-k 20), with `xhigh` reasoning effort
- Maximum new tokens: 4096, including reasoning tokens
- Samples: 128 at concurrency 1; 1,024 at concurrency 8 and 32
- Prompts: benchmark formatting and repeated sampling from [`z-lab/dflash`](https://github.com/z-lab/dflash)

We compare autoregressive decoding, Flash-Next's built-in MTP,
and DFlash 2. Both speculative methods propose three draft tokens per
verification step.

### Acceptance Length

Acceptance length is the per-request mean of completion tokens divided by verification steps at concurrency 1.
Higher is better.

| Task | MTP | DFlash 2 |
| :--- | ---: | ---: |
| GSM8K | 3.08 | **3.26** |
| MATH-500 | 2.93 | **3.17** |
| HumanEval | 2.52 | **2.94** |
| MBPP | 2.65 | **3.06** |
| MT-Bench | 2.50 | **2.83** |

### Throughput

Throughput is total output tokens divided by end-to-end wall time.
Each cell shows `output tok/s (speedup vs. autoregressive)`.

#### Concurrency 1

| Task | Autoregressive | MTP | DFlash 2 |
| :--- | ---: | ---: | ---: |
| GSM8K | 205.4 | 326.0 (1.59×) | **368.2 (1.79×)** |
| MATH-500 | 210.5 | 325.9 (1.55×) | **383.6 (1.82×)** |
| HumanEval | 211.3 | 296.9 (1.40×) | **363.8 (1.72×)** |
| MBPP | 210.5 | 304.7 (1.45×) | **378.3 (1.80×)** |
| MT-Bench | 211.2 | 285.1 (1.35×) | **343.8 (1.63×)** |

#### Concurrency 8

| Task | Autoregressive | MTP | DFlash 2 |
| :--- | ---: | ---: | ---: |
| GSM8K | 924.8 | 1,122.5 (1.21×) | **1,312.6 (1.42×)** |
| MATH-500 | 998.1 | 1,248.4 (1.25×) | **1,473.1 (1.48×)** |
| HumanEval | 999.5 | 1,138.2 (1.14×) | **1,397.5 (1.40×)** |
| MBPP | 987.7 | 1,156.9 (1.17×) | **1,420.0 (1.44×)** |
| MT-Bench | 966.7 | 1,075.6 (1.11×) | **1,261.8 (1.31×)** |

#### Concurrency 32

| Task | Autoregressive | MTP | DFlash 2 |
| :--- | ---: | ---: | ---: |
| GSM8K | 1,680.9 | 2,027.5 (1.21×) | **2,907.3 (1.73×)** |
| MATH-500 | 2,075.9 | 2,566.6 (1.24×) | **3,173.3 (1.53×)** |
| HumanEval | 2,072.2 | 2,321.2 (1.12×) | **3,039.6 (1.47×)** |
| MBPP | 2,044.0 | 2,390.8 (1.17×) | **3,131.0 (1.53×)** |
| MT-Bench | 1,884.1 | 2,201.8 (1.17×) | **2,695.5 (1.43×)** |

## Citation

If you find DFlash 2 useful, please cite:

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}
```

Please also cite the original DFlash paper:

```bibtex
@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
