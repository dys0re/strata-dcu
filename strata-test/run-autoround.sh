#!/usr/bin/env bash
set -eo pipefail
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
# gfx936-only BF16/Q8_1 fast path; set to 0 for the original FP32 sum order.
export STRATA_BF16_Q8_VEC4="${STRATA_BF16_Q8_VEC4:-1}"
# Subsequent Qwen speed tests: all GPU model weights together <= 50% of physical VRAM.
export STRATA_MAX_WEIGHT_VRAM_FRACTION=0.5
export STRATA_BF16_Q8_COLS="${STRATA_BF16_Q8_COLS:-4}"
# Measured on this gfx936 host. Each can be set to 0 for the prior kernel path.
export STRATA_CPU_Q4_MT="${STRATA_CPU_Q4_MT:-1}"
export STRATA_DFLASH_SMALL_GEMV="${STRATA_DFLASH_SMALL_GEMV:-1}"
# This host's enhanced-MTP defaults; individual scheduling variables still override them.
export STRATA_DCU_SCHEDULING_DEFAULTS="${STRATA_DCU_SCHEDULING_DEFAULTS:-1}"
exec python3 /root/strata-dcu/Strata/tools/autoround_pack.py run \
  --model /root/private_data/Qwen3.8-F \
  --out /root/strata-dcu/autoround \
  --llama /root/strata-dcu/llama.cpp \
  --engine /root/strata-dcu/build-gfx936/strata --mtp-mode adaptive "$@"
