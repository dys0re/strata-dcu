#!/usr/bin/env bash
set -eo pipefail
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
exec python3 /root/strata-dcu/Strata/tools/autoround_pack.py run \
  --model /root/private_data/Qwen3.8-F \
  --out /root/strata-dcu/autoround \
  --llama /root/strata-dcu/llama.cpp \
  --engine /root/strata-dcu/build-gfx936/strata "$@"
