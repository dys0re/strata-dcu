#!/usr/bin/env bash
set -eo pipefail
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
cmake -S /root/strata-dcu/Strata -B /root/strata-dcu/build-gfx936 -G Ninja \
  -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx936 \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/dtk \
  -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/dtk \
  -DSTRATA_GGML_DIR=/root/strata-dcu/llama.cpp -DSTRATA_BUILD_TESTS=ON
cmake --build /root/strata-dcu/build-gfx936 --target strata -j 6
