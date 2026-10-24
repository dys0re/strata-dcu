#pragma once
#include <cstdint>

namespace strata::kernels {
// BF16 Y[T,N] = X[T,K] W[N,K]^T, FP32 accumulation followed by BF16 RNE.
// Returns false without launching on unsupported devices/shapes. Tuned for
// gfx936's wave64, T=1..8, with four columns sharing each weight load.
bool bf16_small_mm(const uint16_t* x, const uint16_t* w, uint16_t* y,
                    int t, int n, int k, void* stream);
}  // namespace strata::kernels
