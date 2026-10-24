#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

// Q4_0 rows against Q8_0 activations. Dispatch must check cpu_avx512_ok()
// before calling this translation unit. Weight blocks are decoded once per
// group of up to four tokens; larger groups are split into balanced slices.
bool q4_512_fits(int n);
void q4_512_gu_rows(const uint8_t* blob, size_t row_bytes, size_t up_off, int n,
                    const void* const* act, int nt, float* const* ff, int r0, int r1);
void q4_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act,
                 int nt, float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
