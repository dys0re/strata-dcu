// Q4_0 x Q8_0: two blocks per ZMM, shared weight decoding across tokens.
// VNNI computes unsigned nibble * signed activation, then subtracts the
// zero-point contribution. This also handles a Q8 byte of -128 exactly.
#include "strata/kernels/cpu/q4_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace strata::kernels::cpu {
namespace {
constexpr int kMaxBlocks = 256;
constexpr int kSlice = 4;

inline float h2f(ggml_half h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(int(h))));
}

inline __m256i unpack(const uint8_t* q) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q));
    return _mm256_and_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(v),
                            _mm_srli_epi16(v, 4), 1), _mm256_set1_epi8(15));
}

inline __m512 scales(float a, float b) {
    return _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(a), _mm512_set1_ps(b));
}

struct Acts {
    const block_q8_0* y[kSlice];
    float d[kSlice][kMaxBlocks];
    int nb;
    Acts(const void* const* act, int nt, int n) : nb(n / QK8_0) {
        for (int t = 0; t < nt; ++t) {
            y[t] = static_cast<const block_q8_0*>(act[t]);
            for (int i = 0; i < nb; ++i) d[t][i] = h2f(y[t][i].d);
        }
    }
};

inline __m512i weight_pair(const block_q4_0* w, bool second) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(unpack(w[0].qs)),
                              second ? unpack(w[1].qs) : _mm256_setzero_si256(), 1);
}

inline __m512i act_pair(const block_q8_0* y, bool second) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(y[0].qs))),
                            second ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[1].qs))
                                   : _mm256_setzero_si256(), 1);
}

template<int NT, bool GU>
void rows(const uint8_t* w, size_t row_bytes, size_t up_off, const Acts& a,
           float* const* out, int r0, int r1) {
    const __m512i zero = _mm512_setzero_si512(), eight = _mm512_set1_epi8(8);
    for (int r = r0; r < r1; ++r) {
        const auto* g = reinterpret_cast<const block_q4_0*>(w + size_t(r) * row_bytes);
        const auto* u = reinterpret_cast<const block_q4_0*>(w + up_off + size_t(r) * row_bytes);
        __m512 ag[NT], au[NT];
        for (int t = 0; t < NT; ++t) { ag[t] = _mm512_setzero_ps(); au[t] = _mm512_setzero_ps(); }
        for (int i = 0; i < a.nb; i += 2) {
            const bool second = i + 1 < a.nb;
            const __m512i qg = weight_pair(g + i, second);
            const __m512 dg = scales(h2f(g[i].d), second ? h2f(g[i + 1].d) : 0.f);
            __m512i qu; __m512 du;
            if constexpr (GU) {
                qu = weight_pair(u + i, second);
                du = scales(h2f(u[i].d), second ? h2f(u[i + 1].d) : 0.f);
            }
            for (int t = 0; t < NT; ++t) {
                const __m512i y = act_pair(a.y[t] + i, second);
                const __m512 dy = scales(a.d[t][i], second ? a.d[t][i + 1] : 0.f);
                const __m512i correction = _mm512_dpbusd_epi32(zero, eight, y);
                const __m512 pg = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                                     _mm512_dpbusd_epi32(zero, qg, y), correction));
                ag[t] = _mm512_fmadd_ps(_mm512_mul_ps(dg, dy), pg, ag[t]);
                if constexpr (GU) {
                    const __m512 pu = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                                         _mm512_dpbusd_epi32(zero, qu, y), correction));
                    au[t] = _mm512_fmadd_ps(_mm512_mul_ps(du, dy), pu, au[t]);
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float gval = _mm512_reduce_add_ps(ag[t]);
            if constexpr (GU) {
                const float uval = _mm512_reduce_add_ps(au[t]);
                out[t][r] = (gval / (1.f + std::exp(-gval))) * uval;
            } else out[t][r] = gval;
        }
    }
}

template<bool GU>
void dispatch(const uint8_t* w, size_t row_bytes, size_t up_off, int n,
               const void* const* act, int nt, float* const* out, int r0, int r1) {
    if (nt <= 0 || r0 >= r1) return;
    const int slices = (nt + kSlice - 1) / kSlice;
    for (int s = 0, t0 = 0; s < slices; ++s) {
        const int count = nt / slices + (s < nt % slices);
        const Acts a(act + t0, count, n);
        switch (count) {
            case 1: rows<1, GU>(w, row_bytes, up_off, a, out + t0, r0, r1); break;
            case 2: rows<2, GU>(w, row_bytes, up_off, a, out + t0, r0, r1); break;
            case 3: rows<3, GU>(w, row_bytes, up_off, a, out + t0, r0, r1); break;
            case 4: rows<4, GU>(w, row_bytes, up_off, a, out + t0, r0, r1); break;
        }
        t0 += count;
    }
}
}  // namespace

bool q4_512_fits(int n) { return n > 0 && n % QK8_0 == 0 && n / QK8_0 <= kMaxBlocks; }

void q4_512_gu_rows(const uint8_t* blob, size_t row_bytes, size_t up_off, int n,
                    const void* const* act, int nt, float* const* ff, int r0, int r1) {
    dispatch<true>(blob, row_bytes, up_off, n, act, nt, ff, r0, r1);
}

void q4_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act,
                 int nt, float* const* out, int r0, int r1) {
    dispatch<false>(w, row_bytes, 0, n, act, nt, out, r0, r1);
}
}  // namespace strata::kernels::cpu
