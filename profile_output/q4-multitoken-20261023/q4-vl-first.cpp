// Q4_0 x Q8_0: shared weight decoding and activation loads across tokens.
// Uses AVX-512 VL/BW/VNNI; packed unsigned magnitudes with signed activations.
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

inline float reduce(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    return _mm_cvtss_f32(_mm_add_ss(r, _mm_movehdup_ps(r)));
}

// 256-bit EVEX/VNNI matches Q4_0's natural 32-value block. Packing blocks
// into 512-bit lanes costs more than it saves for this CPU and these shapes.
inline __m256 product(__m256i mag, __mmask32 neg, __m256i y, __mmask32 ymin) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i sy = _mm256_mask_sub_epi8(y, neg, zero, y);
    __m256i p = _mm256_dpbusd_epi32(zero, mag, sy);
    if (ymin) {
        // Negating -128 wraps. Add back 256*|weight| for those elements.
        const __m256i fix = _mm256_maskz_mov_epi8(neg & ymin, mag);
        p = _mm256_add_epi32(p, _mm256_slli_epi32(_mm256_dpbusd_epi32(
                                zero, fix, _mm256_set1_epi8(1)), 8));
    }
    return _mm256_cvtepi32_ps(p);
}

template<int NT, bool GU>
void rows(const uint8_t* w, size_t row_bytes, size_t up_off, const Acts& a,
           float* const* out, int r0, int r1) {
    const __m256i eight = _mm256_set1_epi8(8), minbyte = _mm256_set1_epi8(-128);
    for (int r = r0; r < r1; ++r) {
        const auto* g = reinterpret_cast<const block_q4_0*>(w + size_t(r) * row_bytes);
        const auto* u = reinterpret_cast<const block_q4_0*>(w + up_off + size_t(r) * row_bytes);
        __m256 ag[NT], au[NT];
        for (int t = 0; t < NT; ++t) { ag[t] = _mm256_setzero_ps(); au[t] = _mm256_setzero_ps(); }
        for (int i = 0; i < a.nb; ++i) {
            const __m256i qg = _mm256_sub_epi8(unpack(g[i].qs), eight);
            const __m256i mg = _mm256_abs_epi8(qg);
            const __mmask32 ng = _mm256_movepi8_mask(qg);
            const float dg = h2f(g[i].d);
            __m256i mu; __mmask32 nu; float du;
            if constexpr (GU) {
                const __m256i qu = _mm256_sub_epi8(unpack(u[i].qs), eight);
                mu = _mm256_abs_epi8(qu); nu = _mm256_movepi8_mask(qu); du = h2f(u[i].d);
            }
            for (int t = 0; t < NT; ++t) {
                const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.y[t][i].qs));
                const __mmask32 ymin = _mm256_cmpeq_epi8_mask(y, minbyte);
                ag[t] = _mm256_fmadd_ps(_mm256_set1_ps(dg * a.d[t][i]), product(mg, ng, y, ymin), ag[t]);
                if constexpr (GU)
                    au[t] = _mm256_fmadd_ps(_mm256_set1_ps(du * a.d[t][i]), product(mu, nu, y, ymin), au[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float gval = reduce(ag[t]);
            if constexpr (GU) {
                const float uval = reduce(au[t]);
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
