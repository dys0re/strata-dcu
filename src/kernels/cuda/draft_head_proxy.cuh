// Adapted from jesdga95/basalt head_proxy.cuh, commit 588a4776.
// Copyright (c) 2026 Jesús Gulfo. MIT; see third_party/basalt-head-proxy/LICENSE.
// Port: full draft logits, deterministic bounded selection, BF16/Q8 exact rows; no target-head use.
// Two-bit screening layout, group fit, and the portable CUDA coarse-scoring path.
// The DCU scorer and original-weight rescoring live in draft_head_proxy.cu.
#pragma once

#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels::draft_px {
namespace {

constexpr int kNw4 = 2560 / 64;   // uint4 of weight words per row
constexpr int kPer = 4;           // candidates kept per tile of 32 rows
constexpr int kMaxCand = 64;      // rows rescored exactly
constexpr float kDelta = 5.0f;    // a row is rescored when its approximate score is within this of the best

__host__ __device__ inline int64_t tiles(int64_t n_cap) { return (n_cap + 31) / 32; }

inline size_t bytes(int64_t n_cap) { return (size_t) tiles(n_cap) * kNw4 * 32 * (16 + 4); }
inline size_t scratch_bytes(int64_t n_cap) { return (size_t) tiles(n_cap) * (kPer + 1) * sizeof(float2); }
__host__ __device__ inline uint16_t* scales(void* proxy, int64_t n_cap) {
    return (uint16_t*) ((char*) proxy + (size_t) tiles(n_cap) * kNw4 * 32 * 16);
}

struct X81 {
    half2 ds;
    int8_t qs[32];
};

// The level (0..3) of weight w at quantization step sh.
__device__ __forceinline__ int level(float w, float sh) { return min(3, max(0, (int) floorf(w / sh + 2.0f))); }

// One 32-weight group's scale: four levels with the scale (of a few candidates) that leaves the least squared error.
// `sh` gets the step the levels are taken at; the result is the stored scale.  UNSHRINK: the stored scale is then
// multiplied by sqrt(|w|^2 / <w, w_hat>); the least-squares copy scores the rows near the best low, which inflates
// the draft probability, and half the full correction leaves it close to the exact one.
template <bool UNSHRINK = false>
__device__ __forceinline__ __half fit_group(const float* w, float& sh_out) {
    float amax = 1e-9f;
    for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(w[l]));
    float best_s = amax / 1.5f, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = amax / 1.5f * (1.0f - 0.1f * (float) t);
        const float sh = __half2float(__float2half(s));
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) level(w[l], sh) - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) { best_e = e; best_s = s; }
    }
    __half hs = __float2half(best_s);
    const float sh = __half2float(hs);
    sh_out = sh;
    if constexpr (UNSHRINK) {
        float ww = 0.0f, wr = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) level(w[l], sh) - 1.5f) * sh;
            ww += w[l] * w[l];
            wr += w[l] * r;
        }
        if (wr > 0.0f) hs = __float2half(sh * sqrtf(ww / wr));
    }
    return hs;
}

// Group g (32 weights, dequantized) of row `row` into the copy.
template <bool UNSHRINK = false>
__device__ __forceinline__ void pack_group(const float* w, int64_t row, int g, uint4* __restrict__ wq,
                                           uint16_t* __restrict__ sc) {
    float sh;
    const __half hs = fit_group<UNSHRINK>(w, sh);
    const int tile = (int) (row >> 5), lane = (int) (row & 31), w4 = g / 2, h = g & 1;
    uint32_t words[2] = {0u, 0u};
    for (int c = 0; c < 2; ++c)
        for (int ii = 0; ii < 4; ++ii)
            for (int bb = 0; bb < 4; ++bb) words[c] |= (uint32_t) level(w[c * 16 + 4 * ii + bb], sh) << (8 * bb + 2 * ii);
    uint32_t* wd = reinterpret_cast<uint32_t*>(wq) + (((size_t) tile * kNw4 + w4) * 32 + lane) * 4 + 2 * h;
    wd[0] = words[0];
    wd[1] = words[1];
    sc[(((size_t) tile * kNw4 + w4) * 32 + lane) * 2 + h] = __half_as_ushort(hs);
}

// One tile per block, four warps splitting the row length; lane = row.  Writes the tile's best kPer approximate
// scores and the tile's softmax partial (max, sum of exp(score - max)).
__global__ void __launch_bounds__(128) score_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc,
                                                   const X81* __restrict__ x, int n_rows,
                                                   const int32_t* __restrict__ map, float2* __restrict__ cand,
                                                   float2* __restrict__ part, float* __restrict__ logits) {
    __shared__ __align__(16) int xi[640];
    __shared__ float xs[80], xdd[80];
    __shared__ float red[4][32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, tile = blockIdx.x;
    const size_t base = ((size_t) tile * kNw4 + warp * 10) * 32 + lane;
    uint4 w[10];
    uint32_t s[10];
#pragma unroll
    for (int j = 0; j < 10; ++j) {   // constant data: read before the wait
        w[j] = wq[base + (size_t) j * 32];
        s[j] = sc[base + (size_t) j * 32];
    }
    const int row = tile * 32 + lane;
    const int32_t id = (warp == 0 && row < n_rows) ? (map ? __ldg(map + row) : row) : 0;
    pdl_wait();
    for (int i = tid; i < 640; i += 128) xi[i] = reinterpret_cast<const int*>(x[i / 8].qs)[i % 8];
    if (tid < 80) {
        const int* q = reinterpret_cast<const int*>(x[tid].qs);
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) sum = STRATA_DP4A(0x01010101, q[i], sum);
        const float d8 = __low2float(x[tid].ds);
        xdd[tid] = d8;
        xs[tid] = d8 * (float) sum;
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int w4 = warp * 10 + j;
        const uint32_t ww[4] = {w[j].x, w[j].y, w[j].z, w[j].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int g = 2 * w4 + h;
            int P = 0;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint32_t v = ww[2 * h + c];
                const int4 x0 = *reinterpret_cast<const int4*>(&xi[g * 8 + c * 4]);
                P = STRATA_DP4A((int) (v & 0x03030303u), x0.x, P);
                P = STRATA_DP4A((int) ((v >> 2) & 0x03030303u), x0.y, P);
                P = STRATA_DP4A((int) ((v >> 4) & 0x03030303u), x0.z, P);
                P = STRATA_DP4A((int) ((v >> 6) & 0x03030303u), x0.w, P);
            }
            const float sg = __half2float(__ushort_as_half((unsigned short) (h ? (s[j] >> 16) : (s[j] & 0xffff))));
            acc += sg * (xdd[g] * (float) P - 1.5f * xs[g]);
        }
    }
    red[warp][lane] = acc;
    __syncthreads();
    if (warp != 0) return;
    float v = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
    if (row >= n_rows || id < 0) v = __int_as_float(0xff800000);
    if (row < n_rows) logits[row] = v;
    float m = v;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float e = v > __int_as_float(0xff800000) ? __expf(v - m) : 0.0f;
#pragma unroll
    for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
    if (lane == 0) part[tile] = make_float2(m, e);
    for (int k = 0; k < kPer; ++k) {
        float bv = v;
        int bi = lane;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) cand[tile * kPer + k] = make_float2(bv, __int_as_float(tile * 32 + bi));
        if (lane == bi) v = __int_as_float(0xff800000);
    }
}


} // namespace
} // namespace strata::kernels::draft_px
