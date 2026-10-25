// Arithmetic adapted from the MIT-licensed pinned ggml CUDA
// moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
template<bool ZADD = false, bool GATED = false>
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ output,
                        int64_t n_embd, int k, const float* __restrict__ sg = nullptr) {
    // blockIdx.y = the token of a multi-token launch (0 for the single one)
    const int64_t tk = blockIdx.y;
    parts += tk * k * n_embd; weights += tk * k; if (shared) shared += tk * n_embd; output += tk * n_embd;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    // ZADD: a part is 0.0f + hit (the zeroed row plus moe_hit_add of the verify window), rounded as that add
    float p0 = parts[col];
    if (ZADD) p0 = 0.0f + p0;
    float sum = p0 * weights[0];
    for (int expert = 1; expert < k; ++expert) {
        float p = parts[int64_t(expert) * n_embd + col];
        if (ZADD) p = 0.0f + p;
        sum += p * weights[expert];
    }
    if constexpr (GATED) {
        // contraction off in this block only: the product is rounded on its own (scale_rows' store), then added
#pragma clang fp contract(off)
        const float g = __fdividef(1.0f, 1.0f + __expf(-sg[tk]));
        const float sh = shared[col] * g;
        sum += sh;
    } else {
        if (shared) sum += shared[col];
    }
    output[col] = sum;
}
__global__ void combine_k10_vec4(const float4* __restrict__ parts4, const float* __restrict__ weights,
                                 const float4* __restrict__ shared4, float4* __restrict__ output4,
                                 int64_t n4) {
    const int64_t tk = blockIdx.y;
    parts4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    const float4 p0 = parts4[c4];
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        const float4 p = parts4[int64_t(expert) * n4 + c4];
        // the documented contract spelled out: the first product rounded, then one FMA per expert in order. Left to the
        // compiler's contraction, a build may fuse a different product (gfx1151: the float4 kernel then differed from the
        // scalar one, native_multi_parity)
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x;
    sum.y += sh.y;
    sum.z += sh.z;
    sum.w += sh.w;
    output4[c4] = sum;
}
__global__ void combine_k10_vec4_hits(const float4* __restrict__ parts4, const float4* __restrict__ hits4,
                                      const int32_t* __restrict__ dst, const int32_t* __restrict__ count,
                                      const float* __restrict__ weights, const float4* __restrict__ shared4,
                                      float4* __restrict__ output4, int64_t n4) {
    const int64_t tk = blockIdx.y;
    __shared__ int hit_mask;
    if (threadIdx.x == 0) hit_mask = 0;
    __syncthreads();
    const int nh = *count;
    for (int h = threadIdx.x; h < nh; h += blockDim.x) {
        const int r = dst[h];
        if (r / 10 == (int) tk) atomicOr(&hit_mask, 1 << (r % 10));
    }
    __syncthreads();
    const int mask = hit_mask;
    parts4 += tk * 10 * n4;
    hits4 += tk * 10 * n4;
    weights += tk * 10;
    shared4 += tk * n4;
    output4 += tk * n4;
    const int64_t c4 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c4 >= n4) return;
    const float w0 = __ldg(weights + 0);
    float4 p0 = parts4[c4];
    if (mask & 1) { const float4 hh = hits4[c4]; p0.x += hh.x; p0.y += hh.y; p0.z += hh.z; p0.w += hh.w; }
    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
#pragma unroll
    for (int expert = 1; expert < 10; ++expert) {
        const float w = __ldg(weights + expert);
        float4 p = parts4[int64_t(expert) * n4 + c4];
        if ((mask >> expert) & 1) { const float4 hh = hits4[int64_t(expert) * n4 + c4]; p.x += hh.x; p.y += hh.y; p.z += hh.z; p.w += hh.w; }
        sum.x = fmaf(p.x, w, sum.x);
        sum.y = fmaf(p.y, w, sum.y);
        sum.z = fmaf(p.z, w, sum.z);
        sum.w = fmaf(p.w, w, sum.w);
    }
    const float4 sh = shared4[c4];
    sum.x += sh.x;
    sum.y += sh.y;
    sum.z += sh.z;
    sum.w += sh.w;
    output4[c4] = sum;
}
