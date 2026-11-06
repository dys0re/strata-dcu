#include "strata/kernels/draft_head_proxy.hpp"
#include "draft_head_proxy.cuh"
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int K = 2560, NB = K / 32;
#if defined(STRATA_HIP_GFX906)
// The gfx936 compatibility backend implements CUDA dot4 with scalar byte operations. A two-bit
// code times FP32 activation avoids those signed-byte unpack/multiply sequences. Quantization
// and the stored weight layout are unchanged; only the approximate draft score's reduction changes.
__global__ __launch_bounds__(128) void score_float_kernel(const uint4* wq, const uint32_t* scales,
        const draft_px::X81* x, int rows, const int32_t* map, float2* candidates, float2* part, float* logits) {
    __shared__ float input[K], red[4][32];
    const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, tile = int(blockIdx.x);
    for (int i = tid; i < K; i += 128) input[i] = float(x[i / 32].qs[i % 32]) * __low2float(x[i / 32].ds);
    __syncthreads();
    float acc = 0;
    for (int j = 0; j < 10; ++j) {
        const int group2 = warp * 10 + j;
        const size_t at = (size_t(tile) * 40 + group2) * 32 + lane;
        const uint4 w = wq[at]; const uint32_t sc = scales[at];
        const uint32_t words[4] = {w.x, w.y, w.z, w.w};
        for (int h = 0; h < 2; ++h) {
            const int g = 2 * group2 + h; float v = 0;
            for (int c = 0; c < 2; ++c) {
                const uint32_t packed = words[2 * h + c];
                for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
                    const float code = float((packed >> (8 * b + 2 * a)) & 3u) - 1.5f;
                    v = fmaf(code, input[g * 32 + c * 16 + a * 4 + b], v);
                }
            }
            const float s = __half2float(__ushort_as_half(uint16_t(h ? sc >> 16 : sc)));
            acc = fmaf(s, v, acc);
        }
    }
    red[warp][lane] = acc; __syncthreads();
    if (warp != 0) return;
    const int row = tile * 32 + lane;
    float v = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
    if (row >= rows || (map && map[row] < 0)) v = -INFINITY;
    if (row < rows) logits[row] = v;
    float m = v;
    for (int d = 16; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d, 32));
    float total = isfinite(v) ? expf(v - m) : 0;
    for (int d = 16; d; d >>= 1) total += __shfl_xor_sync(0xffffffffu, total, d, 32);
    if (lane == 0) part[tile] = make_float2(m, total);
    for (int c = 0; c < 4; ++c) {
        float best = v; int at = lane;
        for (int d = 16; d; d >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, d, 32);
            const int index = __shfl_xor_sync(0xffffffffu, at, d, 32);
            if (other > best || (other == best && index < at)) { best = other; at = index; }
        }
        if (lane == 0) candidates[tile * 4 + c] = make_float2(best, __int_as_float(tile * 32 + at));
        if (lane == at) v = -INFINITY;
    }
}
#endif
__device__ int8_t q8_code(const ExecutionWeight& w,int row,int col) {
    if(w.layout==WeightLayout::Gguf)
        return ((const int8_t*)w.data)[(size_t(row)*NB+col/32)*34+2+col%32];
    const int tiles=(w.n_out+15)/16,j=col%32;
    return ((const int8_t*)w.data)[((size_t(col/32)*tiles+row/16)*64+(j/8)*16+row%16)*8+j%8];
}
__device__ float q8_scale(const ExecutionWeight& w,int row,int group) {
    if(w.layout==WeightLayout::Gguf)
        return __half2float(__ushort_as_half(*(const uint16_t*)((const uint8_t*)w.data+(size_t(row)*NB+group)*34)));
    const int tiles=(w.n_out+15)/16;
    return __half2float(((const half*)((const uint8_t*)w.data+size_t(tiles)*16*K))[(size_t(group)*tiles+row/16)*16+row%16]);
}
__device__ float weight_at(const ExecutionWeight& w,int row,int col) {
    if(w.type==30)return __uint_as_float(unsigned(((const uint16_t*)w.data)[size_t(row)*K+col])<<16);
    return q8_scale(w,row,col/32)*float(q8_code(w,row,col));
}

__global__ void pack_kernel(ExecutionWeight weights, void* proxy, int rows) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows * NB) return;
    const int row = i / NB, group = i % NB;
    float w[32];
    for (int j = 0; j < 32; ++j) w[j] = weight_at(weights, row, group * 32 + j);
    draft_px::pack_group<true>(w, row, group, (uint4*) proxy, draft_px::scales(proxy, rows));
}

// Exact top-64 among the tile candidates, with score descending / row ascending order.
// Reducing each block to 64 preserves the global top-64. The final <=4096-item sort fits
// in 32 KiB; this avoids 64 serial global reductions on a single workgroup.
__device__ bool before(float2 a, float2 b) {
    return a.x > b.x || (a.x == b.x && __float_as_int(a.y) < __float_as_int(b.y));
}
__device__ void bitonic(float2* values, int n) {
    const int tid = int(threadIdx.x);
    for (int width = 2; width <= n; width *= 2)
        for (int stride = width / 2; stride; stride /= 2) {
            for (int i = tid; i < n; i += int(blockDim.x)) {
                const int j = i ^ stride;
                if (j > i) {
                    const float2 a = values[i], b = values[j];
                    const bool descending = (i & width) == 0;
                    if (descending ? before(b, a) : before(a, b)) { values[i] = b; values[j] = a; }
                }
            }
            __syncthreads();
        }
}
__global__ void reduce_candidates(const float2* input, int n, float2* output) {
    __shared__ float2 values[256];
    const int tid = int(threadIdx.x), at = int(blockIdx.x) * 256 + tid;
    values[tid] = at < n ? input[at] : make_float2(-INFINITY, __int_as_float(0x7fffffff));
    __syncthreads();
    bitonic(values, 256);
    if (tid < 64) output[int(blockIdx.x) * 64 + tid] = values[tid];
}
__global__ void finish_candidates(const float2* input, int n, int padded, int32_t* selected) {
    extern __shared__ float2 values[];
    const int tid = int(threadIdx.x);
    for (int i = tid; i < padded; i += 256)
        values[i] = i < n ? input[i] : make_float2(-INFINITY, __int_as_float(0x7fffffff));
    __syncthreads();
    bitonic(values, padded);
    if (tid == 0) {
        int count = 0;
        while (count < 64 && count < n && isfinite(values[count].x) && values[count].x >= values[0].x - 5.f) {
            selected[count + 1] = __float_as_int(values[count].y); ++count;
        }
        selected[0] = count;
    }
}

__global__ void exact_kernel(ExecutionWeight weights, const draft_px::X81* x,
                              const int32_t* selected, float* logits) {
    const int c = int(blockIdx.x), tid = int(threadIdx.x);
    if (c >= selected[0]) return;
    const int row = selected[c + 1];
    __shared__ float sum[128];
    float value = 0;
    for (int b = tid; b < NB; b += 128) {
        const float dx = __low2float(x[b].ds);
        if (weights.type == 8) {
            int dot = 0;
            for (int j = 0; j < 32; ++j) dot += int(q8_code(weights, row, b * 32 + j)) * int(x[b].qs[j]);
            value += float(dot) * (q8_scale(weights, row, b) * dx);
        } else {
            float part = 0;
            for (int j = 0; j < 32; ++j) part += weight_at(weights, row, b * 32 + j) * float(x[b].qs[j]);
            value += part * dx;
        }
    }
    sum[tid] = value; __syncthreads();
    for (int s = 64; s; s >>= 1) { if (tid < s) sum[tid] += sum[tid + s]; __syncthreads(); }
    if (tid == 0) logits[row] = sum[0];
}
} // namespace

size_t draft_head_proxy_bytes(int rows) { return rows > 0 ? draft_px::bytes(rows) : 0; }
size_t draft_head_proxy_scratch_bytes(int rows) {
    return rows > 0 ? draft_px::scratch_bytes(rows) + 2 * size_t(draft_px::tiles(rows)) * draft_px::kPer * sizeof(float2) + 65 * sizeof(int32_t) : 0;
}
void draft_head_proxy_pack(int type, const void* weights, void* proxy, int rows, void* stream) {
    draft_head_proxy_pack(ExecutionWeight{weights,type,K,rows},proxy,rows,stream);
}
void draft_head_proxy_pack(const ExecutionWeight& w, void* proxy, int rows, void* stream) {
    const int type=w.type;const void* weights=w.data;
    if(w.n_in!=K || w.n_out!=rows || (type==30 && !w.gguf()))throw std::invalid_argument("proxy execution weight shape/layout");
    if ((type != 30 && type != 8) || !weights || !proxy || rows < 1 || rows > 300000)
        throw std::invalid_argument("draft head proxy: expected BF16/Q8, width2560, <=300000 rows");
    pack_kernel<<<(rows * NB + 255) / 256, 256, 0, (cudaStream_t) stream>>>(w, proxy, rows);
}
bool draft_head_proxy_logits(int type, const void* proxy, const void* weights, const void* x,
                             float* logits, int rows, int columns, const int32_t* map,
                             void* scratch, void* stream, void* const* diagnostic_events) {
    return draft_head_proxy_logits(ExecutionWeight{weights,type,K,rows},proxy,x,logits,rows,columns,map,scratch,stream,diagnostic_events);
}
bool draft_head_proxy_logits(const ExecutionWeight& w, const void* proxy, const void* x,
                             float* logits, int rows, int columns, const int32_t* map,
                             void* scratch, void* stream, void* const* diagnostic_events) {
    const int type=w.type;const void* weights=w.data;
    if(w.n_in!=K || w.n_out!=rows || (type==30 && !w.gguf()))return false;
    if ((type != 30 && type != 8) || columns != 1 || rows < 1 || rows > 300000 ||
        !proxy || !weights || !x || !logits || !scratch) return false;
    const int tiles = int(draft_px::tiles(rows));
    auto* cand = (float2*) scratch;
    auto* part = cand + size_t(tiles) * draft_px::kPer;
    auto* sort0 = part + tiles;
    auto* sort1 = sort0 + size_t(tiles) * draft_px::kPer;
    auto* selected = (int32_t*) (sort1 + size_t(tiles) * draft_px::kPer);
    const auto cs = (cudaStream_t) stream;
    auto mark = [&](int i) { if (diagnostic_events) cudaEventRecord((cudaEvent_t) diagnostic_events[i], cs); };
    // Ordinary ordered launches: HIP has no programmatic launch dependency, and this draft path
    // can follow copies/quantization in a graph. Constant-weight packing is outside the graph.
    mark(0);
#if defined(STRATA_HIP_GFX906)
    score_float_kernel<<<tiles, 128, 0, cs>>>((const uint4*) proxy,
#else
    draft_px::score_kernel<<<tiles, 128, 0, cs>>>((const uint4*) proxy,
#endif
        (const uint32_t*) draft_px::scales(const_cast<void*>(proxy), rows),
        (const draft_px::X81*) x, rows, map, cand, part, logits);
    mark(1);
    int count = tiles * draft_px::kPer;
    const float2* current = cand; float2* next = sort0;
    while (count > 4096) {
        const int blocks = (count + 255) / 256;
        reduce_candidates<<<blocks, 256, 0, cs>>>(current, count, next);
        count = blocks * 64; current = next; next = next == sort0 ? sort1 : sort0;
    }
    int padded = 1; while (padded < count) padded *= 2;
    finish_candidates<<<1, 256, size_t(padded) * sizeof(float2), cs>>>(current, count, padded, selected);
    mark(2);
    exact_kernel<<<64, 128, 0, cs>>>(w, (const draft_px::X81*) x, selected, logits);
    mark(3);
    return true;
}
} // namespace strata::kernels
