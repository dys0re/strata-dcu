#include "strata/kernels/bf16_small.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <cstring>

namespace strata::kernels {
#if defined(STRATA_HIP_GFX906)
namespace {
template<int C>
__global__ __launch_bounds__(256) void small_mm(const uint16_t* __restrict__ x,
        const uint16_t* __restrict__ w, uint16_t* __restrict__ y, int t, int n, int k) {
    const int lane = threadIdx.x % 64;
    const int row = blockIdx.x * 4 + threadIdx.x / 64;
    if (row >= n) return;
    const int col0 = blockIdx.y * C;
    float acc[C][4] = {};
    for (int i = lane * 4; i < k; i += 256) {
        unsigned long long wb;
        __builtin_memcpy(&wb, w + size_t(row) * k + i, 8);
        float weights[4];
#pragma unroll
        for (int j=0;j<4;++j) weights[j]=__uint_as_float(uint32_t(wb>>(j*16))<<16);
#pragma unroll
        for (int c=0;c<C;++c) if (col0+c<t) {
            unsigned long long xb;
            __builtin_memcpy(&xb,x+size_t(col0+c)*k+i,8);
#pragma unroll
            for (int j=0;j<4;++j)
                acc[c][j]=__fmaf_rn(weights[j],__uint_as_float(uint32_t(xb>>(j*16))<<16),acc[c][j]);
        }
    }
#pragma unroll
    for (int c=0;c<C;++c) if (col0+c<t) {
        float v=(acc[c][0]+acc[c][1])+(acc[c][2]+acc[c][3]);
        for(int off=32;off;off>>=1)v+=__shfl_down_sync(0xffffffffu,v,off,64);
        if(lane==0)y[size_t(col0+c)*n+row]=bf16_from_f32(v);
    }
}
}  // namespace
#endif

bool bf16_small_mm(const uint16_t* x,const uint16_t* w,uint16_t* y,
                    int t,int n,int k,void* stream) {
#if defined(STRATA_HIP_GFX906)
    if(t<1||t>8||n<1||k<4||k%4)return false;
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess)return false;
    thread_local int cached=-2;
    thread_local bool enabled=false;
    if(cached!=device) {
        cudaDeviceProp p{};
        enabled=cudaGetDeviceProperties(&p,device)==cudaSuccess && p.warpSize==64 &&
            std::strncmp(p.gcnArchName,"gfx936",6)==0 && (p.gcnArchName[6]=='\0'||p.gcnArchName[6]==':');
        cached=device;
    }
    if(!enabled)return false;
    const dim3 block(256);
    if(t==1)small_mm<1><<<dim3((n+3)/4),block,0,(cudaStream_t)stream>>>(x,w,y,t,n,k);
    else if(t==2)small_mm<2><<<dim3((n+3)/4),block,0,(cudaStream_t)stream>>>(x,w,y,t,n,k);
    else small_mm<4><<<dim3((n+3)/4,(t+3)/4),block,0,(cudaStream_t)stream>>>(x,w,y,t,n,k);
    return true;
#else
    (void)x;(void)w;(void)y;(void)t;(void)n;(void)k;(void)stream;
    return false;
#endif
}
}  // namespace strata::kernels
