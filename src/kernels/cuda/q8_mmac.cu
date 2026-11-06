#include "strata/kernels/q8_mmac.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace strata::kernels {
namespace {
using I4 = int __attribute__((ext_vector_type(4)));
struct Act { half2 ds; int8_t q[32]; };
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
bool stable_widths() {
    static const bool on=[] { const char* v=std::getenv("STRATA_Q8_MMAC_STABLE");
        return v && std::strcmp(v,"1")==0;
    }();
    return on;
}
template<int T> __global__ void compute(const uint64_t* w,const half* d,const Act* x,float* y,int ni,int no,int split) {
    const int part=blockIdx.y,l=threadIdx.x&63,g=l/16,r=l%16;
    const int tile=blockIdx.x*4+threadIdx.x/64,tiles=(no+15)/16,nb=ni/32;
    if(tile>=tiles)return;
    float sum[(T+3)/4]={}; const int span=(nb+split-1)/split;
    for(int kb=part*span;kb<min(nb,(part+1)*span);++kb) {
        uint64_t a=w[(size_t(kb)*tiles+tile)*64+l],b=0;
        if(r<T) __builtin_memcpy(&b,x[size_t(r)*nb+kb].q+g*8,8);
        I4 z={0,0,0,0};
#if defined(__gfx936__)
        z=__builtin_amdgcn_mmac_i32_16x16x32i8(a,b,z);
#endif
        const float dw=__half2float(d[(size_t(kb)*tiles+tile)*16+r]);
#pragma unroll
        for(int j=0;j<(T+3)/4;++j) { const int col=g+4*j;
            if(col<T) sum[j]=fmaf(dw*__low2float(x[size_t(col)*nb+kb].ds),float(z[j]),sum[j]);
        }
    }
#pragma unroll
    for(int j=0;j<(T+3)/4;++j) { const int col=g+4*j;
        if(col<T && tile*16+r<no) y[(size_t(part)*T+col)*no+tile*16+r]=sum[j];
    }
}
__global__ void merge(const float* x,float* y,int n,int parts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { float v=x[i]; for(int p=1;p<parts;++p)v+=x[size_t(p)*n+i]; y[i]=v; }
}
template<int T> void run(const ExecutionWeight&e,const Act*x,float*y,int split,cudaStream_t s) {
    auto* d=reinterpret_cast<const half*>(static_cast<const uint8_t*>(e.data)+size_t((e.n_out+15)/16)*16*e.n_in);
    compute<T><<<dim3((e.n_out+63)/64,split),256,0,s>>>(static_cast<const uint64_t*>(e.data),d,x,split==1?y:e.partial,e.n_in,e.n_out,split);
    if(split>1)merge<<<(e.n_out*T+255)/256,256,0,s>>>(e.partial,y,e.n_out*T,split);
}
}
bool q8_mmac_enabled() {
    static const bool on=[] { const char* v=std::getenv("STRATA_Q8_MMAC");
        if(!v || std::strcmp(v,"1"))return false;
        cudaDeviceProp p{};int dev=0;check(cudaGetDevice(&dev));check(cudaGetDeviceProperties(&p,dev));
        return std::strncmp(p.gcnArchName,"gfx936",6)==0;
    }(); return on;
}
bool q8_mmac_eligible(int ni,int no) { return q8_mmac_enabled() && (ni==2560 || ni==6144) && no>=2048; }
uint64_t q8_mmac_weight_bytes(int ni,int no) { return uint64_t((no+15)/16)*16*(ni/32)*34; }
uint64_t q8_mmac_workspace_bytes(int no) { return no>=65536?0:uint64_t(no)*16*5*4; }
ExecutionWeight q8_mmac_upload_host(const void* host,int ni,int no) {
    if(!host || !q8_mmac_eligible(ni,no)) throw std::invalid_argument("ineligible Q8 MMAC upload");
    const int nb=ni/32,tiles=(no+15)/16;const size_t qs=size_t(tiles)*16*ni;
    const uint64_t bytes=q8_mmac_weight_bytes(ni,no);
    std::vector<uint8_t> packed(bytes,0);const auto* src=static_cast<const uint8_t*>(host);
    for(int kb=0;kb<nb;++kb)for(int tile=0;tile<tiles;++tile)for(int lane=0;lane<64;++lane) {
        const int row=tile*16+lane%16;if(row>=no)continue;
        const auto* b=src+(size_t(row)*nb+kb)*34;
        std::memcpy(packed.data()+(size_t(kb)*tiles+tile)*512+lane*8,b+2+(lane/16)*8,8);
        if(lane<16)std::memcpy(packed.data()+qs+((size_t(kb)*tiles+tile)*16+lane)*2,b,2);
    }
    ExecutionWeight e;e.n_in=ni;e.n_out=no;e.type=8;e.layout=WeightLayout::Q8Mmac;
    e.bytes=bytes;e.workspace_bytes=q8_mmac_workspace_bytes(no);
    void* allocation=nullptr;
    try { check(cudaMalloc(&allocation,bytes));e.data=allocation;
        check(cudaMemcpy(allocation,packed.data(),bytes,cudaMemcpyHostToDevice));
        if(e.workspace_bytes)check(cudaMalloc(&e.partial,e.workspace_bytes));
    } catch(...) { if(e.partial)cudaFree(e.partial);if(allocation)cudaFree(allocation);throw; }
    std::fprintf(stderr,"strata q8 mmac: execution %d->%d, weights=%llu workspace=%llu sole_copy=1\n",ni,no,(unsigned long long)bytes,(unsigned long long)e.workspace_bytes);
    return e;
}

// Non-MMAC widths, including the fast profile's T=1, read the SAME layout.
// One wave per 16 output rows; no reconstruction buffer or token-path allocation.
template<int T> __global__ void vector_compute(const uint64_t* w,const half* d,const Act* x,float* y,int ni,int no,int split) {
    const int lane=threadIdx.x&63,row=(blockIdx.x*4+threadIdx.x/64)*16+lane%16;
    const int group=lane/16,part=blockIdx.y,nb=ni/32,tiles=(no+15)/16;
    if(row/16>=tiles)return;
    const int span=(nb+split-1)/split;
    float sum[T]={};
    // Adjacent lanes read adjacent packed fragments, rather than striding over K.
    // Four lanes cooperate on each row; all four read the same original block scale.
    for(int kb=part*span;kb<min(nb,(part+1)*span);++kb) {
        const size_t at=size_t(kb)*tiles+row/16;
        const uint64_t packed=w[at*64+lane];
        const float dw=__half2float(d[at*16+row%16]);
#pragma unroll
        for(int t=0;t<T;++t) {
            uint64_t q;__builtin_memcpy(&q,x[size_t(t)*nb+kb].q+group*8,8);
            int dot=0;
#if defined(__gfx936__)
            dot=__builtin_amdgcn_sdot4(int(packed),int(q),dot,false);
            dot=__builtin_amdgcn_sdot4(int(packed>>32),int(q>>32),dot,false);
#endif
            sum[t]=fmaf(dw*__low2float(x[size_t(t)*nb+kb].ds),float(dot),sum[t]);
        }
    }
#pragma unroll
    for(int t=0;t<T;++t) {
        float v=sum[t];
        v+=__shfl_down(v,32,64);v+=__shfl_down(v,16,64);
        if(group==0 && row<no)y[(size_t(part)*T+t)*no+row]=v;
    }
}
void q8_mmac_run(const ExecutionWeight& e,const void* x,float* y,int t,void* stream) {
    if(e.layout!=WeightLayout::Q8Mmac || e.type!=8 || !e.data || !x || !y || t<1 || t>8)
        throw std::invalid_argument("invalid MMAC execution view");
    const int ni=e.n_in,no=e.n_out;
    const bool stable=stable_widths();
    static const bool wide=[] { const char* v=std::getenv("STRATA_Q8_MMAC_WIDE");
        return v && std::strcmp(v,"1")==0;
    }();
    auto s=static_cast<cudaStream_t>(stream);const auto*a=static_cast<const Act*>(x);
    if((t>5 && !wide) || (!stable && t==1 && no<65536)) {
        const auto* d=reinterpret_cast<const half*>(static_cast<const uint8_t*>(e.data)+size_t((no+15)/16)*16*ni);
        const int split=no>=65536?1:(ni==6144?8:4);
        if(split>1 && (!e.partial || e.workspace_bytes<uint64_t(no)*t*split*sizeof(float)))
            throw std::invalid_argument("vector execution workspace missing");
        switch(t) {
#define VEC(T) case T:vector_compute<T><<<dim3((no+63)/64,split),256,0,s>>>(static_cast<const uint64_t*>(e.data),d,a,split==1?y:e.partial,ni,no,split);break;
            VEC(1) VEC(2) VEC(3) VEC(4) VEC(5) VEC(6) VEC(7) VEC(8)
#undef VEC
        }
        if(split>1)merge<<<(no*t+255)/256,256,0,s>>>(e.partial,y,no*t,split);
    } else {
        int split=1;
        if(no<65536) {
            // Wide verification uses both accumulator outputs per lane. Bound split*T by the existing
            // 80-row workspace budget; the narrow profile's split=16 would overrun it for T>5.
            if(t>5) { if(ni==6144)split=8;else if(no==6144 || no==12288)split=4;else split=8; }
            else if(stable) { if(ni==6144 || no==6144)split=16;else if(no==12288)split=8;else split=4; }
            else { if(ni==6144)split=t==4?8:16;else if(no==6144)split=t==4?4:16;else if(no==12288)split=t==4?4:8;else split=t==4?8:4; }
        }
        if(split>1 && (!e.partial || e.workspace_bytes<uint64_t(no)*t*split*sizeof(float)))
            throw std::invalid_argument("MMAC execution workspace missing");
        switch(t) {
#define GO(T) case T:run<T>(e,a,y,split,s);break;
            GO(1) GO(2) GO(3) GO(4) GO(5) GO(6) GO(7) GO(8)
#undef GO
        }
    }
    check(cudaGetLastError());
    static bool shown[9]={};if(!shown[t]){shown[t]=true;std::fprintf(stderr,"strata q8 mmac: dispatch T=%d stable=%d sole_copy=1 wide=%d\n",t,int(stable),int(wide));}
}

__device__ int8_t code_at(const uint8_t* w,int ni,int no,int row,int col) {
    const int tiles=(no+15)/16,kb=col/32,j=col%32;
    return reinterpret_cast<const int8_t*>(w)[((size_t(kb)*tiles+row/16)*64+(j/8)*16+row%16)*8+j%8];
}
__device__ half scale_at(const uint8_t* w,int ni,int no,int row,int kb) {
    const int tiles=(no+15)/16;
    return reinterpret_cast<const half*>(w+size_t(tiles)*16*ni)[(size_t(kb)*tiles+row/16)*16+row%16];
}
__global__ void dequant_kernel(const uint8_t* w,int ni,int no,int row0,int rows,int ld,half* out) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*ni)return;
    const int r=int(i/ni),c=int(i%ni);
    out[size_t(r)*ld+c]=__float2half_rn(__half2float(scale_at(w,ni,no,row0+r,c/32))*float(code_at(w,ni,no,row0+r,c)));
}
__global__ void gather_kernel(const uint8_t* w,int ni,int no,const int32_t* indices,int row0,int rows,uint8_t* out) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int nb=ni/32;const size_t rb=size_t(nb)*34;
    if(i>=size_t(rows)*rb)return;
    const int dst=int(i/rb),j=int(i%rb),kb=j/34,within=j%34;
    const int row=indices?indices[dst]:row0+dst;
    if(row<0 || row>=no)return;
    if(within<2) { const half d=scale_at(w,ni,no,row,kb);const auto bits=__half_as_ushort(d);out[i]=uint8_t(bits>>(8*within)); }
    else out[i]=uint8_t(code_at(w,ni,no,row,kb*32+within-2));
}
void q8_mmac_dequant_f16(const ExecutionWeight& e,int64_t row0,int64_t rows,int64_t ld,uint16_t* out,void* stream) {
    if(e.layout!=WeightLayout::Q8Mmac || row0<0 || rows<0 || row0+rows>e.n_out || ld<e.n_in || !out)
        throw std::invalid_argument("invalid MMAC dequant slice");
    if(!rows)return;
    dequant_kernel<<<(size_t(rows)*e.n_in+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(e.data),e.n_in,e.n_out,int(row0),int(rows),int(ld),reinterpret_cast<half*>(out));
    check(cudaGetLastError());
}
void q8_mmac_gather_rows(const ExecutionWeight& e,const int32_t* indices,int64_t row0,int64_t rows,void* out,void* stream) {
    if(e.layout!=WeightLayout::Q8Mmac || rows<0 || (!indices && (row0<0 || row0+rows>e.n_out)) || !out)
        throw std::invalid_argument("invalid MMAC row gather");
    if(!rows)return;
    const size_t n=size_t(rows)*(e.n_in/32)*34;
    gather_kernel<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(e.data),e.n_in,e.n_out,indices,int(row0),int(rows),static_cast<uint8_t*>(out));
    check(cudaGetLastError());
}
}
