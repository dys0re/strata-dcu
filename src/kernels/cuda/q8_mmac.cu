#include "strata/kernels/q8_mmac.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
namespace strata::kernels {
namespace {
using I4 = int __attribute__((ext_vector_type(4)));
struct Act { half2 ds; int8_t q[32]; };
struct Entry { void* packed=nullptr; float* partial=nullptr; int ni=0,no=0; };
auto& entries() { static std::unordered_map<const void*,Entry> e; return e; }
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
template<int T> void run(const Entry&e,const Act*x,float*y,int split,cudaStream_t s) {
    auto* d=reinterpret_cast<const half*>(static_cast<const uint8_t*>(e.packed)+size_t((e.no+15)/16)*16*e.ni);
    compute<T><<<dim3((e.no+63)/64,split),256,0,s>>>(static_cast<const uint64_t*>(e.packed),d,x,split==1?y:e.partial,e.ni,e.no,split);
    if(split>1)merge<<<(e.no*T+255)/256,256,0,s>>>(e.partial,y,e.no*T,split);
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
uint64_t q8_mmac_register_host(const void* canonical,const void* host,int ni,int no) {
    if(!q8_mmac_eligible(ni,no))return 0;
    if(entries().count(canonical))throw std::runtime_error("Q8 MMAC duplicate registration");
    const int nb=ni/32,tiles=(no+15)/16;const size_t qs=size_t(tiles)*16*ni;
    const uint64_t bytes=q8_mmac_weight_bytes(ni,no);
    std::vector<uint8_t> packed(bytes,0);const auto* src=static_cast<const uint8_t*>(host);
    for(int kb=0;kb<nb;++kb)for(int tile=0;tile<tiles;++tile)for(int lane=0;lane<64;++lane) {
        const int row=tile*16+lane%16;if(row>=no)continue;
        const auto* b=src+(size_t(row)*nb+kb)*34;
        std::memcpy(packed.data()+(size_t(kb)*tiles+tile)*512+lane*8,b+2+(lane/16)*8,8);
        if(lane<16)std::memcpy(packed.data()+qs+((size_t(kb)*tiles+tile)*16+lane)*2,b,2);
    }
    Entry e;e.ni=ni;e.no=no;
    try { check(cudaMalloc(&e.packed,bytes));check(cudaMemcpy(e.packed,packed.data(),bytes,cudaMemcpyHostToDevice));
        const auto scratch=q8_mmac_workspace_bytes(no);if(scratch)check(cudaMalloc(&e.partial,scratch));
        entries().emplace(canonical,e);
    } catch(...) { if(e.partial)cudaFree(e.partial);if(e.packed)cudaFree(e.packed);throw; }
    std::fprintf(stderr,"strata q8 mmac: packed %d->%d, extra weights=%llu workspace=%llu\n",ni,no,(unsigned long long)bytes,(unsigned long long)q8_mmac_workspace_bytes(no));
    return bytes;
}
void q8_mmac_unregister(const void* key) {auto it=entries().find(key);if(it==entries().end())return;
    if(it->second.partial)cudaFree(it->second.partial);cudaFree(it->second.packed);entries().erase(it);
}
bool q8_mmac_try(const void* key,const void* x,float* y,int ni,int no,int t,void* stream) {
    if(!q8_mmac_enabled() || t<1 || t>5)return false;
    const bool stable=stable_widths();
    if(!stable && t==1 && no<65536)return false;
    auto it=entries().find(key);if(it==entries().end())return false;const Entry&e=it->second;
    if(e.ni!=ni || e.no!=no)throw std::runtime_error("Q8 MMAC shape mismatch");
    int split=1;if(no<65536) {
        // Adaptive verification changes T between requests. Keeping the same kernel path and
        // split for every width preserves each row's accumulation order, including T=1.
        if(stable) {
            if(ni==6144 || no==6144)split=16;
            else if(no==12288)split=8;
            else split=4;
        } else {
            if(ni==6144)split=t==4?8:16;
            else if(no==6144)split=t==4?4:16;
            else if(no==12288)split=t==4?4:8;
            else split=t==4?8:4;
        }
    }
    auto s=static_cast<cudaStream_t>(stream);const auto*a=static_cast<const Act*>(x);
    switch(t) {
#define GO(T) case T:run<T>(e,a,y,split,s);break;
        GO(1) GO(2) GO(3) GO(4) GO(5)
#undef GO
    }
    check(cudaGetLastError());
    static bool shown[6]={};if(!shown[t]){shown[t]=true;std::fprintf(stderr,"strata q8 mmac: dispatch T=%d split=%d stable=%d\n",t,split,int(stable));}
    return true;
}
}
