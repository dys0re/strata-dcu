#include "strata/kernels/bf16_small.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/prefill/gemm.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
    using namespace strata::kernels;
    try {
        cudaStream_t s;ck(cudaStreamCreate(&s));
        uint16_t *dx=nullptr,*dw=nullptr,*dy=nullptr;
        ck(cudaMalloc((void**)&dx,32));ck(cudaMalloc((void**)&dw,32));ck(cudaMalloc((void**)&dy,32));
        ck(cudaMemset(dx,0,32));ck(cudaMemset(dw,0,32));ck(cudaMemset(dy,0,32));
        if(!bf16_small_mm(dx,dw,dy,1,1,4,s)) {
            ck(cudaFree(dx));ck(cudaFree(dw));ck(cudaFree(dy));ck(cudaStreamDestroy(s));
            std::puts("SKIP BF16 small mm: unsupported GPU");return 0;
        }
        ck(cudaStreamSynchronize(s));ck(cudaFree(dx));ck(cudaFree(dw));ck(cudaFree(dy));
        std::puts("k,n,columns,small_us,blas_us,speedup,sampled_scaled_error");
        for(auto [k,n]:{std::pair{4,1},{32,5},{96,33},{2560,256},{2560,640},
                       {2560,1024},{2560,2560},{2560,4096},{4096,2560},
                       {2560,7680},{7680,2560},{12800,2560},{2560,248320}}) {
            std::vector<uint16_t> x(size_t(k)*8),w(size_t(k)*n);
            uint32_t state=1729;
            auto value=[&] {state=1664525u*state+1013904223u;return float(int(state>>24)-128)/1024.f;};
            for(auto& v:x)v=bf16_from_f32(value());
            for(auto& v:w)v=bf16_from_f32(value());
            ck(cudaMalloc((void**)&dx,x.size()*2));ck(cudaMalloc((void**)&dw,w.size()*2));
            ck(cudaMalloc((void**)&dy,(size_t(n)*8+16)*2));
            float* df=nullptr;uint16_t* ref=nullptr;
            ck(cudaMalloc((void**)&df,size_t(n)*8*4));ck(cudaMalloc((void**)&ref,size_t(n)*8*2));
            ck(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
            ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
            strata::prefill::Gemm gemm;std::string err;
            if(!gemm.init(s,0,err))throw std::runtime_error(err);
            std::vector<uint16_t> single(size_t(n)*8);
            for(int c=0;c<8;++c) {
                if(!bf16_small_mm(dx+size_t(c)*k,dw,ref+size_t(c)*n,1,n,k,s))throw std::runtime_error("dispatch");
            }
            ck(cudaStreamSynchronize(s));ck(cudaMemcpy(single.data(),ref,single.size()*2,cudaMemcpyDeviceToHost));
            for(int t=1;t<=8;++t) {
                ck(cudaMemset(dy,0xa5,(size_t(n)*8+16)*2));
                if(!bf16_small_mm(dx,dw,dy,t,n,k,s))throw std::runtime_error("dispatch");
                ck(cudaStreamSynchronize(s));
                std::vector<uint16_t> actual(size_t(n)*t+16);
                ck(cudaMemcpy(actual.data(),dy,actual.size()*2,cudaMemcpyDeviceToHost));
                if(std::memcmp(actual.data(),single.data(),size_t(n)*t*2))throw std::runtime_error("column reuse mismatch");
                for(size_t i=size_t(n)*t;i<actual.size();++i)
                    if(actual[i]!=0xa5a5u)throw std::runtime_error("output guard");
                double worst=0;
                for(int c=0;c<t;++c)for(int sample=0;sample<std::min(n,127);++sample) {
                    const int r=int((size_t(sample)*3571)%n);double dot=0,abs_sum=0;
                    for(int i=0;i<k;++i) {
                        const double a=f32_from_bf16(x[size_t(c)*k+i]),b=f32_from_bf16(w[size_t(r)*k+i]);
                        dot+=a*b;abs_sum+=std::abs(a*b);
                    }
                    const double got=f32_from_bf16(actual[size_t(c)*n+r]);
                    const double e=std::abs(got-dot)/(1+abs_sum);
                    worst=std::max(worst,e);
                    // BF16 output rounding plus FP32 accumulation error. Using
                    // only a relative L1 bound would be weak near cancellation.
                    const double tolerance=0.004*std::abs(dot)+1e-5*abs_sum+1e-7;
                    if(!std::isfinite(got)||std::abs(got-dot)>tolerance)
                        throw std::runtime_error("scalar reference");
                }
                auto measure=[&](bool fast) {
                    cudaGraph_t g;cudaGraphExec_t exec;cudaEvent_t b,e;
                    ck(cudaEventCreate(&b));ck(cudaEventCreate(&e));
                    auto launch=[&] {
                        if(fast)bf16_small_mm(dx,dw,dy,t,n,k,s);
                        else {gemm.bf16(dx,dw,df,t,n,k);f32_to_bf16_bulk(df,dy,size_t(t)*n,s);}
                    };
                    launch();ck(cudaStreamSynchronize(s));
                    ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
                    for(int i=0;i<8;++i)launch();
                    ck(cudaStreamEndCapture(s,&g));ck(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
                    ck(cudaGraphLaunch(exec,s));ck(cudaStreamSynchronize(s));
                    std::vector<float> times;
                    for(int rep=0;rep<5;++rep) {
                        ck(cudaEventRecord(b,s));ck(cudaGraphLaunch(exec,s));ck(cudaEventRecord(e,s));ck(cudaEventSynchronize(e));
                        float ms=0;ck(cudaEventElapsedTime(&ms,b,e));times.push_back(ms*1000/8);
                    }
                    std::sort(times.begin(),times.end());
                    if(fast) {
                        std::vector<uint16_t> graph(size_t(t)*n);
                        ck(cudaMemcpy(graph.data(),dy,graph.size()*2,cudaMemcpyDeviceToHost));
                        if(std::memcmp(graph.data(),actual.data(),graph.size()*2))throw std::runtime_error("graph mismatch");
                    }
                    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(g));ck(cudaEventDestroy(b));ck(cudaEventDestroy(e));
                    return times[2];
                };
                float fast=measure(true),blas=measure(false);
                std::printf("%d,%d,%d,%.3f,%.3f,%.3f,%.4g\n",k,n,t,fast,blas,blas/fast,worst);std::fflush(stdout);
            }
            ck(cudaFree(df));ck(cudaFree(ref));ck(cudaFree(dy));ck(cudaFree(dw));ck(cudaFree(dx));
        }
        ck(cudaStreamDestroy(s));std::puts("PASS BF16_SMALL_MM");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
