// Standalone production-dispatch diagnostic. GPU graph timings are kernel-only;
// these are not model TPS. All columns must match separate vec4 calls bitwise.
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
int main() {
    using namespace strata::kernels;
    try {
        cudaStream_t stream;
        cudaEvent_t start, end;
        ck(cudaStreamCreate(&stream)); ck(cudaEventCreate(&start)); ck(cudaEventCreate(&end));
        std::puts("n_in,n_out,columns,median_us,min_us,max_us,bitwise,guard");
        for (auto [k,n] : {std::pair{2560,248320}, {2560,10240}, {2560,6144},
                           {2560,512}, {6144,2560}, {640,2560}, {32,5},
                           {2560,640}, {2560,6145}, {1024,12288}, {2560,2560}, {2560,12288}}) {
            std::vector<uint16_t> w(size_t(k)*n);
            for (size_t i=0;i<w.size();++i)
                w[i] = uint16_t(0x3b00u + ((i*131u+i/67u)%1536u)) | uint16_t((i&1u)<<15);
            std::vector<float> x(size_t(k)*8);
            for (size_t i=0;i<x.size();++i) x[i]=std::cos(float(i)*.17f);
            void *dw=nullptr,*dq=nullptr; float *dx=nullptr,*dy=nullptr,*ref=nullptr;
            ck(cudaMalloc(&dw,w.size()*2)); ck(cudaMalloc((void**)&dx,x.size()*4));
            ck(cudaMalloc(&dq,native_q8_1_bytes(k,8)));
            ck(cudaMalloc((void**)&dy,(size_t(n)*8+16)*4));
            ck(cudaMalloc((void**)&ref,size_t(n)*8*4));
            ck(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
            ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
            native_quantize_q8_1(dx,dq,k,8,stream);
            for (int c=0;c<8;++c)
                native_mmvq(30,dw,(uint8_t*)dq+size_t(c)*native_q8_1_bytes(k,1),ref+size_t(c)*n,k,n,1,stream);
            ck(cudaStreamSynchronize(stream));
            for (int t=1;t<=8;++t) {
                ck(cudaMemset(dy,0xa5,(size_t(n)*8+16)*4));
                native_mmvq(30,dw,dq,dy,k,n,t,stream);
                ck(cudaStreamSynchronize(stream));
                cudaGraph_t graph; cudaGraphExec_t exec;
                constexpr int calls=16;
                ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
                for (int i=0;i<calls;++i) native_mmvq(30,dw,dq,dy,k,n,t,stream);
                ck(cudaStreamEndCapture(stream,&graph));
                ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
                std::vector<float> times;
                for (int sample=0;sample<7;++sample) {
                    ck(cudaEventRecord(start,stream)); ck(cudaGraphLaunch(exec,stream));
                    ck(cudaEventRecord(end,stream)); ck(cudaEventSynchronize(end));
                    float ms=0; ck(cudaEventElapsedTime(&ms,start,end)); times.push_back(ms*1000/calls);
                }
                std::sort(times.begin(),times.end());
                std::vector<uint32_t> got(size_t(n)*t+16),expected(size_t(n)*t);
                ck(cudaMemcpy(got.data(),dy,got.size()*4,cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(expected.data(),ref,expected.size()*4,cudaMemcpyDeviceToHost));
                if(std::memcmp(got.data(),expected.data(),expected.size()*4))
                    throw std::runtime_error("multi-column vs single-column mismatch");
                for(size_t i=expected.size();i<got.size();++i)
                    if(got[i]!=0xa5a5a5a5u) throw std::runtime_error("output guard overwritten");
                for(auto bits:expected) { float f; std::memcpy(&f,&bits,4);
                    if(!std::isfinite(f)) throw std::runtime_error("nonfinite output"); }
                std::printf("%d,%d,%d,%.3f,%.3f,%.3f,1,1\n",k,n,t,times[3],times.front(),times.back());
                std::fflush(stdout);
                ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
            }
            ck(cudaFree(ref)); ck(cudaFree(dy)); ck(cudaFree(dq)); ck(cudaFree(dx)); ck(cudaFree(dw));
        }
        ck(cudaEventDestroy(end)); ck(cudaEventDestroy(start)); ck(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
