#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_mmac.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}

int main(int argc,char** argv) {
    const bool require_stable=argc==2 && std::strcmp(argv[1],"--require-stable")==0;
    std::mt19937 rng(936);
    const int shapes[][2]={{2560,2560},{2560,6144},{2560,10240},{2560,12288},{6144,2560},{2560,248320}};
    size_t all_mismatches=0;
    for (const auto& shape:shapes) {
        const int ni=shape[0],no=shape[1];
        std::vector<unsigned char> weights(size_t(ni)*no/32*34);
        for (size_t i=0;i<weights.size();i+=34) {
            const uint16_t scale=uint16_t(0x1000+(rng()%0x2000));
            std::memcpy(weights.data()+i,&scale,2);
            for (int j=0;j<32;++j) weights[i+2+j]=uint8_t(rng());
        }
        std::vector<float> input(ni*5);
        for (float& x:input) x=std::normal_distribution<float>(0,1)(rng);
        void* dw=nullptr;void* xq=nullptr;float* dx=nullptr;float* dy=nullptr;cudaStream_t stream;
        check(cudaMalloc(&dw,weights.size()));check(cudaMalloc(&dx,input.size()*sizeof(float)));
        check(cudaMalloc(&dy,size_t(no)*5*sizeof(float)));
        check(cudaMalloc(&xq,strata::kernels::native_q8_1_bytes(ni,5)));
        check(cudaMemcpy(dw,weights.data(),weights.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaStreamCreate(&stream));
        strata::kernels::native_quantize_q8_1(dx,xq,ni,5,stream);check(cudaStreamSynchronize(stream));
        std::vector<std::vector<float>> reference(5),got(5);
        for(int t=1;t<=5;++t) {
            reference[t-1].resize(size_t(no)*t);
            strata::kernels::native_mmvq(8,dw,xq,dy,ni,no,t,stream);check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(reference[t-1].data(),dy,size_t(no)*t*sizeof(float),cudaMemcpyDeviceToHost));
        }
        if(strata::kernels::q8_mmac_register_host(dw,weights.data(),ni,no)!=weights.size())return 3;
        double max_relative=0;
        for(int t=1;t<=5;++t) {
            cudaGraph_t graph;cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            strata::kernels::native_mmvq(8,dw,xq,dy,ni,no,t,stream);
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            got[t-1].resize(size_t(no)*t);
            for(int repeat=0;repeat<2;++repeat) {
                check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
                std::vector<float> out(size_t(no)*t);check(cudaMemcpy(out.data(),dy,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
                double error=0,norm=0;
                for(size_t i=0;i<out.size();++i) {
                    if(!std::isfinite(out[i]))return 4;
                    const double delta=double(out[i])-reference[t-1][i];error+=delta*delta;norm+=double(reference[t-1][i])*reference[t-1][i];
                }
                max_relative=std::max(max_relative,std::sqrt(error/norm));
                if(repeat && std::memcmp(out.data(),got[t-1].data(),out.size()*sizeof(float)))return 5;
                got[t-1]=std::move(out);
            }
            cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
        }
        size_t mismatch=0;
        for(int t=2;t<=5;++t)for(int row=0;row<t-1;++row)for(int i=0;i<no;++i)
            mismatch+=std::memcmp(&got[t-1][size_t(row)*no+i],&got[t-2][size_t(row)*no+i],sizeof(float))!=0;
        all_mismatches+=mismatch;
        std::printf("SHAPE %d %d prefix_bit_mismatches=%zu max_relative_error=%.9g\n",ni,no,mismatch,max_relative);std::fflush(stdout);
        if(max_relative>1e-5)return 6;
        strata::kernels::q8_mmac_unregister(dw);
        cudaFree(dw);cudaFree(dx);cudaFree(dy);cudaFree(xq);cudaStreamDestroy(stream);
    }
    std::printf("MMAC_WIDTH_%s mismatches=%zu\n",all_mismatches?"DIFFERENT":"STABLE",all_mismatches);
    return require_stable && all_mismatches ? 7 : 0;
}
