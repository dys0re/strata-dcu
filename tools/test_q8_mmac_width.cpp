#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_mmac.hpp"
#include "strata/kernels/execution_weight_ops.hpp"
#include "strata/kernels/draft_head_proxy.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <functional>

void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}

double measure(const std::function<void()>& call,cudaStream_t stream) {
    cudaGraph_t graph;cudaGraphExec_t exec;cudaEvent_t begin,end;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));call();
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    for(int i=0;i<5;++i)check(cudaGraphLaunch(exec,stream));
    check(cudaEventRecord(begin,stream));
    for(int i=0;i<50;++i)check(cudaGraphLaunch(exec,stream));
    check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float ms=0;
    check(cudaEventElapsedTime(&ms,begin,end));
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));return ms*1000/50;
}

int main(int argc,char** argv) {
    const bool require_stable=argc==2 && std::strcmp(argv[1],"--require-stable")==0;
    if(!strata::kernels::q8_mmac_enabled()) {
        std::puts("SKIP: requires gfx936 and STRATA_Q8_MMAC=1");return 77;
    }
    const bool timing=std::getenv("STRATA_EXECUTION_TIMING")!=nullptr;
    std::mt19937 rng(936);
    const int shapes[][2]={{2560,2560},{2560,6144},{2560,10240},{2560,12288},{6144,2560},{2560,248320},{2560,2051}};
    size_t all_mismatches=0;
    for (const auto& shape:shapes) {
        const int ni=shape[0],no=shape[1];
        std::vector<unsigned char> weights(size_t(ni)*no/32*34);
        for (size_t i=0;i<weights.size();i+=34) {
            const uint16_t scale=uint16_t(0x1000+(rng()%0x2000));
            std::memcpy(weights.data()+i,&scale,2);
            for (int j=0;j<32;++j) weights[i+2+j]=uint8_t(rng());
        }
        std::vector<float> input(ni*8);
        for (float& x:input) x=std::normal_distribution<float>(0,1)(rng);
        void* dw=nullptr;void* xq=nullptr;float* dx=nullptr;float* dy=nullptr;cudaStream_t stream;
        check(cudaMalloc(&dw,weights.size()));check(cudaMalloc(&dx,input.size()*sizeof(float)));
        check(cudaMalloc(&dy,size_t(no)*8*sizeof(float)));
        check(cudaMalloc(&xq,strata::kernels::native_q8_1_bytes(ni,8)));
        check(cudaMemcpy(dw,weights.data(),weights.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaStreamCreate(&stream));
        strata::kernels::native_quantize_q8_1(dx,xq,ni,8,stream);check(cudaStreamSynchronize(stream));
        std::vector<std::vector<float>> reference(8),got(8);
        double ref_us[8]={};
        for(int t=1;t<=8;++t) {
            if(timing)ref_us[t-1]=measure([&]{strata::kernels::native_mmvq(8,dw,xq,dy,ni,no,t,stream);},stream);
            reference[t-1].resize(size_t(no)*t);
            strata::kernels::native_mmvq(8,dw,xq,dy,ni,no,t,stream);check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(reference[t-1].data(),dy,size_t(no)*t*sizeof(float),cudaMemcpyDeviceToHost));
        }
        auto execution=strata::kernels::q8_mmac_upload_host(weights.data(),ni,no);
        if(execution.bytes!=strata::kernels::q8_mmac_weight_bytes(ni,no))return 3;
        // Independent canonical dequantizer vs sliced execution-layout reads, including padded output rows.
        const int row0=no/2-3,nr=37,ld=ni+64;
        uint16_t *plain=nullptr,*packed=nullptr;
        check(cudaMalloc(&plain,size_t(nr)*ni*2));check(cudaMalloc(&packed,size_t(nr)*ld*2));
        strata::kernels::dequant_f16(8,dw,row0,nr,ni,plain,stream);
        if(!strata::kernels::execution_dequant_f16(execution,row0,nr,ld,packed,stream))return 14;
        check(cudaStreamSynchronize(stream));
        std::vector<uint16_t> a(size_t(nr)*ni),b(size_t(nr)*ld);
        check(cudaMemcpy(a.data(),plain,a.size()*2,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(),packed,b.size()*2,cudaMemcpyDeviceToHost));
        for(int r=0;r<nr;++r)if(std::memcmp(a.data()+size_t(r)*ni,b.data()+size_t(r)*ld,size_t(ni)*2))return 8;
        cudaFree(plain);cudaFree(packed);
        // MTP subset reconstruction must preserve every code and scale byte, including duplicate and last rows.
        const int32_t ids[]={no-1,0,no/2,15,16,no-1};int32_t* dids=nullptr;void* subset=nullptr;
        const size_t rb=size_t(ni/32)*34;
        check(cudaMalloc(&dids,sizeof(ids)));check(cudaMalloc(&subset,6*rb));
        check(cudaMemcpy(dids,ids,sizeof(ids),cudaMemcpyHostToDevice));
        strata::kernels::execution_gather_rows(execution,dids,6,(uint8_t*)subset,stream);
        check(cudaStreamSynchronize(stream));std::vector<uint8_t> gathered(6*rb);
        check(cudaMemcpy(gathered.data(),subset,gathered.size(),cudaMemcpyDeviceToHost));
        for(int r=0;r<6;++r)if(std::memcmp(gathered.data()+r*rb,weights.data()+size_t(ids[r])*rb,rb))return 9;
        cudaFree(dids);cudaFree(subset);
        if(ni==2560) {
            const size_t pb=strata::kernels::draft_head_proxy_bytes(no),sb=strata::kernels::draft_head_proxy_scratch_bytes(no);
            void *pa=nullptr,*pbuff=nullptr,*scratch=nullptr;
            check(cudaMalloc(&pa,pb));check(cudaMalloc(&pbuff,pb));check(cudaMalloc(&scratch,sb));
            check(cudaMemsetAsync(pa,0,pb,stream));check(cudaMemsetAsync(pbuff,0,pb,stream));
            strata::kernels::draft_head_proxy_pack(8,dw,pa,no,stream);
            strata::kernels::draft_head_proxy_pack(execution,pbuff,no,stream);
            check(cudaStreamSynchronize(stream));
            std::vector<uint8_t> ap(pb),bp(pb);
            check(cudaMemcpy(ap.data(),pa,pb,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(bp.data(),pbuff,pb,cudaMemcpyDeviceToHost));
            if(ap!=bp)return 13;
            // Includes shortlist rescoring from the original execution layout.
            if(!strata::kernels::draft_head_proxy_logits(8,pa,dw,xq,dy,no,1,nullptr,scratch,stream))return 10;
            check(cudaStreamSynchronize(stream));std::vector<float> la(no),lb(no);
            check(cudaMemcpy(la.data(),dy,size_t(no)*4,cudaMemcpyDeviceToHost));
            if(!strata::kernels::draft_head_proxy_logits(execution,pbuff,xq,dy,no,1,nullptr,scratch,stream))return 11;
            check(cudaStreamSynchronize(stream));check(cudaMemcpy(lb.data(),dy,size_t(no)*4,cudaMemcpyDeviceToHost));
            if(std::memcmp(la.data(),lb.data(),size_t(no)*4))return 12;
            cudaFree(pa);cudaFree(pbuff);cudaFree(scratch);
        }
        cudaFree(dw);dw=nullptr; // execution can no longer fall back to a hidden canonical device copy.
        double max_relative=0;
        for(int t=1;t<=8;++t) {
            if(timing) {
                const double us=measure([&]{strata::kernels::native_mmvq(execution,xq,dy,ni,no,t,stream);},stream);
                std::printf("TIMING %d %d T=%d gguf_us=%.3f execution_us=%.3f\n",ni,no,t,ref_us[t-1],us);
            }
            cudaGraph_t graph;cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            strata::kernels::native_mmvq(execution,xq,dy,ni,no,t,stream);
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
        std::printf("SHAPE %d %d prefix_bit_mismatches=%zu max_relative_error=%.9g layout_readback=PASS proxy=%s widths=1..8\n",ni,no,mismatch,max_relative,ni==2560?"PASS":"N/A");std::fflush(stdout);
        if(max_relative>1e-5)return 6;
        cudaFree(execution.partial);cudaFree(const_cast<void*>(execution.data));
        cudaFree(dw);cudaFree(dx);cudaFree(dy);cudaFree(xq);cudaStreamDestroy(stream);
    }
    std::printf("MMAC_WIDTH_%s mismatches=%zu\n",all_mismatches?"DIFFERENT":"STABLE",all_mismatches);
    return require_stable && all_mismatches ? 7 : 0;
}
