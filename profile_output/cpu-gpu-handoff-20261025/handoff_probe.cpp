// Standalone hypothesis probe. No production dispatch or engine change.
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <thread>
#include <vector>
#define CK(x) do { auto probe_status=(x); if(probe_status!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(probe_status));std::exit(2);} } while(0)
#include "production_copy.inc"
#include "production_combine.inc"

// Same expert order and explicit FMA as production. Read a CPU row directly,
// instead of materializing that row in VRAM then consuming it in a second node.
__global__ void direct_combine(const float4* cpu, const float4* hits,
                               const int32_t* dst,const int32_t* count,
                               const float* weights,const float4* shared,float4* output,
                               const volatile uint32_t* flag,uint32_t want,int n4) {
    const int tk=blockIdx.y;
    __shared__ int hit_mask;
    if(threadIdx.x==0){
        while(*flag<want) strata_spin_pause();
        __threadfence_system();
        hit_mask=0;
    }
    __syncthreads();
    for(int h=threadIdx.x;h<*count;h+=blockDim.x){
        const int r=dst[h];
        if(r/10==tk) atomicOr(&hit_mask,1<<(r%10));
    }
    __syncthreads();
    const int mask=hit_mask;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=n4)return;
    const int base=tk*10*n4+c;
    float4 p=mask&1?hits[base]:cpu[base];
    if(mask&1){p.x=0.f+p.x;p.y=0.f+p.y;p.z=0.f+p.z;p.w=0.f+p.w;}
    float w=weights[tk*10];
    float4 sum=make_float4(p.x*w,p.y*w,p.z*w,p.w*w);
#pragma unroll
    for(int j=1;j<10;j++){
        p=mask&(1<<j)?hits[base+j*n4]:cpu[base+j*n4];
        if(mask&(1<<j)){p.x=0.f+p.x;p.y=0.f+p.y;p.z=0.f+p.z;p.w=0.f+p.w;}
        w=weights[tk*10+j];
        sum.x=fmaf(p.x,w,sum.x);sum.y=fmaf(p.y,w,sum.y);
        sum.z=fmaf(p.z,w,sum.z);sum.w=fmaf(p.w,w,sum.w);
    }
    const float4 s=shared[tk*n4+c];
    sum.x+=s.x;sum.y+=s.y;sum.z+=s.z;sum.w+=s.w;
    output[tk*n4+c]=sum;
}

struct Probe {
    static constexpr int N=2560,K=10,MT=5;
    float *h=nullptr,*m=nullptr,*parts=nullptr,*hits=nullptr,*w=nullptr,*sh=nullptr,*out=nullptr;
    int32_t *dst=nullptr,*count=nullptr;
    uint32_t *flag=nullptr,*dflag=nullptr;
    cudaStream_t stream;
    std::vector<float> hv,hitsv,wv,shv;
    std::vector<int32_t> ds;
    Probe():hv(MT*K*N),hitsv(hv.size()),wv(MT*K),shv(MT*N){
        CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        CK(hipHostMalloc((void**)&h,hv.size()*4,hipHostMallocMapped|hipHostMallocCoherent));
        CK(cudaHostGetDevicePointer((void**)&m,h,0));
        CK(hipHostMalloc((void**)&flag,64,hipHostMallocMapped|hipHostMallocCoherent));
        CK(cudaHostGetDevicePointer((void**)&dflag,flag,0));
        CK(cudaMalloc((void**)&parts,hv.size()*4));CK(cudaMalloc((void**)&hits,hv.size()*4));
        CK(cudaMalloc((void**)&w,wv.size()*4));CK(cudaMalloc((void**)&sh,shv.size()*4));
        CK(cudaMalloc((void**)&out,shv.size()*4));
        CK(cudaMalloc((void**)&dst,MT*K*4));CK(cudaMalloc((void**)&count,4));
        std::mt19937 rng(25);std::uniform_real_distribution<float> dist(-2.f,2.f);
        for(auto& x:hv)x=dist(rng);for(auto& x:hitsv)x=dist(rng);
        for(auto& x:wv)x=dist(rng)/10;for(auto& x:shv)x=dist(rng);
        CK(cudaMemcpy(hits,hitsv.data(),hitsv.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(w,wv.data(),wv.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(sh,shv.data(),shv.size()*4,cudaMemcpyHostToDevice));
    }
    ~Probe(){
        CK(cudaStreamSynchronize(stream));
        for(void* p:{(void*)parts,(void*)hits,(void*)w,(void*)sh,(void*)out,(void*)dst,(void*)count})CK(cudaFree(p));
        CK(cudaFreeHost(h));CK(cudaFreeHost(flag));CK(cudaStreamDestroy(stream));
    }
    void setup(int t,int nh,int seed){
        ds.resize(t*K);for(int i=0;i<t*K;i++)ds[i]=i;
        std::shuffle(ds.begin(),ds.end(),std::mt19937(seed));ds.resize(nh);
        CK(cudaMemcpy(dst,ds.data(),nh*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(count,&nh,4,cudaMemcpyHostToDevice));
        std::memcpy(h,hv.data(),hv.size()*4);_mm_sfence();__atomic_store_n(flag,1u,__ATOMIC_RELEASE);
    }
    void launch(int method,int t,uint32_t want=1){
        if(method==2){
            direct_combine<<<dim3(5,t),128,0,stream>>>((float4*)m,(float4*)hits,dst,count,w,(float4*)sh,(float4*)out,dflag,want,N/4);
        }else{
            if(method==0) wait_flag_ge_kernel<<<1,1,0,stream>>>(dflag,want);
            copy_rows_from_mapped_kernel<<<t*K,128,0,stream>>>((float4*)parts,(volatile float4*)m,N/4,dst,count,method==1?dflag:nullptr,want);
            if(method==0 || t==1){
                add_hits_kernel<<<dim3(8,t*K),256,0,stream>>>(parts,hits,dst,count,N);
                combine_k10_vec4<<<dim3(5,t),128,0,stream>>>((float4*)parts,w,(float4*)sh,(float4*)out,N/4);
            }else{
                combine_k10_vec4_hits<<<dim3(5,t),128,0,stream>>>((float4*)parts,(float4*)hits,dst,count,w,(float4*)sh,(float4*)out,N/4);
            }
        }
        CK(cudaGetLastError());
    }
    std::vector<float> read(int t){
        CK(cudaStreamSynchronize(stream));std::vector<float> o(t*N);
        CK(cudaMemcpy(o.data(),out,o.size()*4,cudaMemcpyDeviceToHost));return o;
    }
    void check(const std::vector<float>& o,int t,int round,int method){
        std::vector<float> ref(t*N);
        for(int tk=0;tk<t;tk++)for(int c=0;c<N;c++){
            float sum=0;
            for(int j=0;j<K;j++){
                const int row=tk*K+j;const bool hit=std::find(ds.begin(),ds.end(),row)!=ds.end();
                float v=hit?0.f+hitsv[row*N+c]:hv[row*N+c];
                if(j==0)sum=v*wv[row];else sum=std::fma(v,wv[row],sum);
            }
            ref[tk*N+c]=sum+shv[tk*N+c];
        }
        if(std::memcmp(o.data(),ref.data(),o.size()*4)){
            double worst=0;int differences=0;
            for(size_t i=0;i<o.size();i++){if(o[i]!=ref[i])differences++;worst=std::max(worst,(double)std::abs(o[i]-ref[i]));}
            std::fprintf(stderr,"numeric mismatch t=%d round=%d method=%d differences=%d max=%g\n",t,round,method,differences,worst);std::exit(3);
        }
    }
    cudaGraphExec_t graph(int method,int t){
        cudaGraph_t g;cudaGraphExec_t e;
        CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        launch(method,t);CK(cudaStreamEndCapture(stream,&g));
        CK(hipGraphInstantiateWithFlags(&e,g,0));CK(cudaGraphDestroy(g));return e;
    }
    double bench(cudaGraphExec_t e){
        constexpr int R=150;
        for(int i=0;i<20;i++)CK(cudaGraphLaunch(e,stream));CK(cudaStreamSynchronize(stream));
        cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a,stream));for(int i=0;i<R;i++)CK(cudaGraphLaunch(e,stream));
        CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
        CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/R;
    }
};
int main(){
    Probe p;int cases=0;
    for(int t:{1,2,3,4,5})for(int pct:{0,30,60,90,100}){
        const int nh=t*10*pct/100;p.setup(t,nh,25+t+pct);
        for(int method:{0,1,2}){p.launch(method,t);p.check(p.read(t),t,-1,method);cases++;}
        cudaGraphExec_t graphs[3]={p.graph(0,t),p.graph(1,t),p.graph(2,t)};
        for(int r=0;r<7;r++)for(int x=0;x<3;x++){
            int method=(x+r)%3;double us=p.bench(graphs[method]);
            p.check(p.read(t),t,r,method);
            std::printf("{\"kind\":\"timing\",\"t\":%d,\"hits\":%d,\"rows\":%d,\"round\":%d,\"method\":%d,\"us\":%.8f}\n",t,nh,t*10,r,method,us);std::fflush(stdout);
        }
        for(auto graph:graphs)CK(cudaGraphExecDestroy(graph));
        // Real late producer, fresh payload each round; no synchronization or
        // driver query before publishing the flag. Flags carry the round epoch.
        for(int method:{0,1,2})for(int r=0;r<12;r++){
            __atomic_store_n(p.flag,0u,__ATOMIC_RELEASE);
            const int idx=(r*7919)%(t*10*p.N);p.hv[idx]+=0.03125f;
            p.launch(method,t,r+2);
            std::this_thread::sleep_for(std::chrono::microseconds(50+(r%5)*50));
            std::memcpy(p.h,p.hv.data(),p.hv.size()*4);_mm_sfence();
            __atomic_store_n(p.flag,uint32_t(r+2),__ATOMIC_RELEASE);
            p.check(p.read(t),t,r,method);cases++;
        }
    }
    std::printf("{\"kind\":\"complete\",\"numeric_cases\":%d,\"failures\":0}\n",cases);
}
