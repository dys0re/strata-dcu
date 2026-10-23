// Native execution of the pinned upstream DFlash2 network. See third_party/dflash/VERSION.
// Its trained operators (including BF16 rounding boundaries) follow model.py; no Python runtime.
#include "strata/core/dflash_native.hpp"
#include "strata/core/native_head.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace strata::core {
namespace {
using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;
constexpr int H=2560, Q=4096, KV=1024, D=128, FF=7680, V=248320;
constexpr int RANK=256, TOP=16, WINDOW=2048, RING=WINDOW+8, CHUNK=32;
using Clock=std::chrono::steady_clock;
__device__ float bf(float x) { return f32_from_bf16(bf16_from_f32(x)); }

__global__ void cast16(const float* x,uint16_t* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=bf16_from_f32(x[i]);
}
__global__ void features16(const float* x,uint16_t* y,int cap,int row0,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*5*H)return;
    int c=i%H,j=(i/H)%5,t=i/(5*H);float s=0;
    for(int h=0;h<4;++h)s+=x[((size_t(j)*cap+row0+t)*4+h)*H+c];
    y[i]=bf16_from_f32(s*.25f);
}
__global__ void rms(const uint16_t* x,const uint16_t* w,uint16_t* y,int cols) {
    __shared__ float a[256];int t=blockIdx.x,l=threadIdx.x;float s=0;
    for(int c=l;c<cols;c+=256){float v=f32_from_bf16(x[t*cols+c]);s+=v*v;}
    a[l]=s;__syncthreads();for(int n=128;n;n>>=1){if(l<n)a[l]+=a[l+n];__syncthreads();}
    float r=rsqrtf(a[0]/cols+1e-6f);
    for(int c=l;c<cols;c+=256)y[t*cols+c]=bf16_from_f32(bf(f32_from_bf16(x[t*cols+c])*r)*f32_from_bf16(w[c]));
}
__global__ void add(const uint16_t* a,const uint16_t* b,uint16_t* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=bf16_from_f32(f32_from_bf16(a[i])+f32_from_bf16(b[i]));
}
__global__ void swiglu(const uint16_t* g,const uint16_t* u,uint16_t* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=f32_from_bf16(g[i]);out[i]=bf16_from_f32(bf(v/(1+expf(-v)))*f32_from_bf16(u[i]));}
}
// One projection produces both the input and output convolution filters. Each offset has
// distinct BF16 mul/add/addcmul rounding, as in _grouped_dynamic_convolve.
__global__ void conv(const uint16_t* x,const uint16_t* dyn,const uint16_t* base,
                     uint16_t* out,int rows,int side) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*H)return;
    int t=i/H,c=i%H;float acc=0;
    for(int o=0;o<2;++o){float v=t>=o?f32_from_bf16(x[(t-o)*H+c]):0;
        float b=f32_from_bf16(base[(side*2+o)*H+c]);
        float d=f32_from_bf16(dyn[t*640+side*320+o*160+c/16]);
        acc=bf(acc+bf(b*v));acc=bf(fmaf(d,v,acc));
    }out[i]=bf16_from_f32(acc);
}
// RoPE is applied after Q/K head RMSNorm. cos/sin and each product are BF16.
__global__ void rope(const uint16_t* x,uint16_t* y,int rows,int width,
                     const int64_t* position,int offset) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*width)return;
    int c=i%D,t=i/width,base=i-c;
    float theta=float(*position+offset+t)*powf(10000.f,-float((c%64)*2)/128.f);
    float cs=bf(cosf(theta)),sn=bf(sinf(theta));
    float a=f32_from_bf16(x[i]),b=f32_from_bf16(x[base+(c+64)%128]);if(c<64)b=-b;
    y[i]=bf16_from_f32(bf(a*cs)+bf(b*sn));
}
__global__ void write_kv(const uint16_t* k,const uint16_t* v,uint16_t* kc,uint16_t* vc,
                         int rows,const int64_t* position,int offset) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*KV)return;
    int t=i/KV,c=i%KV;int slot=int((*position+offset+t)%RING);
    kc[slot*KV+c]=k[i];vc[slot*KV+c]=v[i];
}
// Small-query GQA with the checkpoint's symmetric noncausal sliding window.
// Cache slots are absolute-position modulo RING; no stale/ephemeral row is visible.
__global__ void attention(const uint16_t* q,const uint16_t* kc,const uint16_t* vc,
                          uint16_t* out,int rows,const int64_t* position) {
    __shared__ float scores[RING],reduce[256];int l=threadIdx.x,h=blockIdx.x%32,t=blockIdx.x/32;
    int64_t qp=*position+t,lo=max(int64_t(0),qp-WINDOW+1),hi=*position+rows;
    int n=int(hi-lo),kh=h/4;float largest=-INFINITY;
    for(int j=l;j<n;j+=256){int slot=int((lo+j)%RING);float s=0;
        for(int d=0;d<D;++d)s+=f32_from_bf16(q[t*Q+h*D+d])*f32_from_bf16(kc[slot*KV+kh*D+d]);
        scores[j]=s*(1.f/sqrtf(float(D)));largest=fmaxf(largest,scores[j]);
    }
    reduce[l]=largest;__syncthreads();for(int k=128;k;k>>=1){if(l<k)reduce[l]=fmaxf(reduce[l],reduce[l+k]);__syncthreads();}
    float m=reduce[0],sum=0;for(int j=l;j<n;j+=256){scores[j]=expf(scores[j]-m);sum+=scores[j];}
    reduce[l]=sum;__syncthreads();for(int k=128;k;k>>=1){if(l<k)reduce[l]+=reduce[l+k];__syncthreads();}
    if(l<D){float s=0;for(int j=0;j<n;++j){int slot=int((lo+j)%RING);s+=scores[j]*f32_from_bf16(vc[slot*KV+kh*D+l]);}
        out[t*Q+h*D+l]=bf16_from_f32(s/reduce[0]);}
}
// Exact top-16 over BF16 head logits; each block contributes 16 local candidates.
__global__ void top_parts(const uint16_t* logits,float* scores,int32_t* ids,int parts) {
    __shared__ float val[256];__shared__ int idx[256];int l=threadIdx.x,t=blockIdx.y;
    int i=blockIdx.x*256+l;float v=i<V?f32_from_bf16(logits[t*V+i]):-INFINITY;
    for(int k=0;k<TOP;++k){val[l]=v;idx[l]=i;__syncthreads();
        for(int s=128;s;s>>=1){if(l<s&&(val[l+s]>val[l]||(val[l+s]==val[l]&&idx[l+s]<idx[l]))){val[l]=val[l+s];idx[l]=idx[l+s];}__syncthreads();}
        if(l==0){int p=(t*parts+blockIdx.x)*TOP+k;scores[p]=val[0];ids[p]=idx[0];}
        if(i==idx[0])v=-INFINITY;__syncthreads();
    }
}
__global__ void top_merge(const float* scores,const int32_t* ids,float* best,int32_t* candidates,int parts) {
    // Parallel merge: each lane sorts its small partition, then the block emits 16 winners.
    __shared__ float values[256];__shared__ int indices[256];
    int t=blockIdx.x,l=threadIdx.x;float s[TOP];int ix[TOP];
    for(int k=0;k<TOP;++k){s[k]=-INFINITY;ix[k]=-1;}
    for(int j=l;j<parts*TOP;j+=256){float v=scores[t*parts*TOP+j];int id=ids[t*parts*TOP+j];
        for(int k=0;k<TOP;++k)if(v>s[k]||(v==s[k]&&id<ix[k])){
            for(int z=TOP-1;z>k;--z){s[z]=s[z-1];ix[z]=ix[z-1];}s[k]=v;ix[k]=id;break;}
    }
    for(int k=0;k<TOP;++k){values[l]=s[0];indices[l]=ix[0];__syncthreads();
        for(int step=128;step;step>>=1){if(l<step&&(values[l+step]>values[l]||
            (values[l+step]==values[l]&&indices[l+step]<indices[l]))){values[l]=values[l+step];indices[l]=indices[l+step];}__syncthreads();}
        if(l==0){best[t*TOP+k]=values[0];candidates[t*TOP+k]=indices[0];}
        if(ix[0]==indices[0]){for(int z=0;z<TOP-1;++z){s[z]=s[z+1];ix[z]=ix[z+1];}s[TOP-1]=-INFINITY;ix[TOP-1]=-1;}
        __syncthreads();
    }
}
__global__ void select_path(const uint16_t* hidden,const float* unary,const int32_t* candidates,
                           const uint16_t* pred,const uint16_t* succ,const int32_t* anchor,
                           int32_t* out,int rows) {
    __shared__ float partial[256];__shared__ float scores[TOP];__shared__ int previous;
    int l=threadIdx.x;if(l==0)previous=*anchor;__syncthreads();
    for(int t=0;t<rows;++t){for(int k=0;k<TOP;++k){int id=candidates[t*TOP+k];
            // The codebook product and einsum output are BF16 in the reference selector.
            bool valid=id>=0&&id<V&&previous>=0&&previous<V;
            float v=valid?bf(f32_from_bf16(pred[previous*RANK+l])*f32_from_bf16(hidden[t*RANK+l])):0;
            partial[l]=valid?v*f32_from_bf16(succ[id*RANK+l]):0;__syncthreads();
            for(int s=128;s;s>>=1){if(l<s)partial[l]+=partial[l+s];__syncthreads();}
            if(l==0)scores[k]=valid?bf(unary[t*TOP+k]+bf(partial[0])):-INFINITY;__syncthreads();
        }
        if(l==0){int best=0;for(int k=1;k<TOP;++k)if(scores[k]>scores[best])best=k;previous=candidates[t*TOP+best];out[t]=previous;}__syncthreads();
    }
}
__global__ void widen(const uint16_t* x,float* y,int n) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=f32_from_bf16(x[i]);}
bool check(cudaError_t e,const char* what,std::string& err){if(e==cudaSuccess)return true;err=std::string("DFlash2: ")+what+": "+cudaGetErrorString(e);return false;}
}

struct DflashNative::Impl {
    struct Tensor {uint64_t off,bytes;std::vector<int> dims;};
    struct Layer {const uint16_t *ln1,*ln2,*q,*k,*v,*o,*qn,*kn,*gate,*up,*down,*acp,*acb,*mcp,*mcb;uint16_t *kc,*vc;};
    std::map<std::string,Tensor> tensors;
    std::vector<void*> allocations;
    std::array<Layer,5> layers{};
    uint16_t *weights=nullptr,*feat=nullptr,*ctx=nullptr,*normed=nullptr,*x=nullptr,*ln=nullptr,*cv=nullptr,*proj=nullptr;
    uint16_t *q=nullptr,*qr=nullptr,*k=nullptr,*kr=nullptr,*v=nullptr,*att=nullptr,*out=nullptr,*gate=nullptr,*up=nullptr,*dyn=nullptr;
    uint16_t *logits=nullptr,*selector=nullptr;
    float *tmp=nullptr,*embedf=nullptr,*partscores=nullptr,*best=nullptr;
    int32_t *partids=nullptr,*candidates=nullptr,*tokens=nullptr,*anchor_dev=nullptr;
    int64_t* position=nullptr;
    const NativeHead* head=nullptr;const NativeEmbed* embed=nullptr;
    uint64_t bytes=0;int max_context=0;int64_t received=0;cudaStream_t cs=nullptr;
    strata::prefill::Gemm gemm;
    cudaGraphExec_t graphs[8]={};bool graph=true,adaptive=false;
    Clock::time_point last_start;int last_k=0,rounds=0;
    std::array<std::vector<std::pair<int,double>>,8> samples;
    uint64_t calls=0,feature_rows=0;double elapsed_ms=0;
    ~Impl(){if(cs)cudaStreamSynchronize(cs);for(auto g:graphs)if(g)cudaGraphExecDestroy(g);
        for(auto a:allocations)cudaFree(a);if(cs)cudaStreamDestroy(cs);
        if(calls)std::fprintf(stderr,"strata dflash native: calls=%llu feature_rows=%llu draft_ms=%.3f graphs=%d\n",
            (unsigned long long)calls,(unsigned long long)feature_rows,elapsed_ms,graph?7:0);}
    template<class T> bool alloc(T*& ptr,size_t n,std::string& err){if(!check(cudaMalloc((void**)&ptr,n*sizeof(T)),"allocate",err))return false;allocations.push_back(ptr);return true;}
    const uint16_t* weight(const std::string& name,std::initializer_list<int> shape){
        auto i=tensors.find(name);if(i==tensors.end()||i->second.dims!=std::vector<int>(shape))throw std::runtime_error("DFlash2: missing/invalid tensor "+name);
        return weights+i->second.off/2;
    }
    void linear(const uint16_t* in,const uint16_t* w,uint16_t* dst,int rows,int n,int k){
        gemm.bf16(in,w,tmp,rows,n,k);cast16<<<(rows*n+255)/256,256,0,cs>>>(tmp,dst,rows*n);
    }
    void norm(const uint16_t* in,const uint16_t* w,uint16_t* dst,int rows,int cols){rms<<<rows,256,0,cs>>>(in,w,dst,cols);}
    void inject(Layer& l,const uint16_t* in,int rows,int offset){
        linear(in,l.k,k,rows,KV,H);linear(in,l.v,v,rows,KV,H);
        norm(k,l.kn,kr,rows*8,D);rope<<<(rows*KV+255)/256,256,0,cs>>>(kr,k,rows,KV,position,offset);
        write_kv<<<(rows*KV+255)/256,256,0,cs>>>(k,v,l.kc,l.vc,rows,position,offset);
    }
    void append_project(int rows){
        linear(feat,weight("fc.weight",{H,5*H}),ctx,rows,H,5*H);
        norm(ctx,weight("hidden_norm.weight",{H}),normed,rows,H);
        for(auto& l:layers)inject(l,normed,rows,0);
    }
    void forward(int count){
        const int rows=count+1;
        for(auto& l:layers){
            norm(x,l.ln1,ln,rows,H);
            linear(ln,l.acp,dyn,rows,640,H);
            conv<<<(rows*H+255)/256,256,0,cs>>>(ln,dyn,l.acb,cv,rows,0);
            linear(cv,l.q,q,rows,Q,H);norm(q,l.qn,qr,rows*32,D);
            rope<<<(rows*Q+255)/256,256,0,cs>>>(qr,q,rows,Q,position,0);
            inject(l,cv,rows,0);
            attention<<<rows*32,256,0,cs>>>(q,l.kc,l.vc,att,rows,position);
            linear(att,l.o,proj,rows,H,Q);
            conv<<<(rows*H+255)/256,256,0,cs>>>(proj,dyn,l.acb,out,rows,1);
            add<<<(rows*H+255)/256,256,0,cs>>>(x,out,x,rows*H);
            norm(x,l.ln2,ln,rows,H);
            linear(ln,l.mcp,dyn,rows,640,H);
            conv<<<(rows*H+255)/256,256,0,cs>>>(ln,dyn,l.mcb,cv,rows,0);
            linear(cv,l.gate,gate,rows,FF,H);linear(cv,l.up,up,rows,FF,H);
            swiglu<<<(rows*FF+255)/256,256,0,cs>>>(gate,up,gate,rows*FF);
            linear(gate,l.down,proj,rows,H,FF);
            conv<<<(rows*H+255)/256,256,0,cs>>>(proj,dyn,l.mcb,out,rows,1);
            add<<<(rows*H+255)/256,256,0,cs>>>(x,out,x,rows*H);
        }
        norm(x,weight("norm.weight",{H}),ln,rows,H);
        linear(ln+H,(const uint16_t*)head->weights(),logits,count,V,H);
        linear(ln+H,weight("candidate_selector.hidden_projection.weight",{RANK,H}),selector,count,RANK,H);
        int parts=(V+255)/256;
        top_parts<<<dim3(parts,count),256,0,cs>>>(logits,partscores,partids,parts);
        top_merge<<<count,256,0,cs>>>(partscores,partids,best,candidates,parts);
        select_path<<<1,256,0,cs>>>(selector,best,candidates,
            weight("candidate_selector.predecessor_codebook",{V,RANK}),
            weight("candidate_selector.successor_codebook",{V,RANK}),anchor_dev,tokens,count);
    }
    int choose(int cap){
        if(!adaptive)return cap;
        for(int k=1;k<=cap;++k)if(samples[k].size()<2)return k;
        if(rounds%32==0)return 1+(rounds/32)%cap;
        int bestk=1;double bestscore=0;
        for(int k=1;k<=cap;++k){double n=0,ms=0;for(auto s:samples[k]){n+=s.first;ms+=s.second;}
            if(ms>0&&n/ms>bestscore){bestscore=n/ms;bestk=k;}}
        return bestk;
    }
    void observe(int rows){if(!last_k)return;double ms=std::chrono::duration<double,std::milli>(Clock::now()-last_start).count();
        auto& s=samples[last_k];s.push_back({rows,ms});if(s.size()>12)s.erase(s.begin());last_k=0;++rounds;}
};

DflashNative::DflashNative():p_(new Impl){}
DflashNative::~DflashNative()=default;
uint64_t DflashNative::weight_bytes()const{return p_->bytes;}
bool DflashNative::load(const std::string& dir,const NativeHead& head,const NativeEmbed& embed,
                       int max_context,bool graph,bool adaptive,std::string& err){
    auto& p=*p_;
    if(!head.loaded()||head.type()!=30||embed.type()!=30||max_context<2||max_context>262144){
        err="DFlash2 native requires unchanged BF16 target head/embedding and context 2..262144";return false;}
    p.head=&head;p.embed=&embed;p.max_context=max_context;p.graph=graph;p.adaptive=adaptive;
    std::ifstream manifest(dir+"/manifest.txt");std::string line;
    if(!std::getline(manifest,line)||line!="STRATA_DFLASH2 1"||!std::getline(manifest,line)||
       line!="source d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"){
        err="DFlash2: invalid native manifest; run tools/dflash2_pack.py";return false;}
    uint64_t total=0;
    while(std::getline(manifest,line)){
        std::istringstream in(line);std::string tag,name;Impl::Tensor t{};int ndim=0;
        if(!(in>>tag>>name>>t.off>>t.bytes>>ndim)||tag!="tensor"||ndim<1||ndim>3||t.off!=total||t.bytes%2){err="DFlash2: invalid tensor record";return false;}
        uint64_t elems=1;for(int i=0;i<ndim;++i){int d=0;if(!(in>>d)||d<1||d>V||elems>1206131200u/d){err="DFlash2: invalid tensor shape";return false;}t.dims.push_back(d);elems*=d;}
        if(elems*2!=t.bytes||!p.tensors.emplace(name,t).second){err="DFlash2: duplicate/invalid tensor size";return false;}total+=t.bytes;
    }
    if(total!=1206131200){err="DFlash2: wrong parameter byte count";return false;}
    if(!check(cudaStreamCreateWithFlags(&p.cs,cudaStreamNonBlocking),"stream",err)||!p.alloc(p.weights,total/2,err))return false;
    std::ifstream file(dir+"/weights.bin",std::ios::binary|std::ios::ate);
    if(!file||file.tellg()!=std::streampos(total)){err="DFlash2: missing/truncated native weights";return false;}
    file.seekg(0);std::vector<char> chunk(8<<20);
    for(uint64_t off=0;off<total;){size_t n=size_t(std::min<uint64_t>(chunk.size(),total-off));
        if(!file.read(chunk.data(),n)||!check(cudaMemcpy((char*)p.weights+off,chunk.data(),n,cudaMemcpyHostToDevice),"weight upload",err))return false;off+=n;}
    p.bytes=total;
    try{
        p.weight("fc.weight",{H,H*5});p.weight("hidden_norm.weight",{H});p.weight("norm.weight",{H});
        p.weight("candidate_selector.hidden_projection.weight",{RANK,H});
        p.weight("candidate_selector.predecessor_codebook",{V,RANK});p.weight("candidate_selector.successor_codebook",{V,RANK});
        for(int i=0;i<5;++i){auto& l=p.layers[i];std::string b="layers."+std::to_string(i)+".";
            l.ln1=p.weight(b+"input_layernorm.weight",{H});l.ln2=p.weight(b+"post_attention_layernorm.weight",{H});
            l.q=p.weight(b+"self_attn.q_proj.weight",{Q,H});l.k=p.weight(b+"self_attn.k_proj.weight",{KV,H});
            l.v=p.weight(b+"self_attn.v_proj.weight",{KV,H});l.o=p.weight(b+"self_attn.o_proj.weight",{H,Q});
            l.qn=p.weight(b+"self_attn.q_norm.weight",{D});l.kn=p.weight(b+"self_attn.k_norm.weight",{D});
            l.gate=p.weight(b+"mlp.gate_proj.weight",{FF,H});l.up=p.weight(b+"mlp.up_proj.weight",{FF,H});l.down=p.weight(b+"mlp.down_proj.weight",{H,FF});
            l.acp=p.weight(b+"attention_conv.kernel_projection.weight",{640,H});l.acb=p.weight(b+"attention_conv.base_kernel",{2,2,H});
            l.mcp=p.weight(b+"mlp_conv.kernel_projection.weight",{640,H});l.mcb=p.weight(b+"mlp_conv.base_kernel",{2,2,H});
            if(!p.alloc(l.kc,size_t(RING)*KV,err)||!p.alloc(l.vc,size_t(RING)*KV,err))return false;
        }
    }catch(const std::exception& e){err=e.what();return false;}
    bool ok=p.alloc(p.feat,CHUNK*H*5,err)&&p.alloc(p.ctx,CHUNK*H,err)&&p.alloc(p.normed,CHUNK*H,err)&&
        p.alloc(p.x,8*H,err)&&p.alloc(p.ln,8*H,err)&&p.alloc(p.cv,8*H,err)&&p.alloc(p.proj,8*H,err)&&
        p.alloc(p.q,8*Q,err)&&p.alloc(p.qr,8*Q,err)&&p.alloc(p.k,CHUNK*KV,err)&&p.alloc(p.kr,CHUNK*KV,err)&&
        p.alloc(p.v,CHUNK*KV,err)&&p.alloc(p.att,8*Q,err)&&p.alloc(p.out,8*H,err)&&p.alloc(p.gate,8*FF,err)&&
        p.alloc(p.up,8*FF,err)&&p.alloc(p.dyn,8*640,err)&&p.alloc(p.logits,7*V,err)&&p.alloc(p.selector,7*RANK,err)&&
        p.alloc(p.tmp,std::max(CHUNK*H,7*V),err)&&p.alloc(p.embedf,8*H,err)&&
        p.alloc(p.partscores,7*((V+255)/256)*TOP,err)&&p.alloc(p.partids,7*((V+255)/256)*TOP,err)&&
        p.alloc(p.best,7*TOP,err)&&p.alloc(p.candidates,7*TOP,err)&&p.alloc(p.tokens,7,err)&&
        p.alloc(p.anchor_dev,1,err)&&p.alloc(p.position,1,err)&&p.gemm.init(p.cs,0,err);
    if(!ok)return false;
    int64_t zero=0;int32_t anchor=0;
    cudaMemcpy(p.position,&zero,8,cudaMemcpyHostToDevice);cudaMemcpy(p.anchor_dev,&anchor,4,cudaMemcpyHostToDevice);
    cudaMemsetAsync(p.x,0,8*H*2,p.cs);
    for(auto& l:p.layers){cudaMemsetAsync(l.kc,0,size_t(RING)*KV*2,p.cs);cudaMemsetAsync(l.vc,0,size_t(RING)*KV*2,p.cs);}
    if(graph)for(int count=1;count<=7;++count){
        p.forward(count);if(!check(cudaStreamSynchronize(p.cs),"graph warmup",err))return false;
        cudaGraph_t g=nullptr;
        if(!check(cudaStreamBeginCapture(p.cs,cudaStreamCaptureModeThreadLocal),"begin capture",err))return false;
        p.forward(count);
        if(!check(cudaStreamEndCapture(p.cs,&g),"end capture",err))return false;
        auto status=cudaGraphInstantiate(&p.graphs[count],g,nullptr,nullptr,0);cudaGraphDestroy(g);
        if(!check(status,"instantiate",err))return false;
    }
    std::fprintf(stderr,"strata dflash native: weights=%llu shared_head=%llu max_candidates=7 window=2048 context=%d graphs=%d adaptive=%d\n",
        (unsigned long long)p.bytes,(unsigned long long)head.weight_bytes(),max_context,graph?7:0,adaptive?1:0);
    return reset(err);
}
bool DflashNative::reset(std::string& err){auto& p=*p_;
    if(!check(cudaStreamSynchronize(p.cs),"reset",err))return false;
    p.received=0;p.last_k=0;p.rounds=0;for(auto& s:p.samples)s.clear();
    // Logical length masks all old rows; each visible row is overwritten before reading.
    return true;
}
bool DflashNative::append(const float* source,int cap,int rows,int64_t pos,std::string& err){auto& p=*p_;
    if(!source||rows<1||rows>cap||pos!=p.received||pos+rows>p.max_context){err="DFlash2: noncontiguous/out-of-range target features";return false;}
    p.observe(rows);
    for(int row0=0;row0<rows;row0+=CHUNK){int n=std::min(CHUNK,rows-row0);int64_t at=pos+row0;
        if(!check(cudaMemcpyAsync(p.position,&at,8,cudaMemcpyHostToDevice,p.cs),"position",err))return false;
        features16<<<(n*5*H+255)/256,256,0,p.cs>>>(source,p.feat,cap,row0,n);p.append_project(n);
        // The producer can reuse its feature buffers once append returns.
        if(!check(cudaStreamSynchronize(p.cs),"append features",err))return false;
    }p.received+=rows;p.feature_rows+=rows;return true;
}
bool DflashNative::append_host(const float* f,int rows,int64_t pos,std::string& err){auto& p=*p_;
    if(!f||rows<1||pos!=p.received||pos+rows>p.max_context){err="DFlash2 witness: invalid features";return false;}
    std::vector<uint16_t> host(CHUNK*5*H);
    for(int row0=0;row0<rows;row0+=CHUNK){int n=std::min(CHUNK,rows-row0);int64_t at=pos+row0;
        for(int i=0;i<n*5*H;++i)host[i]=bf16_from_f32(f[row0*5*H+i]);
        cudaMemcpyAsync(p.position,&at,8,cudaMemcpyHostToDevice,p.cs);cudaMemcpyAsync(p.feat,host.data(),n*5*H*2,cudaMemcpyHostToDevice,p.cs);
        p.append_project(n);if(!check(cudaStreamSynchronize(p.cs),"witness append",err))return false;
    }p.received+=rows;return true;
}
bool DflashNative::draft(int64_t pos,int anchor,int cap,int32_t* ids,int& got,std::string& err){auto& p=*p_;got=0;
    if(pos!=p.received||anchor<0||anchor>=V||cap<1||cap>7||pos+cap>=p.max_context){err="DFlash2: invalid draft request";return false;}
    int count=p.choose(cap);p.last_start=Clock::now();
    cudaMemcpyAsync(p.position,&pos,8,cudaMemcpyHostToDevice,p.cs);cudaMemcpyAsync(p.anchor_dev,&anchor,4,cudaMemcpyHostToDevice,p.cs);
    p.embed->gather_one(anchor,p.embedf,p.cs);
    for(int i=1;i<=count;++i)p.embed->gather_one(248077,p.embedf+i*H,p.cs);
    cast16<<<((count+1)*H+255)/256,256,0,p.cs>>>(p.embedf,p.x,(count+1)*H);
    if(p.graph)cudaGraphLaunch(p.graphs[count],p.cs);else p.forward(count);
    cudaMemcpyAsync(ids,p.tokens,count*4,cudaMemcpyDeviceToHost,p.cs);
    if(!check(cudaStreamSynchronize(p.cs),"draft",err))return false;
    for(int i=0;i<count;++i)if(ids[i]<0||ids[i]>=V){err="DFlash2: invalid proposal ID";return false;}
    got=count;p.last_k=count;++p.calls;p.elapsed_ms+=std::chrono::duration<double,std::milli>(Clock::now()-p.last_start).count();return true;
}
bool DflashNative::hidden(float* out,int rows,std::string& err)const{auto& p=*p_;
    if(rows<1||rows>7)return false;
    widen<<<(rows*H+255)/256,256,0,p.cs>>>(p.ln+H,p.tmp,rows*H);
    cudaMemcpyAsync(out,p.tmp,rows*H*4,cudaMemcpyDeviceToHost,p.cs);return check(cudaStreamSynchronize(p.cs),"hidden witness",err);
}
} // namespace strata::core
