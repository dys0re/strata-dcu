#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
static void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
static void require(bool b,const char* s) { if(!b)throw std::runtime_error(s); }
template<class T> T* alloc(size_t n) { T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));return p; }
int main() {
 try {
  constexpr int cap=80,width=2560,po=((4+cap+1+2*cap)+1)&~1,words=po+5*cap+1;
  cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  auto *logits=alloc<float>(8*512),*x=alloc<float>(8*width),*weights=alloc<float>(80);
  auto *ids=alloc<int>(80),*res=alloc<int>(512),*plan=alloc<int>(words);
  auto* offsets=alloc<unsigned long long>(512);auto* cache=alloc<unsigned char>(512*64);
  float *hx=nullptr,*mx=nullptr;int *hi=nullptr,*mi=nullptr;unsigned *hs=nullptr,*ms=nullptr;
  ck(cudaHostAlloc(&hx,8*width*4,cudaHostAllocMapped));ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mx),hx,0));
  ck(cudaHostAlloc(&hi,80*4,cudaHostAllocMapped));ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mi),hi,0));
  ck(cudaHostAlloc(&hs,64,cudaHostAllocMapped));ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ms),hs,0));
  std::mt19937 rng(1729);std::uniform_real_distribution<float> random(-3,3);
  int cases=0;
  for(int nt=1;nt<=8;++nt)for(int hits=0;hits<=4;++hits) {
   std::vector<float> l(nt*512),input(nt*width),actual(nt*10);
   for(auto&v:l)v=hits==4?0:random(rng);
   std::vector<int> r(512),expected_ids(nt*10),actual_ids(nt*10),p(words);
   std::vector<unsigned long long> off(512);
   for(int e=0;e<512;++e){r[e]=e%4<hits?e:-1;off[e]=e*64;}
   for(int t=0;t<nt;++t){
    std::vector<int> order(512);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return l[t*512+a]>l[t*512+b];});
    for(int k=0;k<10;++k)expected_ids[t*10+k]=order[k];
   }
   ck(cudaMemcpy(logits,l.data(),l.size()*4,cudaMemcpyHostToDevice));
   ck(cudaMemcpy(res,r.data(),r.size()*4,cudaMemcpyHostToDevice));
   ck(cudaMemcpy(offsets,off.data(),off.size()*8,cudaMemcpyHostToDevice));
   strata::kernels::RouterPublishArgs a;
   a.logits=logits;a.x=x;a.ids=ids;a.weights=weights;a.resident=res;a.cache=cache;
   a.slot_offsets=offsets;a.blob_bytes=64;a.cap=cap;a.plan=plan;
   a.host_x=mx;a.host_ids=mi;a.host_seq=ms;a.n_tok=nt;a.n_embd=width;a.ring=cases+1;
   cudaGraph_t graph;cudaGraphExec_t exec;
   ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
   strata::kernels::native_router_top10_publish(a,stream);
   ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
   for(int replay=0;replay<2;++replay){
    for(size_t i=0;i<input.size();++i)input[i]=float(i)*.001f+replay;
    std::fill(hx,hx+input.size(),-9876.f);std::fill(hi,hi+nt*10,-999);
    std::atomic_ref<unsigned>(*hs).store(0,std::memory_order_release);
    ck(cudaMemset(plan,0xcc,words*4));ck(cudaMemcpyAsync(x,input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));
    ck(cudaGraphLaunch(exec,stream));
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(std::atomic_ref<unsigned>(*hs).load(std::memory_order_acquire)!=a.ring)
     require(std::chrono::steady_clock::now()<deadline,"payload publication timed out");
    // Read immediately on the CPU signal, BEFORE stream synchronization or plan completion.
    require(std::equal(hi,hi+nt*10,expected_ids.begin()),"published IDs differ from CPU top-k");
    bool miss=false;for(int e:expected_ids)miss|=r[e]<0;
    if(miss)require(std::equal(hx,hx+input.size(),input.begin()),"CPU observed stale payload after signal");
    else require(std::all_of(hx,hx+input.size(),[](float v){return v==-9876.f;}),"all-hit payload should be skipped");
    ck(cudaStreamSynchronize(stream));
    ck(cudaMemcpy(p.data(),plan,words*4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(actual.data(),weights,actual.size()*4,cudaMemcpyDeviceToHost));
    int groups=0,entries=0;
    for(int i=0;i<nt*10;++i){
     int e=expected_ids[i];if(r[e]<0)continue;
     if(std::find(expected_ids.begin(),expected_ids.begin()+i,e)!=expected_ids.begin()+i)continue;
     require(p[4+groups]==entries,"resident group start differs");
     unsigned long long ptr;std::memcpy(&ptr,&p[po+groups*2],8);
     require(ptr==reinterpret_cast<unsigned long long>(cache)+off[e],"resident pointer differs");
     for(int j=i;j<nt*10;++j)if(expected_ids[j]==e){
      require(p[4+cap+1+entries]==j,"resident destination differs");
      require(p[4+cap+1+cap+entries]==j/10,"resident token differs");++entries;
     }++groups;
    }
    require(p[0]==groups&&p[1]==entries&&p[2]==0&&p[4+groups]==entries&&p[po+4*cap]==entries,"resident plan counts differ");
    for(int t=0;t<nt;++t){
     double sum=0,ref[10];for(int k=0;k<10;++k){ref[k]=std::exp(double(l[t*512+expected_ids[t*10+k]])-l[t*512+expected_ids[t*10]]);sum+=ref[k];}
     for(int k=0;k<10;++k)require(std::abs(actual[t*10+k]-ref[k]/sum)<1e-5,"routing weights differ from CPU reference");
    }++cases;
   }
   ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
  }
  ck(cudaStreamDestroy(stream));
  std::printf("ROUTE_PUBLISH_PASS cases=%d widths=1..8 immediate_payload=PASS resident_plan=PASS weights=PASS\n",cases);
  return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"ROUTE_PUBLISH_FAIL %s\n",e.what());return 1;}
}
