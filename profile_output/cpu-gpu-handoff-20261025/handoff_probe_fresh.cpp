// Supplement: CPU changes every payload, one graph followed by completion.
// Wall latency includes graph launch + synchronization. This is not decode TPS.
#define main warm_probe_main
#include "handoff_probe.cpp"
#undef main
int main(){
    Probe p;int checks=0;
    for(int t:{1,2,3,4,5})for(int pct:{0,30,60,90,100}){
        const int nh=t*10*pct/100;p.setup(t,nh,25+t+pct);
        cudaGraphExec_t graphs[3]={p.graph(0,t),p.graph(1,t),p.graph(2,t)};
        for(int round=0;round<7;round++)for(int x=0;x<3;x++){
            const int method=(x+round)%3;
            std::vector<double> samples;
            for(int i=0;i<30;i++){
                CK(cudaStreamSynchronize(p.stream));
                __atomic_store_n(p.flag,0u,__ATOMIC_RELEASE);
                // Change all active rows, then publish; do not repeatedly time
                // the same unmodified mapped host payload.
                for(int r=0;r<t*10;r++)p.hv[r*p.N+(i*37+round*13)%p.N]+=.0078125f;
                std::memcpy(p.h,p.hv.data(),p.hv.size()*4);_mm_sfence();
                __atomic_store_n(p.flag,1u,__ATOMIC_RELEASE);
                const auto start=std::chrono::steady_clock::now();
                CK(cudaGraphLaunch(graphs[method],p.stream));
                CK(cudaStreamSynchronize(p.stream));
                const auto end=std::chrono::steady_clock::now();
                samples.push_back(std::chrono::duration<double,std::micro>(end-start).count());
            }
            p.check(p.read(t),t,round,method);checks++;
            for(int i=0;i<(int)samples.size();i++)
                std::printf("{\"kind\":\"fresh\",\"t\":%d,\"hits\":%d,\"rows\":%d,\"round\":%d,\"method\":%d,\"sample\":%d,\"us\":%.8f}\n",t,nh,t*10,round,method,i,samples[i]);
            std::fflush(stdout);
        }
        for(auto g:graphs)CK(cudaGraphExecDestroy(g));
    }
    std::printf("{\"kind\":\"complete\",\"numeric_cases\":%d,\"failures\":0}\n",checks);
    return 0;
}
