// Standalone synthetic same-layout Q4 experiment, including -128 activations.
// No engine or public wrapper changes. Timings include per-call input preparation.
#include "strata/kernels/cpu/q4_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace strata::kernels::cpu {
void angle_gu_rows(const uint8_t*,size_t,size_t,int,const void* const*,int,float* const*,int,int);
void angle_rows(const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int);
}
namespace cpu=strata::kernels::cpu;
int main(int argc,char** argv) {
    const bool stream=argc>1&&std::strcmp(argv[1],"--stream")==0;
    ggml_cpu_init();
    if(!cpu::cpu_avx512_ok())return 77;
    std::mt19937 rng(1729);std::normal_distribution<float> normal(0.f,.03f);
    const auto* q4=ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    const auto* q8=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    int failures=0;
    for(int nt:{1,2,3,4})for(int stage:{0,1})for(int tile:{0,1}) {
        if(stream&&tile)continue;
        const int n=stage?640:2560, nr=stage?2560:640;
        const int rows=tile?(stage?128:32):nr;
        const size_t stride=size_t(n/32)*18,up=stride*nr;
        const size_t bank_bytes=(stage?1:2)*up;
        const int banks=stream?64:1;int bank=0;
        std::vector<uint8_t> weights(bank_bytes*banks);
        std::vector<float> f(n);
        for(int r=0;r<(stage?nr:2*nr);++r) {
            for(auto& x:f)x=normal(rng);
            q4->from_float(f.data(),weights.data()+r*stride,n);
        }
        for(int b=1;b<banks;++b)std::memcpy(weights.data()+bank_bytes*b,weights.data(),bank_bytes);
        std::vector<std::vector<uint8_t>> aq(nt,std::vector<uint8_t>(size_t(n/32)*34));
        std::vector<const void*> ap(nt);std::vector<float*> op(nt);
        std::vector<std::vector<float>> dst(nt,std::vector<float>(rows));
        for(int t=0;t<nt;++t) {
            for(auto& x:f)x=normal(rng)*20;
            q8->from_float(f.data(),aq[t].data(),n);ap[t]=aq[t].data();op[t]=dst[t].data();
        }
        auto run=[&](int method) {
            const auto* w=weights.data()+bank_bytes*bank;
            if(method==0) {
                for(int r=0;r<rows;++r)for(int t=0;t<nt;++t) {
                    float g=0,u=0;q4->vec_dot(n,&g,0,w+r*stride,0,ap[t],0,1);
                    if(!stage)q4->vec_dot(n,&u,0,w+up+r*stride,0,ap[t],0,1);
                    dst[t][r]=stage?g:(g/(1+std::exp(-g)))*u;
                }
            } else if(method==1) {
                if(stage)cpu::q4_512_rows(w,stride,n,ap.data(),nt,op.data(),0,rows);
                else cpu::q4_512_gu_rows(w,stride,up,n,ap.data(),nt,op.data(),0,rows);
            } else {
                if(stage)cpu::angle_rows(w,stride,n,ap.data(),nt,op.data(),0,rows);
                else cpu::angle_gu_rows(w,stride,up,n,ap.data(),nt,op.data(),0,rows);
            }
        };
        // Existing fused kernel is an independent implementation; ggml is also checked.
        run(0);const auto ref=dst;
        for(bool edge:{false,true}) {
            if(edge)for(auto& a:aq)for(size_t i=2;i<a.size();i+=34)a[i]=128;
            run(1);const auto existing=dst;run(2);
            double diff=0,norm=0,ggdiff=0,ggnorm=0;
            for(int t=0;t<nt;++t)for(int r=0;r<rows;++r) {
                diff+=std::pow(double(dst[t][r])-existing[t][r],2);norm+=double(existing[t][r])*existing[t][r];
                if(!edge){ggdiff+=std::pow(double(dst[t][r])-ref[t][r],2);ggnorm+=double(ref[t][r])*ref[t][r];}
                failures+=!std::isfinite(dst[t][r]);
            }
            failures+=std::sqrt(diff/(norm+1e-30))>1e-6;
            if(!edge)failures+=std::sqrt(ggdiff/(ggnorm+1e-30))>3e-5;
            std::printf("{\"kind\":\"numeric\",\"nt\":%d,\"stage\":%d,\"tile\":%d,\"edge\":%d,\"relative\":%.9g,\"ggml_relative\":%.9g}\n",nt,stage,tile,int(edge),std::sqrt(diff/(norm+1e-30)),std::sqrt(ggdiff/(ggnorm+1e-30)));
        }
        // Restore ordinary inputs for timing; each method sees the same weights and input.
        for(int t=0;t<nt;++t){for(auto& x:f)x=normal(rng)*20;q8->from_float(f.data(),aq[t].data(),n);}
        std::vector<double> samples[3];
        for(int round=0;round<9;++round)for(int j=0;j<3;++j) {
            const int method=(j+round)%3;run(method);
            const int reps=stream?128:(tile?120:20);
            const auto start=std::chrono::steady_clock::now();
            for(int rep=0;rep<reps;++rep){bank=rep%banks;run(method);}
            const auto end=std::chrono::steady_clock::now();
            const double us=std::chrono::duration<double,std::micro>(end-start).count()/reps;
            samples[method].push_back(us);
            std::printf("{\"kind\":\"timing\",\"nt\":%d,\"stage\":%d,\"tile\":%d,\"method\":%d,\"round\":%d,\"us\":%.9g}\n",nt,stage,tile,method,round,us);
        }
        for(int method=0;method<3;++method) {
            auto v=samples[method];std::sort(v.begin(),v.end());
            std::printf("{\"kind\":\"median\",\"nt\":%d,\"stage\":%d,\"tile\":%d,\"method\":%d,\"us\":%.9g}\n",nt,stage,tile,method,v[4]);
        }
    }
    std::printf("{\"kind\":\"complete\",\"failures\":%d}\n",failures);
    return failures?1:0;
}
