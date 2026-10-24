// Independent scalar decoding covers nibble order, signed scales, -128 Q8
// values, odd block counts, row slices, and token groups crossing SIMD slices.
#include "strata/kernels/cpu/q4_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace {
float half(const uint8_t* p) { ggml_fp16_t h; std::memcpy(&h,p,2); return ggml_fp16_to_fp32(h); }
void put_half(uint8_t* p,float x) { auto h=ggml_fp32_to_fp16(x);std::memcpy(p,&h,2); }
double scalar(const uint8_t* w,const uint8_t* a,int n) {
    double sum=0;
    for(int b=0;b<n/32;++b) {
        const uint8_t* x=w+18*b;const uint8_t* y=a+34*b;
        int dot=0;
        for(int j=0;j<32;++j) {
            const int q=((x[2+j%16]>>(j<16?0:4))&15)-8;
            int8_t v;std::memcpy(&v,y+2+j,1);dot+=q*int(v);
        }
        sum+=double(half(x))*double(half(y))*dot;
    }
    return sum;
}
}
int main() {
    ggml_cpu_init();
    const bool has512=cpu::cpu_avx512_ok();
    if(!has512)std::puts("SKIP direct AVX-512 checks; testing public dispatch fallback");
    std::mt19937 rng(1729);int failures=0,checks=0;
    double worst=0;
    if(has512)for(int n:{32,64,96,640,2560,6144,8192}) {
        constexpr int rows=11,maxt=17;
        const size_t stride=size_t(n/32)*18+2,up=stride*rows;
        std::vector<uint8_t> weights(2*up+4,0xa5);auto* w=weights.data()+2;
        for(int r=0;r<2*rows;++r)for(int b=0;b<n/32;++b) {
            auto* x=w+r*stride+b*18;
            put_half(x,b%7==0?0.f:float(int(rng()%33)-16)*0.003f);
            for(int j=0;j<16;++j)x[2+j]=uint8_t(rng());
        }
        std::vector<std::vector<uint8_t>> acts(maxt,std::vector<uint8_t>(n/32*34+2));
        std::vector<const void*> ap(maxt);
        std::vector<std::vector<float>> dst(maxt,std::vector<float>(rows,-12345.f));
        std::vector<float*> op(maxt);
        for(int t=0;t<maxt;++t) {
            ap[t]=acts[t].data()+2;op[t]=dst[t].data();
            for(int b=0;b<n/32;++b) {
                auto* y=acts[t].data()+2+34*b;put_half(y,float(int(rng()%31)-15)*0.002f);
                for(int j=0;j<32;++j)y[2+j]=uint8_t((t+b+j)%5==0?128:rng());
            }
        }
        for(int nt:{1,2,3,4,5,7,8,9,17})for(bool gu:{false,true})for(bool bias:{false,true}) {
            for(auto& v:dst)std::fill(v.begin(),v.end(),-12345.f);
            if(gu)cpu::q4_512_gu_rows(w,stride,up,n,ap.data(),nt,op.data(),1,rows-1,bias);
            else cpu::q4_512_rows(w,stride,n,ap.data(),nt,op.data(),1,rows-1,bias);
            for(int t=0;t<maxt;++t)for(int r=0;r<rows;++r) {
                ++checks;
                if(t>=nt||r==0||r==rows-1) { failures+=dst[t][r]!=-12345.f;continue; }
                double g=scalar(w+r*stride,static_cast<const uint8_t*>(ap[t]),n),u=0,ref=g;
                if(gu) { u=scalar(w+up+r*stride,static_cast<const uint8_t*>(ap[t]),n);ref=g/(1+std::exp(-g))*u; }
                const double err=std::abs(dst[t][r]-ref)/(1+std::abs(ref)+std::abs(g)+std::abs(u));
                worst=std::max(worst,err);
                if(!std::isfinite(dst[t][r])||err>3e-5)++failures;
            }
        }
    }
    // Exercise the public expert dispatch, quantized hidden rows and down matrix
    // against ggml's original per-token implementation on ordinary activations.
    cpu::NativeFmt fmt;std::string error;
    if(!cpu::native_fmt(GGML_TYPE_Q4_0,GGML_TYPE_Q4_0,2560,640,fmt,error)) {
        std::fprintf(stderr,"%s\n",error.c_str());return 1;
    }
    std::vector<uint8_t> blob(fmt.bytes);std::vector<float> row(2560);
    std::normal_distribution<float> normal(0.f,0.02f);
    const auto* wt=ggml_get_type_traits_cpu(GGML_TYPE_Q4_0);
    for(int r=0;r<1280;++r) {
        for(auto& x:row)x=normal(rng);
        wt->from_float(row.data(),blob.data()+r*fmt.gu_row,2560);
    }
    for(int r=0;r<2560;++r) {
        for(int j=0;j<640;++j)row[j]=normal(rng);
        wt->from_float(row.data(),blob.data()+fmt.down_off+r*fmt.d_row,640);
    }
    for(int nt:{1,2,3,4,5,9,17}) {
    std::vector<std::vector<uint8_t>> aq(nt,std::vector<uint8_t>(fmt.act_bytes)),hq(nt,std::vector<uint8_t>(fmt.h_bytes));
    std::vector<std::vector<float>> ff(nt,std::vector<float>(640)),out(nt,std::vector<float>(2560));
    std::vector<const void*> aa(nt),hh(nt);std::vector<float*> fp(nt),oo(nt);
    const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_Q4_0)->vec_dot;
    for(int t=0;t<nt;++t) {
        for(auto& x:row)x=normal(rng)*50;
        cpu::native_quant_act(fmt,row.data(),aq[t].data());
        aa[t]=aq[t].data();fp[t]=ff[t].data();oo[t]=out[t].data();
    }
    cpu::native_gu_rows(fmt,blob.data(),aa.data(),nt,fp.data(),0,640);
    for(int t=0;t<nt;++t){cpu::native_quant_h(fmt,ff[t].data(),hq[t].data());hh[t]=hq[t].data();}
    cpu::native_down_rows(fmt,blob.data(),hh.data(),nt,oo.data(),0,2560);
    double squared=0,ref_squared=0;
    for(int t=0;t<nt;++t) {
        std::vector<float> ref_ff(640);std::vector<uint8_t> ref_h(fmt.h_bytes);
        for(int r=0;r<640;++r) {
            float g=0,u=0;dot(2560,&g,0,blob.data()+r*fmt.gu_row,0,aa[t],0,1);
            dot(2560,&u,0,blob.data()+fmt.up_off+r*fmt.gu_row,0,aa[t],0,1);
            ref_ff[r]=(g/(1.f+std::exp(-g)))*u;
        }
        cpu::native_quant_h(fmt,ref_ff.data(),ref_h.data());
        for(int r=0;r<2560;++r) {
            float ref=0;dot(640,&ref,0,blob.data()+fmt.down_off+r*fmt.d_row,0,ref_h.data(),0,1);
            const double d=out[t][r]-ref;squared+=d*d;ref_squared+=double(ref)*ref;
        }
    }
    const double relative=std::sqrt(squared/(ref_squared+1e-30));
    failures+=!std::isfinite(relative)||relative>3e-4;
    std::printf("Q4 whole expert: nt=%d relative_L2=%.3g\n",nt,relative);
    }
    std::printf("Q4 CPU: checks=%d worst_scaled_error=%.3g failures=%d\n",checks,worst,failures);
    return failures?1:0;
}
