// Standalone diagnostic: compare BF16/Q8_1 row layouts without changing weights.
// The 32-lane variants retain the production kernel's FMA and reduction order.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Q81 { half2 ds; signed char q[32]; };
static_assert(sizeof(Q81)==36);
void ck(hipError_t e) { if(e!=hipSuccess) { std::fprintf(stderr,"%s\n",hipGetErrorString(e)); std::exit(1); } }
__global__ void init(unsigned short* w, size_t size, Q81* x, int k) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<size; i+=size_t(gridDim.x)*blockDim.x)
        w[i]=static_cast<unsigned short>(0x3b00u+((i*131u+i/67u)%1536u));
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<k/32;i+=gridDim.x*blockDim.x) {
        x[i].ds=__floats2half2_rn(.0078125f,0);
        for(int j=0;j<32;++j) x[i].q[j]=static_cast<signed char>((i*17+j*13)%255-127);
    }
}
template<int Rows,int Lanes,bool Cache>
__global__ __launch_bounds__(Rows*Lanes) void gemv(const unsigned short* __restrict__ w,
        const Q81* __restrict__ x, float* __restrict__ y,int k,int n) {
    extern __shared__ float cached[];
    if constexpr(Cache) {
        for(int i=threadIdx.x;i<k;i+=blockDim.x)
            cached[i]=__half2float(__low2half(x[i/32].ds))*float(x[i/32].q[i%32]);
        __syncthreads();
    }
    int lane=threadIdx.x%Lanes, row=blockIdx.x*Rows+threadIdx.x/Lanes;
    if(row>=n) return;
    float sum=0;
    for(int i=lane;i<k;i+=Lanes) {
        float weight=__uint_as_float(unsigned(w[size_t(row)*k+i])<<16);
        float value;
        if constexpr(Cache) value=cached[i];
        else value=__half2float(__low2half(x[i/32].ds))*float(x[i/32].q[i%32]);
        sum=__fmaf_rn(weight,value,sum);
    }
    for(int off=Lanes/2;off;off/=2) sum+=__shfl_down(sum,off,Lanes);
    if(!lane) y[row]=sum;
}
template<int R,int L,bool C>
void launch(const unsigned short* w,const Q81* x,float* y,int k,int n,hipStream_t s) {
    gemv<R,L,C><<<(n+R-1)/R,R*L,C?k*4:0,s>>>(w,x,y,k,n);
}
template<int Lanes>
__global__ __launch_bounds__(256) void vector4(const unsigned short* __restrict__ w,
        const Q81* __restrict__ x,float* __restrict__ y,int k,int n) {
    int lane=threadIdx.x%Lanes,row=blockIdx.x*(256/Lanes)+threadIdx.x/Lanes;
    if(row>=n)return;
    float a[4]={};
    for(int i=lane*4;i<k;i+=Lanes*4) {
        unsigned long long bits=*reinterpret_cast<const unsigned long long*>(w+size_t(row)*k+i);
        unsigned q=*reinterpret_cast<const unsigned*>(x[i/32].q+i%32);
        float scale=__half2float(__low2half(x[i/32].ds));
        #pragma unroll
        for(int j=0;j<4;++j) a[j]=__fmaf_rn(__uint_as_float(unsigned((bits>>(j*16))&65535u)<<16),
                scale*float(static_cast<signed char>(q>>(j*8))),a[j]);
    }
    float sum=(a[0]+a[1])+(a[2]+a[3]);
    for(int off=Lanes/2;off;off/=2)sum+=__shfl_down(sum,off,Lanes);
    if(!lane)y[row]=sum;
}
template<int L>void launch4(const unsigned short* w,const Q81* x,float* y,int k,int n,hipStream_t s) {
    vector4<L><<<(n+256/L-1)/(256/L),256,0,s>>>(w,x,y,k,n);
}
using Fn=void(*)(const unsigned short*,const Q81*,float*,int,int,hipStream_t);
int main() {
    Fn fn[]={launch<4,32,false>,launch<8,32,false>,launch<4,32,true>,launch<8,32,true>,
             launch<4,64,false>,launch<8,16,false>,launch<16,16,true>,launch<16,32,true>,launch4<32>,launch4<64>};
    const char* names[]={"baseline","rows8","cache4","cache8","wave64","lanes16","cache16x16","cache16x32","vector4x32","vector4x64"};
    hipStream_t s; ck(hipStreamCreate(&s));
    std::puts("K,N,variant,median_us,max_abs_error,bit_different");
    for(auto shape:std::vector<std::pair<int,int>>{{2560,6144},{2560,10240},{6144,2560},{2560,640},{640,2560},{2560,512},{2560,248320}}) {
        int k=shape.first,n=shape.second;
        unsigned short* w;Q81* x;float* y;
        ck(hipMalloc(&w,size_t(k)*n*2));ck(hipMalloc(&x,k/32*36));ck(hipMalloc(&y,n*4));
        init<<<1024,256,0,s>>>(w,size_t(k)*n,x,k);ck(hipStreamSynchronize(s));
        fn[0](w,x,y,k,n,s);ck(hipStreamSynchronize(s));
        std::vector<float> ref(n),got(n);ck(hipMemcpy(ref.data(),y,n*4,hipMemcpyDeviceToHost));
        hipEvent_t a,b;ck(hipEventCreate(&a));ck(hipEventCreate(&b));
        for(int v=0;v<10;++v) {
            std::vector<float> times;
            for(int rep=0;rep<5;++rep) {
                for(int t=0;t<5;++t) fn[v](w,x,y,k,n,s);
                ck(hipEventRecord(a,s));
                for(int t=0;t<30;++t) fn[v](w,x,y,k,n,s);
                ck(hipEventRecord(b,s));ck(hipEventSynchronize(b));
                float ms;ck(hipEventElapsedTime(&ms,a,b));times.push_back(ms*1000/30);
            }
            ck(hipGetLastError());ck(hipMemcpy(got.data(),y,n*4,hipMemcpyDeviceToHost));
            float maxerr=0;int bits=0;
            for(int i=0;i<n;++i) { maxerr=std::max(maxerr,std::fabs(ref[i]-got[i]));bits+=ref[i]!=got[i]; }
            if(v!=4&&v!=5&&v!=6&&v<8&&bits) { std::fprintf(stderr,"32-lane variant changed sums\n");return 2; }
            std::sort(times.begin(),times.end());
            std::printf("%d,%d,%s,%.3f,%.9g,%d\n",k,n,names[v],times[2],maxerr,bits);std::fflush(stdout);
        }
        ck(hipEventDestroy(a));ck(hipEventDestroy(b));ck(hipFree(y));ck(hipFree(x));ck(hipFree(w));
    }
    ck(hipStreamDestroy(s));
}
