// Diagnostic sequential-read ceiling. 2 GiB exceeds LLC; 12 pinned physical cores.
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
int main(){
 constexpr size_t n=size_t(2)*1024*1024*1024/sizeof(float);
 float* p=nullptr;if(posix_memalign((void**)&p,4096,n*sizeof(float)))return 1;
 #pragma omp parallel for schedule(static)
 for(size_t i=0;i<n;++i)p[i]=1.0f;
 for(int rep=0;rep<7;++rep){
  double sum=0;auto b=std::chrono::steady_clock::now();
  #pragma omp parallel reduction(+:sum)
  {
   size_t lo=(n*omp_get_thread_num()/omp_get_num_threads())/128*128;
   size_t hi=(n*(omp_get_thread_num()+1)/omp_get_num_threads())/128*128;
   __m512 a[8];for(auto& v:a)v=_mm512_setzero_ps();
   for(size_t i=lo;i<hi;i+=128)for(int j=0;j<8;++j)a[j]=_mm512_add_ps(a[j],_mm512_load_ps(p+i+16*j));
   for(auto v:a)sum+=_mm512_reduce_add_ps(v);
  }
  double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-b).count();
  std::printf("%d,%.6f,%.3f,%.0f\n",rep,sec,n*sizeof(float)/sec/1e9,sum);
 }
 free(p);
}
