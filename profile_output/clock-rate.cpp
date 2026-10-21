#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    int khz=0;
    auto e=hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0);
    printf("Wall clock query: status=%s rate=%d kHz ns_per_tick=%.6f\n",hipGetErrorString(e),khz,khz?1e6/khz:0);
    return e==hipSuccess?0:1;
}
