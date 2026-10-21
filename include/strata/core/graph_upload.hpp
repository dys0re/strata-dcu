#pragma once
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstring>

namespace strata::core {
// Upload is an optional eager warm-up, not graph instantiation or execution.
// DTK 26.04's hipGraphUpload segfaults for a real gfx936 verify graph (core
// backtrace recorded in the DCU validation logs). Let the first launch upload
// on that device; retain the runtime call and error behavior everywhere else.
inline cudaError_t graph_upload(cudaGraphExec_t exec, cudaStream_t stream) {
#if defined(STRATA_HIP_GFX906)
    int device = -1;
    cudaDeviceProp p{};
    if (exec && cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&p, device) == cudaSuccess &&
        std::strncmp(p.gcnArchName, "gfx936", 6) == 0 &&
        (p.gcnArchName[6] == 0 || p.gcnArchName[6] == ':')) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr, "strata: gfx936: skip DTK hipGraphUpload warm-up; graphs instantiate and launch normally\n");
        return cudaSuccess;
    }
#endif
    return cudaGraphUpload(exec, stream);
}
} // namespace strata::core
