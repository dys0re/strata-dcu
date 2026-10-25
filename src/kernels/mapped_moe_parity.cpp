// Actual production mapped-combine entry: independent ordered-FMA reference,
// changing payloads, delayed publication and captured graph replay.
#include "strata/kernels/native_moe.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* upload(const std::vector<T>& data) {
    T* p = nullptr; ck(cudaMalloc((void**) &p, std::max<size_t>(data.size(), 1) * sizeof(T)));
    if (!data.empty()) ck(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
int main() {
    using namespace strata::kernels;
    std::mt19937 rng(20261025);
    std::uniform_real_distribution<float> random(-2.f, 2.f);
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    uint32_t *flag, *mapped_flag;
    ck(cudaHostAlloc((void**) &flag, 64, cudaHostAllocMapped));
    ck(cudaHostGetDevicePointer((void**) &mapped_flag, flag, 0));
    int cases = 0, failures = 0;
    // Include partial last vector blocks and model-sized output widths.
    for (int width : {4, 516, 2560, 8192}) for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8}) {
        const int rows = tokens * 10;
        std::vector<float> cpu(rows * width), hits(cpu.size()), weights(rows), shared(tokens * width);
        for (float& v : cpu) v = random(rng);
        for (float& v : hits) v = random(rng);
        for (float& v : weights) v = random(rng) * .1f;
        for (float& v : shared) v = random(rng);
        // Include signed zeros, exact cancellation and negative router weights.
        for (int r = 0; r < rows; ++r) { cpu[r * width] = -0.f; hits[r * width] = -0.f; }
        float *host, *mapped;
        ck(cudaHostAlloc((void**) &host, cpu.size() * 4, cudaHostAllocMapped));
        ck(cudaHostGetDevicePointer((void**) &mapped, host, 0));
        float* dh = upload(hits); float* dw = upload(weights); float* ds = upload(shared);
        float* output; ck(cudaMalloc((void**) &output, shared.size() * 4));
        if (!native_moe_combine_mapped_supported(mapped, dh, ds, output, width, 10, tokens) ||
            native_moe_combine_mapped_supported(mapped + 1, dh, ds, output, width, 10, tokens) ||
            native_moe_combine_mapped_supported(mapped, dh, ds, output, width, 9, tokens) ||
            native_moe_combine_mapped_supported(mapped, dh, ds, output, width, 10, 9) ||
            native_moe_combine_mapped_supported(nullptr, dh, ds, output, width, 10, tokens)) return 3;
        for (int percent : {0, 30, 60, 90, 100}) {
            std::vector<int32_t> gpu(rows); std::iota(gpu.begin(), gpu.end(), 0);
            std::shuffle(gpu.begin(), gpu.end(), rng); gpu.resize(rows * percent / 100);
            const int32_t nh = (int32_t) gpu.size();
            int32_t* dg = upload(gpu); int32_t* dc = upload(std::vector<int32_t>{nh});
            std::vector<bool> is_gpu(rows, false); for (int r : gpu) is_gpu[r] = true;
            cudaGraph_t graph; cudaGraphExec_t exec;
            *flag = 0;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            native_moe_combine_mapped_after(mapped, dh, dg, dc, dw, ds, output, mapped_flag, 7,
                                            width, 10, tokens, stream);
            ck(cudaStreamEndCapture(stream, &graph));
            ck(hipGraphInstantiateWithFlags(&exec, graph, 0));
            for (int round = 0; round < 6; ++round) {
                std::atomic_ref<uint32_t>(*flag).store(0u, std::memory_order_release);
                for (int r = 0; r < rows; ++r) cpu[r * width + (round + 1) % width] += .03125f;
                std::fill(host, host + cpu.size(), 123456.f); // stale payload must not be consumed
                ck(cudaGraphLaunch(exec, stream));
                if (round & 1) std::this_thread::sleep_for(std::chrono::microseconds(50 + round * 40));
                // No query or synchronization before the producer publishes.
                std::memcpy(host, cpu.data(), cpu.size() * 4); _mm_sfence();
                std::atomic_ref<uint32_t>(*flag).store(7u, std::memory_order_release);
                ck(cudaStreamSynchronize(stream));
                std::vector<float> got(shared.size());
                ck(cudaMemcpy(got.data(), output, got.size() * 4, cudaMemcpyDeviceToHost));
                for (int t = 0; t < tokens; ++t) for (int col = 0; col < width; ++col) {
                    float sum = 0;
                    for (int j = 0; j < 10; ++j) {
                        const int r = t * 10 + j, at = r * width + col;
                        const float value = is_gpu[r] ? 0.f + hits[at] : cpu[at];
                        if (j == 0) sum = value * weights[r]; else sum = std::fma(value, weights[r], sum);
                    }
                    const float ref = sum + shared[t * width + col];
                    if (std::memcmp(&ref, &got[t * width + col], 4)) ++failures;
                }
                ++cases;
            }
            ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaFree(dg)); ck(cudaFree(dc));
        }
        ck(cudaFreeHost(host)); for (void* p : {(void*) dh, (void*) dw, (void*) ds, (void*) output}) ck(cudaFree(p));
    }
    ck(cudaFreeHost(flag)); ck(cudaStreamDestroy(stream));
    std::printf("MAPPED_MOE_PARITY cases=%d failures=%d\n", cases, failures);
    return failures ? 1 : 0;
}
