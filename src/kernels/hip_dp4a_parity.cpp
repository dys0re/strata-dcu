// Exercise the production compatibility intrinsic, including signed bytes and
// accumulator wraparound. The host oracle uses unsigned arithmetic explicitly.
#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void dots(const int* a, const int* b, const int* c, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __dp4a(a[i], b[i], c[i]);
}

uint32_t reference(uint32_t a, uint32_t b, uint32_t c) {
    for (int lane = 0; lane < 4; ++lane) {
        int x = (a >> (8 * lane)) & 255u;
        int y = (b >> (8 * lane)) & 255u;
        if (x >= 128) x -= 256;
        if (y >= 128) y -= 256;
        c += static_cast<uint32_t>(x * y);
    }
    return c;
}
}  // namespace

int main() {
    static_assert(sizeof(int) == sizeof(uint32_t));
    constexpr int pairs = 4 * 256 * 256, random = 1 << 20, n = pairs + random;
    constexpr uint32_t accumulators[] = {0u, 0x7fffffffu, 0x80000000u, 0xffffffffu,
                                         1u, 0x7fff8000u, 0x80008000u, 0xffff0000u};
    std::mt19937 rng(936);
    std::vector<int> a(n), b(n), c(n), got(n);
    std::vector<uint32_t> ref(n);
    for (int i = 0; i < n; ++i) {
        uint32_t x = rng(), y = rng();
        if (i < pairs) {
            const int lane = i / 65536, shift = lane * 8;
            const uint32_t mask = 255u << shift;
            x = (x & ~mask) | (uint32_t((i / 256) & 255) << shift);
            y = (y & ~mask) | (uint32_t(i & 255) << shift);
        }
        const uint32_t z = i < pairs ? accumulators[i % 8] : uint32_t(rng());
        a[i] = std::bit_cast<int>(x); b[i] = std::bit_cast<int>(y); c[i] = std::bit_cast<int>(z);
        ref[i] = reference(x, y, z);
    }
    int *da = nullptr, *db = nullptr, *dc = nullptr, *dy = nullptr;
    const size_t bytes = size_t(n) * sizeof(int);
    check(cudaMalloc(&da, bytes)); check(cudaMalloc(&db, bytes));
    check(cudaMalloc(&dc, bytes)); check(cudaMalloc(&dy, bytes));
    check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc, c.data(), bytes, cudaMemcpyHostToDevice));
    dots<<<(n + 255) / 256, 256>>>(da, db, dc, dy, n);
    check(cudaGetLastError());
    check(cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost));
    size_t failures = 0;
    for (int i = 0; i < n; ++i) {
        if (std::bit_cast<uint32_t>(got[i]) != ref[i]) {
            if (failures < 8) std::fprintf(stderr, "case %d: got %08x expected %08x\n", i,
                                           std::bit_cast<uint32_t>(got[i]), ref[i]);
            ++failures;
        }
    }
    check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dc)); check(cudaFree(dy));
    std::printf("hip_dp4a_parity: %d cases, %zu failures (signed byte pairs, random words, accumulator wraparound)\n",
                n, failures);
    return failures ? 1 : 0;
}
