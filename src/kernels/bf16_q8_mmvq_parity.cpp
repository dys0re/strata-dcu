// BF16 native projection against a double-precision host dot of the exact Q8_1
// activation bytes. Includes row tails and several independent token columns.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
int main() {
    using namespace strata::kernels;
    try {
        cudaStream_t stream;
        check(cudaStreamCreate(&stream));
        for (int k : {64, 2560}) for (int n : {5, 33}) for (int t : {1, 3, 8}) {
            std::vector<uint16_t> w(size_t(k) * n);
            std::vector<float> x(size_t(k) * t), got(size_t(n) * t);
            for (size_t i = 0; i < w.size(); ++i) w[i] = bf16_from_f32(std::sin(float(i) * .13f) * .1f);
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * .17f);
            void *dw = nullptr, *dx = nullptr, *dq = nullptr, *dy = nullptr;
            const size_t qb = native_q8_1_bytes(k, t);
            check(cudaMalloc(&dw, w.size() * 2)); check(cudaMalloc(&dx, x.size() * 4));
            check(cudaMalloc(&dq, qb)); check(cudaMalloc(&dy, got.size() * 4));
            check(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            bool rejected = false;
            try { native_mmvq(30, dw, dq, static_cast<float*>(dy), k, n, t, nullptr); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("null stream was not rejected");
            native_quantize_q8_1(static_cast<float*>(dx), dq, k, t, stream);
            native_mmvq(30, dw, dq, static_cast<float*>(dy), k, n, t, stream);
            check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<uint8_t> q(qb);
            check(cudaMemcpy(q.data(), dq, qb, cudaMemcpyDeviceToHost));
            for (int col = 0; col < t; ++col) for (int row = 0; row < n; ++row) {
                double expected = 0, magnitude = 0;
                for (int j = 0; j < k; ++j) {
                    const uint8_t* b = q.data() + (size_t(col) * (k / 32) + j / 32) * 36;
                    uint16_t scale; std::memcpy(&scale, b, 2);
                    const double v = double(f32_from_bf16(w[size_t(row) * k + j])) *
                                     double(f32_from_f16(scale)) * double(int8_t(b[4 + j % 32]));
                    expected += v; magnitude += std::fabs(v);
                }
                const float value = got[size_t(col) * n + row];
                if (!std::isfinite(value) || std::fabs(double(value) - expected) > 2e-6 * magnitude + 1e-6)
                    throw std::runtime_error("BF16/Q8_1 projection disagrees with host reference");
            }
            void* converted = nullptr;
            check(cudaMalloc(&converted, w.size() * 4));
            dequant_f32(30, dw, 1, n - 1, k, static_cast<float*>(converted), stream);
            check(cudaStreamSynchronize(stream));
            std::vector<float> floats(size_t(n - 1) * k);
            check(cudaMemcpy(floats.data(), converted, floats.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < floats.size(); ++i)
                if (floats[i] != f32_from_bf16(w[i + k])) throw std::runtime_error("BF16 row-slice copy changed values");
            dequant_f16(30, dw, 1, n - 1, k, static_cast<uint16_t*>(converted), stream);
            check(cudaStreamSynchronize(stream));
            std::vector<uint16_t> halves(floats.size());
            check(cudaMemcpy(halves.data(), converted, halves.size() * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < halves.size(); ++i)
                if (halves[i] != f16_from_f32(floats[i])) throw std::runtime_error("BF16 prompt conversion differs");
            check(cudaFree(converted));
            check(cudaFree(dy)); check(cudaFree(dq)); check(cudaFree(dx)); check(cudaFree(dw));
        }
        check(cudaStreamDestroy(stream));
        std::puts("bf16_q8_mmvq_parity OK: 12 shapes, all outputs finite and within summation tolerance");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
