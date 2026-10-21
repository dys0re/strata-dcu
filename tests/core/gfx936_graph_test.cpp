// Regression for DTK's default block limit and crashing eager graph upload.
// Verify the graph actually computes, not just that capture returns success.
#include "strata/core/graph_upload.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
int main() {
    try {
        constexpr int cols = 2560, rows = 3;
        std::vector<float> input(cols * rows), gamma(cols * rows), output(cols * rows);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::sin(float(i) * .17f);
            gamma[i] = 1 + std::cos(float(i) * .31f) * .1f;
        }
        void *x = nullptr, *g = nullptr, *y = nullptr;
        check(cudaMalloc(&x, input.size() * 4)); check(cudaMalloc(&g, gamma.size() * 4));
        check(cudaMalloc(&y, output.size() * 4));
        check(cudaMemcpy(x, input.data(), input.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(g, gamma.data(), gamma.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemset(y, 0, output.size() * 4));
        cudaStream_t stream; check(cudaStreamCreate(&stream));
        cudaGraph_t graph; cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        strata::kernels::native_gr_rms_norm_weighted(static_cast<float*>(x), static_cast<float*>(g),
                                                    static_cast<float*>(y), cols, rows, 1e-6f, stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        check(strata::core::graph_upload(exec, stream));
        check(cudaGraphLaunch(exec, stream));
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(output.data(), y, output.size() * 4, cudaMemcpyDeviceToHost));
        for (int row = 0; row < rows; ++row) {
            double square = 0;
            for (int j = 0; j < cols; ++j) square += double(input[row * cols + j]) * input[row * cols + j];
            const double scale = 1 / std::sqrt(square / cols + 1e-6);
            for (int j = 0; j < cols; ++j) {
                const size_t i = row * cols + j;
                const double expected = scale * input[i] * gamma[i];
                if (!std::isfinite(output[i]) || std::fabs(output[i] - expected) > 2e-5 * std::fabs(expected) + 1e-6)
                    throw std::runtime_error("captured 1024-thread RMSNorm differs from host reference");
            }
        }
        check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
        check(cudaFree(y)); check(cudaFree(g)); check(cudaFree(x));
        // The real PLE gate launches 512 threads. With zero convolution taps,
        // result = hidden + sigmoid(signed_sqrt(norm(key).norm(hidden)/sqrt(N))).
        // A silently rejected launch would leave gate zero and must fail here.
        constexpr int dim = 4 * cols;
        std::vector<float> hidden(dim), ones(dim, 1.0f), values(cols, 1.0f), got(dim), gates(4);
        for (int i = 0; i < dim; ++i) hidden[i] = std::sin(float(i) * .07f);
        std::vector<void*> allocations;
        auto alloc = [&](size_t bytes) { void* p = nullptr; check(cudaMalloc(&p, bytes)); check(cudaMemset(p, 0, bytes)); allocations.push_back(p); return p; };
        auto* key = static_cast<float*>(alloc(dim * 4));
        auto* h = static_cast<float*>(alloc(dim * 4));
        auto* val = static_cast<float*>(alloc(cols * 4));
        auto* gamma_p = static_cast<float*>(alloc(dim * 4));
        auto* hist = static_cast<float*>(alloc(9 * dim * 4));
        auto* query = static_cast<float*>(alloc(dim * 4));
        auto* gated = static_cast<float*>(alloc(dim * 4));
        auto* gate = static_cast<float*>(alloc(4 * 4));
        auto* taps = static_cast<uint16_t*>(alloc(dim * 4 * 2));
        auto* snap = static_cast<float*>(alloc(9 * dim * 4));
        strata::kernels::PleWeights weights{};
        weights.norm_key = weights.norm_query = weights.norm_conv = gamma_p;
        weights.conv1d_f16 = taps;
        check(cudaMemcpy(val, values.data(), values.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(gamma_p, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
        for (bool snapshot : {false, true}) {
            check(cudaMemcpy(key, hidden.data(), hidden.size() * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(h, hidden.data(), hidden.size() * 4, cudaMemcpyHostToDevice));
            if (snapshot) strata::kernels::native_ple_postops_batch_snap(key, h, val, hist, weights, query, gated, gate, 1, snap, stream);
            else strata::kernels::native_ple_postops_batch(key, h, val, hist, weights, query, gated, gate, 1, stream);
            check(cudaStreamSynchronize(stream));
            check(cudaMemcpy(got.data(), h, got.size() * 4, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(gates.data(), gate, gates.size() * 4, cudaMemcpyDeviceToHost));
            for (int row = 0; row < 4; ++row) {
                double square = 0;
                for (int j = 0; j < cols; ++j) square += double(hidden[row * cols + j]) * hidden[row * cols + j];
                const double dot = square / (square / cols + 1e-6);
                const double expected = 1 / (1 + std::exp(-std::sqrt(dot / std::sqrt(double(cols)))));
                if (std::fabs(gates[row] - expected) > 1e-5) throw std::runtime_error("PLE 512-thread gate differs");
                for (int j = 0; j < cols; ++j)
                    if (std::fabs(got[row * cols + j] - (hidden[row * cols + j] + expected)) > 1e-5)
                        throw std::runtime_error("PLE gate/residual differs");
            }
        }
        for (void* p : allocations) check(cudaFree(p));
        check(cudaStreamDestroy(stream));
        std::puts("gfx936_graph_test OK: 1024-thread norm captured, launched, and matched host reference");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
