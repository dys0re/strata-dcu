#include "strata/kernels/draft_head_proxy.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
constexpr int K = 2560, NB = K / 32;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
float fb(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
float fh(uint16_t h) { return k::f32_from_f16(h); }
uint16_t hf(float f) { return k::f16_from_f32(f); }
float noise(uint32_t& s) { s = s * 1664525u + 1013904223u; return float(int(s >> 16) - 32768) / 32768.f; }
float load(const std::vector<uint8_t>& w, int type, int row, int col) {
    if (type == 30) { uint16_t b; std::memcpy(&b, w.data() + (size_t(row) * K + col) * 2, 2); return fb(b); }
    const auto* p = w.data() + (size_t(row) * NB + col / 32) * 34;
    uint16_t d; std::memcpy(&d, p, 2); return fh(d) * float(int8_t(p[2 + col % 32]));
}

int main() try {
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    int cases = 0;
    for (int rows : {1, 31, 32, 33, 257, 4097, 106299}) for (int type : {30, 8}) {
        uint32_t random = 20261024;
        const size_t row_bytes = type == 30 ? K * 2 : NB * 34;
        std::vector<uint8_t> weights(size_t(rows) * row_bytes);
        for (int row = 0; row < rows; ++row) for (int b = 0; b < NB; ++b) {
            float block[32], amax = 0;
            for (float& f : block) { f = rows == 1 ? 0.f : fb(bf(noise(random) * .03f)); amax = std::max(amax, std::fabs(f)); }
            if (type == 30) {
                for (int j = 0; j < 32; ++j) { const uint16_t h = bf(block[j]); std::memcpy(weights.data() + (size_t(row) * K + b * 32 + j) * 2, &h, 2); }
            } else {
                auto* p = weights.data() + (size_t(row) * NB + b) * 34;
                const float d = amax / 127.f; const uint16_t h = hf(d); std::memcpy(p, &h, 2);
                for (int j = 0; j < 32; ++j) p[2 + j] = uint8_t(int8_t(d > 0 ? std::round(block[j] / d) : 0));
            }
        }
        std::vector<float> input(K), output(rows), again(rows), reference(rows);
        for (auto& v : input) v = noise(random);
        void *w, *proxy, *scratch, *xq; float *x, *y;
        ck(cudaMalloc(&w, weights.size())); ck(cudaMalloc(&proxy, k::draft_head_proxy_bytes(rows)));
        ck(cudaMalloc(&scratch, k::draft_head_proxy_scratch_bytes(rows)));
        ck(cudaMalloc(&xq, NB * 36)); ck(cudaMalloc(&x, K * sizeof(float))); ck(cudaMalloc(&y, rows * sizeof(float)));
        ck(cudaMemcpyAsync(w, weights.data(), weights.size(), cudaMemcpyHostToDevice, stream));
        ck(cudaMemcpyAsync(x, input.data(), K * 4, cudaMemcpyHostToDevice, stream));
        k::native_quantize_q8_1(x, xq, K, 1, stream);
        k::draft_head_proxy_pack(type, w, proxy, rows, stream);
        if (k::draft_head_proxy_logits(type, proxy, w, xq, y, rows, 2, nullptr, scratch, stream)) return 2;
        if (!k::draft_head_proxy_logits(type, proxy, w, xq, y, rows, 1, nullptr, scratch, stream)) return 2;
        ck(cudaMemcpyAsync(output.data(), y, rows * 4, cudaMemcpyDeviceToHost, stream));
        std::vector<uint8_t> pq(k::draft_head_proxy_bytes(rows)), act(NB * 36);
        int32_t selected[65] = {};
        ck(cudaMemcpyAsync(pq.data(), proxy, pq.size(), cudaMemcpyDeviceToHost, stream));
        ck(cudaMemcpyAsync(act.data(), xq, act.size(), cudaMemcpyDeviceToHost, stream));
        ck(cudaMemcpyAsync(selected, (uint8_t*) scratch + k::draft_head_proxy_scratch_bytes(rows) - sizeof(selected), sizeof(selected), cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));
        if (selected[0] < 1 || selected[0] > 64) return 3;
        std::vector<bool> exact(rows, false);
        for (int i = 1; i <= selected[0]; ++i) {
            if (selected[i] < 0 || selected[i] >= rows || exact[selected[i]]) return 4;
            exact[selected[i]] = true;
        }
        const int tiles = (rows + 31) / 32;
        const size_t scale_off = size_t(tiles) * 40 * 32 * 16;
        auto coarse = [&](int row) {
            double expected = 0;
            for (int b = 0; b < NB; ++b) {
                const size_t at = ((size_t(row / 32) * 40 + b / 2) * 32 + row % 32);
                uint16_t s, dx; std::memcpy(&s, pq.data() + scale_off + (at * 2 + b % 2) * 2, 2);
                std::memcpy(&dx, act.data() + b * 36, 2);
                for (int j = 0; j < 32; ++j) {
                    uint32_t word; std::memcpy(&word, pq.data() + (at * 4 + 2 * (b % 2) + j / 16) * 4, 4);
                    const int code = int((word >> (8 * (j % 4) + 2 * ((j % 16) / 4))) & 3);
                    expected += (code - 1.5) * fh(s) * int8_t(act[b * 36 + 4 + j]) * fh(dx);
                }
            }
            return expected;
        };
        for (int row = 0; row < rows; row += std::max(1, rows / 17)) {
            if (exact[row]) continue;
            const double expected = coarse(row);
            if (!std::isfinite(output[row]) || std::abs(output[row] - expected) > 2e-4 * (1 + std::abs(expected))) {
                std::printf("coarse mismatch type=%d rows=%d row=%d got=%g expected=%g\n", type, rows, row, output[row], expected); return 5;
            }
        }
        if (rows <= 257) {
            using Item = std::pair<double, int>;
            const auto order = [](Item a, Item b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
            std::vector<Item> candidates;
            for (int base = 0; base < rows; base += 32) {
                std::vector<Item> tile;
                for (int row = base; row < std::min(rows, base + 32); ++row) tile.push_back({coarse(row), row});
                std::sort(tile.begin(), tile.end(), order);
                candidates.insert(candidates.end(), tile.begin(), tile.begin() + std::min(size_t(4), tile.size()));
            }
            std::sort(candidates.begin(), candidates.end(), order);
            int expected = 0;
            while (expected < 64 && expected < int(candidates.size()) && candidates[expected].first >= candidates[0].first - 5) ++expected;
            if (selected[0] != expected) return 8;
            for (int i = 0; i < expected; ++i) if (selected[i + 1] != candidates[i].second) {
                std::printf("candidate order mismatch type=%d rows=%d at=%d\n", type, rows, i); return 9;
            }
        }
        for (int i = 1; i <= selected[0]; ++i) {
            const int row = selected[i]; double expected = 0;
            for (int b = 0; b < NB; ++b) {
                uint16_t d; std::memcpy(&d, act.data() + b * 36, 2);
                for (int j = 0; j < 32; ++j) expected += double(load(weights, type, row, b * 32 + j)) * int8_t(act[b * 36 + 4 + j]) * fh(d);
            }
            if (std::abs(output[row] - expected) > 2e-4 * (1 + std::abs(expected))) {
                std::printf("exact mismatch type=%d rows=%d row=%d got=%g expected=%g\n", type, rows, row, output[row], expected); return 6;
            }
        }
        // Captured execution must select the same shortlist and write the same logits on buffer reuse.
        cudaGraph_t graph; cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        k::draft_head_proxy_logits(type, proxy, w, xq, y, rows, 1, nullptr, scratch, stream);
        ck(cudaStreamEndCapture(stream, &graph)); ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int repeat = 0; repeat < 3; ++repeat) ck(cudaGraphLaunch(exec, stream));
        ck(cudaMemcpyAsync(again.data(), y, rows * 4, cudaMemcpyDeviceToHost, stream)); ck(cudaStreamSynchronize(stream));
        if (std::memcmp(output.data(), again.data(), rows * 4)) return 7;
        k::native_mmvq(type, w, xq, y, K, rows, 1, stream);
        ck(cudaMemcpyAsync(reference.data(), y, rows * 4, cudaMemcpyDeviceToHost, stream)); ck(cudaStreamSynchronize(stream));
        const auto pick = [](const std::vector<float>& a) { return int(std::max_element(a.begin(), a.end()) - a.begin()); };
        std::printf("PASS type=%d rows=%d rescored=%d pick_proxy=%d pick_reference=%d\n", type, rows, selected[0], pick(output), pick(reference));
        if (rows == 106299) {
            cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b)); float full_ms, proxy_ms;
            ck(cudaEventRecord(a, stream));
            for (int i = 0; i < 30; ++i) k::native_mmvq(type, w, xq, y, K, rows, 1, stream);
            ck(cudaEventRecord(b, stream)); ck(cudaEventSynchronize(b)); ck(cudaEventElapsedTime(&full_ms, a, b));
            ck(cudaEventRecord(a, stream)); for (int i = 0; i < 30; ++i) ck(cudaGraphLaunch(exec, stream));
            ck(cudaEventRecord(b, stream)); ck(cudaEventSynchronize(b)); ck(cudaEventElapsedTime(&proxy_ms, a, b));
            std::printf("MICROBENCH type=%d full_us=%.3f proxy_us=%.3f\n", type, full_ms * 1000 / 30, proxy_ms * 1000 / 30);
            cudaEvent_t marks[4]; void* events[4];
            for (int j = 0; j < 4; ++j) { ck(cudaEventCreate(&marks[j])); events[j] = marks[j]; }
            k::draft_head_proxy_logits(type, proxy, w, xq, y, rows, 1, nullptr, scratch, stream, events);
            ck(cudaEventSynchronize(marks[3]));
            for (int j = 0; j < 3; ++j) { float ms; ck(cudaEventElapsedTime(&ms, marks[j], marks[j+1])); std::printf("STAGE type=%d stage=%d us=%.3f\n",type,j,ms*1000); }
            for (auto event : marks) ck(cudaEventDestroy(event));
            ck(cudaEventDestroy(a)); ck(cudaEventDestroy(b));
        }
        if (rows == 33) {
            int32_t* map; ck(cudaMalloc(&map, rows * sizeof(int32_t))); ck(cudaMemsetAsync(map, 0xff, rows * sizeof(int32_t), stream));
            k::draft_head_proxy_logits(type, proxy, w, xq, y, rows, 1, map, scratch, stream);
            ck(cudaMemcpyAsync(again.data(), y, rows * 4, cudaMemcpyDeviceToHost, stream)); ck(cudaStreamSynchronize(stream));
            for (float v : again) if (v != -INFINITY) return 10;
            ck(cudaFree(map)); ++cases;
        }
        ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
        ck(cudaFree(w)); ck(cudaFree(proxy)); ck(cudaFree(scratch)); ck(cudaFree(xq)); ck(cudaFree(x)); ck(cudaFree(y)); ++cases;
    }
    ck(cudaStreamDestroy(stream)); std::printf("draft_head_proxy_test: %d cases passed\n", cases); return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
