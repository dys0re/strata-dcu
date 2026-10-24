// CPU-only task partition parity. Optional real expert: pool_tasks_test GU_TYPE DOWN_TYPE BLOB_FILE
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace cpu = strata::kernels::cpu;

void pipeline_env(const char* value) {
#if defined(_WIN32)
    _putenv_s("STRATA_CPU_EXPERT_PIPELINE", value);
#else
    setenv("STRATA_CPU_EXPERT_PIPELINE", value, 1);
#endif
}

struct ReadyCheck {
    const std::vector<float>* reference;
    const std::vector<float>* output;
    int calls = 0;
    bool equal = true;
    static void check(void* ctx) {
        auto& c = *(ReadyCheck*) ctx;
        ++c.calls;
        c.equal &= std::memcmp(c.reference->data(), c.output->data(), c.output->size() * sizeof(float)) == 0;
    }
};

int main(int argc, char** argv) {
    if (argc != 1 && argc != 4) return 2;
    // its activation quantiser is the AVX one at least (act_quant_any): a CPU without AVX (Westmere) would stop on an illegal instruction
    if (!cpu::cpu_avx1_ok()) { std::printf("pool_tasks_test: no AVX on this CPU, skipped\n"); return 0; }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(argc == 4 ? std::atoi(argv[1]) : GGML_TYPE_Q4_0,
                         argc == 4 ? std::atoi(argv[2]) : GGML_TYPE_Q4_0, cpu::H, cpu::FF, f, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::vector<uint8_t> blob(f.bytes);
    if (argc == 4) {
        std::FILE* file = std::fopen(argv[3], "rb");
        if (!file) return 2;
        const size_t got = std::fread(blob.data(), 1, blob.size(), file);
        std::fclose(file);
        if (got != blob.size()) return 2;
    } else {
        std::vector<float> weights((size_t) cpu::H * cpu::FF);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.02f * std::sin((float) i * 0.17f);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data(), 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), blob.data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
    }
    for (int tasks : {-1, cpu::ExpertPool::kMaxTasks + 1}) {
        try { cpu::ExpertPool invalid(1, false, true, cpu::PoolAffinity::All, tasks); return 1; }
        catch (const std::invalid_argument&) {}
    }
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<cpu::ActQ> qact(cpu::MAXT);
    std::vector<float> x(cpu::H);
    for (int t = 0; t < cpu::MAXT; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[t].data());
        cpu::act_quant_any(x.data(), cpu::H, qact[t]);
    }
    int checks = 0;
    for (int n : {1, 7, cpu::ExpertPool::kMaxSplitMulti + 1}) {
        std::vector<cpu::ExpertJobMulti> jobs(n);
        std::vector<float> out((size_t) n * cpu::MAXT * cpu::H);
        for (int e = 0; e < n; ++e) {
            jobs[e].blob = blob.data();
            jobs[e].nt = e % 3 == 0 ? 1 : e % 3 == 1 ? 4 : cpu::MAXT;
            for (int t = 0; t < jobs[e].nt; ++t) {
                jobs[e].nact[t] = act[t].data();
                jobs[e].act[t] = &qact[t];
                jobs[e].out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
            }
        }
        const char* pipe = std::getenv("STRATA_CPU_EXPERT_PIPELINE");
        const std::string pipeline = pipe ? pipe : "0";
        pipeline_env("0");
        { // The default partition is the arithmetic reference; inactive token slots remain zero.
            cpu::ExpertPool baseline(3, false);
            baseline.run_split_multi_native(f, jobs.data(), n);
        }
        pipeline_env(pipeline.c_str());
        const auto ref = out;
        for (float v : ref) if (!std::isfinite(v)) return 1;
        for (bool host : {false, true}) for (int workers : {1, 3, 12})
            for (int tasks : {0, 1, 41, 128, 192, 256, cpu::ExpertPool::kMaxTasks}) {
                cpu::ExpertPool pool(workers, false, host, cpu::PoolAffinity::All, tasks);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    std::fill(out.begin(), out.end(), 0.0f);
                    for (const auto& job : jobs) for (int t = 0; t < job.nt; ++t)
                        std::fill(job.out[t], job.out[t] + cpu::H, std::numeric_limits<float>::quiet_NaN());
                    ReadyCheck ready{&ref, &out};
                    pool.run_split_multi_native(f, jobs.data(), n, &ReadyCheck::check, &ready);
                    if (ready.calls != 1 || !ready.equal || std::memcmp(ref.data(), out.data(), out.size() * sizeof(float)) != 0) {
                        std::fprintf(stderr, "mismatch n=%d workers=%d host=%d tasks=%d repeat=%d\n",
                                     n, workers, host, tasks, repeat);
                        return 1;
                    }
                    ++checks;
                }
            }
    }
    // Reuse one 12-worker pool across rapidly changing batch sizes and all window widths. A late worker must
    // never consume the next batch's description under the previous batch's epoch (including sleep/wake).
    std::vector<float> expected((size_t) cpu::MAXT * cpu::H), output((size_t) 13 * cpu::MAXT * cpu::H);
    std::vector<cpu::ExpertJobMulti> jobs(13);
    for (int e = 0; e < 13; ++e) {
        jobs[e].blob = blob.data(); jobs[e].nt = cpu::MAXT;
        for (int t = 0; t < cpu::MAXT; ++t) {
            jobs[e].nact[t] = act[t].data(); jobs[e].act[t] = &qact[t];
            jobs[e].out[t] = output.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
        }
    }
    const char* pipe = std::getenv("STRATA_CPU_EXPERT_PIPELINE");
    const std::string pipeline = pipe ? pipe : "0";
    pipeline_env("0");
    { cpu::ExpertPool refpool(3, false); refpool.run_split_multi_native(f, jobs.data(), 1); }
    std::copy(output.begin(), output.begin() + expected.size(), expected.begin());
    pipeline_env(pipeline.c_str());
    std::mt19937 random(20261024);
    {
        cpu::ExpertPool pool(12, false);
        for (int iteration = 0; iteration < 300; ++iteration) {
            const int n = (int) (random() % 14);
            for (int e = 0; e < n; ++e) {
                jobs[e].nt = (int) (random() % cpu::MAXT) + 1;
                for (int t = 0; t < jobs[e].nt; ++t)
                    std::fill(jobs[e].out[t], jobs[e].out[t] + cpu::H, std::numeric_limits<float>::quiet_NaN());
            }
            if (iteration % 7 == 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
            pool.run_split_multi_native(f, jobs.data(), n);
            for (int e = 0; e < n; ++e) for (int t = 0; t < jobs[e].nt; ++t)
                if (std::memcmp(jobs[e].out[t], expected.data() + (size_t) t * cpu::H, cpu::H * sizeof(float))) {
                    std::fprintf(stderr, "changing-batch mismatch iteration=%d n=%d e=%d t=%d\n", iteration, n, e, t);
                    return 1;
                }
            ++checks;
        }
    }
    std::printf("pool tasks: %d bitwise comparisons passed\n", checks);
    return 0;
}
