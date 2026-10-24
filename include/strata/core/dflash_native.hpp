#pragma once
#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {
class NativeHead;
class NativeEmbed;

// Owns only the trained draft parameters and state. The target owns embedding/head.
class DflashNative {
public:
    DflashNative();
    ~DflashNative();
    DflashNative(const DflashNative&) = delete;
    DflashNative& operator=(const DflashNative&) = delete;
    bool load(const std::string& dir, const NativeHead& head, const NativeEmbed& embed,
              int max_context, bool graph, bool adaptive, std::string& err);
    bool reset(std::string& err);
    // Source layout [5, source_capacity, 4, 2560], FP32, from captured target graphs.
    bool append(const float* source, int source_capacity, int rows, int64_t pos, std::string& err);
    bool draft(int64_t pos, int anchor, int cap, int32_t* ids, int& got, std::string& err);
    // MTP prefix tokens are embedded as known noise rows and locked in the
    // selector. All subsequent proposals are produced by DFlash2, conditioned
    // on that prefix. Total candidate count is explicit (joint budget owns it).
    bool draft_seeded(int64_t pos, int anchor, int count, const int32_t* prefix,
                      int n_prefix, double preparation_ms, int32_t* ids, int& got, std::string& err);
    // Forget only proposal feedback when prompt lookup replaces it; preserve target KV.
    void discard_feedback();
    uint64_t weight_bytes() const;
    // Standalone real-weight witness: packed FP32 [rows,12800], with the same forward/cache path.
    bool append_host(const float* features, int rows, int64_t pos, std::string& err);
    bool hidden(float* out, int rows, std::string& err) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::core
