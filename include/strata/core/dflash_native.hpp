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
    uint64_t weight_bytes() const;
    // Standalone real-weight witness: packed FP32 [rows,12800], with the same forward/cache path.
    bool append_host(const float* features, int rows, int64_t pos, std::string& err);
    bool hidden(float* out, int rows, std::string& err) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::core
