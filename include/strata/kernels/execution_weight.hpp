#pragma once
#include <cstdint>

namespace strata::kernels {

enum class WeightLayout : uint8_t { Gguf, Q8Mmac };

// A borrowed execution view. The loader owns both allocations and must outlive every
// graph using them. GGUF type describes quantization; layout describes device storage.
// A Q8Mmac view never needs a second, row-major copy of the matrix.
struct ExecutionWeight {
    const void* data = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
    WeightLayout layout = WeightLayout::Gguf;
    float* partial = nullptr;
    uint64_t bytes = 0, workspace_bytes = 0;
    explicit operator bool() const { return data != nullptr; }
    bool gguf() const { return layout == WeightLayout::Gguf; }
};

} // namespace strata::kernels
