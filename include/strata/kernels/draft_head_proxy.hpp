#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
// Optional approximate DRAFT logits only, fixed input width 2560. GGML BF16(30) and Q8_0(8).
// 32-weight groups: four levels plus one FP16 scale. Best four rows per 32-row tile feed a
// deterministic global shortlist (<=64 within five logits); those rows are recomputed from
// the original weights. Other logits stay approximate, including their softmax mass.
// All allocations and packing happen before capture. No target-verifier calls this API.
size_t draft_head_proxy_bytes(int rows);
size_t draft_head_proxy_scratch_bytes(int rows);
// Last 65 int32 words of scratch: count, then <=64 selected row indices (diagnostics/tests).
void draft_head_proxy_pack(int type, const void* weights, void* proxy, int rows, void* stream);
bool draft_head_proxy_logits(int type, const void* proxy, const void* weights, const void* x_q8_1,
                             float* logits, int rows, int columns, const int32_t* map,
                             void* scratch, void* stream, void* const* diagnostic_events = nullptr);
} // namespace strata::kernels
