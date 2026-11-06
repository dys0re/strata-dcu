#pragma once
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

namespace strata::kernels {

// Reconstruct only the requested slice into caller-owned storage. No allocation
// or synchronization; the caller orders reuse on its stream. False means the
// GGUF format has no padded-row dequantizer and nothing was launched.
inline bool execution_dequant_f16(const ExecutionWeight& w, int64_t row0, int64_t rows,
                                  int64_t ld, uint16_t* out, void* stream) {
    if (!w.data || row0 < 0 || rows < 0 || row0 > w.n_out || rows > w.n_out - row0 || ld < w.n_in || !out)
        throw std::invalid_argument("execution weight dequant slice mismatch");
    if (!rows) return true;
    if (!w.gguf()) q8_mmac_dequant_f16(w, row0, rows, ld, out, stream);
    else if (ld == w.n_in) dequant_f16(w.type, w.data, row0, rows, w.n_in, out, stream);
    else return dequant_f16_ld(w.type, w.data, row0, rows, w.n_in, ld, out, stream);
    return true;
}

// Validated device row IDs -> canonical GGUF rows, preserving original codes and
// scales. Used for MTP subsets; does not reconstruct the complete source matrix.
inline void execution_gather_rows(const ExecutionWeight& w, const int32_t* ids, int64_t rows,
                                   uint8_t* out, void* stream) {
    if (!w.data || !ids || rows < 0 || !out)
        throw std::invalid_argument("invalid execution weight row gather");
    if (!rows) return;
    if (w.gguf()) gather_rows(static_cast<const uint8_t*>(w.data),
                              native_mmvq_weight_bytes(w.type, w.n_in, 1), ids, rows, out, stream);
    else q8_mmac_gather_rows(w, ids, 0, rows, out, stream);
}

} // namespace strata::kernels
