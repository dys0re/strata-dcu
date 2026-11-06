#pragma once
#include "strata/kernels/execution_weight.hpp"
#include <stdexcept>
namespace strata::kernels {
#if defined(STRATA_HIP_GFX906)
bool q8_mmac_enabled();
bool q8_mmac_eligible(int n_in, int n_out);
uint64_t q8_mmac_weight_bytes(int n_in, int n_out);
uint64_t q8_mmac_workspace_bytes(int n_out);
// Allocates the sole execution copy and its scratch. Caller owns both allocations.
ExecutionWeight q8_mmac_upload_host(const void* host_blocks, int n_in, int n_out);
void q8_mmac_run(const ExecutionWeight& w, const void* x, float* y, int ncols, void* stream);
void q8_mmac_dequant_f16(const ExecutionWeight& w, int64_t row0, int64_t rows,
                        int64_t ld, uint16_t* out, void* stream);
// Reconstruct canonical Q8_0 rows without dequantizing/requantizing. Null indices = contiguous rows from row0.
void q8_mmac_gather_rows(const ExecutionWeight& w, const int32_t* indices, int64_t row0,
                         int64_t rows, void* out, void* stream);
#else
inline bool q8_mmac_enabled() { return false; }
inline bool q8_mmac_eligible(int, int) { return false; }
inline uint64_t q8_mmac_weight_bytes(int, int) { return 0; }
inline uint64_t q8_mmac_workspace_bytes(int) { return 0; }
inline ExecutionWeight q8_mmac_upload_host(const void*, int, int) { throw std::runtime_error("MMAC unavailable"); }
inline void q8_mmac_run(const ExecutionWeight&, const void*, float*, int, void*) { throw std::runtime_error("MMAC unavailable"); }
inline void q8_mmac_dequant_f16(const ExecutionWeight&, int64_t, int64_t, int64_t, uint16_t*, void*) { throw std::runtime_error("MMAC unavailable"); }
inline void q8_mmac_gather_rows(const ExecutionWeight&, const int32_t*, int64_t, int64_t, void*, void*) { throw std::runtime_error("MMAC unavailable"); }
#endif
}
