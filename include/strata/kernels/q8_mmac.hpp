#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::kernels {
#if defined(STRATA_HIP_GFX906)
bool q8_mmac_enabled();
bool q8_mmac_eligible(int n_in, int n_out);
uint64_t q8_mmac_weight_bytes(int n_in, int n_out);
uint64_t q8_mmac_workspace_bytes(int n_out);
uint64_t q8_mmac_register_host(const void* canonical, const void* host_blocks, int n_in, int n_out);
void q8_mmac_unregister(const void* canonical);
bool q8_mmac_try(const void* canonical, const void* x, float* y, int n_in, int n_out, int ncols, void* stream);
#else
inline bool q8_mmac_enabled() { return false; }
inline bool q8_mmac_eligible(int, int) { return false; }
inline uint64_t q8_mmac_weight_bytes(int, int) { return 0; }
inline uint64_t q8_mmac_workspace_bytes(int) { return 0; }
inline uint64_t q8_mmac_register_host(const void*, const void*, int, int) { return 0; }
inline void q8_mmac_unregister(const void*) {}
inline bool q8_mmac_try(const void*, const void*, float*, int, int, int, void*) { return false; }
#endif
}
