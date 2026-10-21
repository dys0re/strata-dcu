#pragma once
// DTK otherwise gives an unannotated HIP kernel a 256-thread limit. These
// kernels really launch larger blocks. Keep every other backend unchanged.
#if defined(STRATA_HIP_GFX906)
#define STRATA_WAVE64_LAUNCH_BOUNDS(n) __launch_bounds__(n)
#else
#define STRATA_WAVE64_LAUNCH_BOUNDS(n)
#endif
