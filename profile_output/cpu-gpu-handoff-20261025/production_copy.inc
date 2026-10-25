__global__ void copy_rows_from_mapped_kernel(float4* __restrict__ dst, const volatile float4* src, int64_t row4,
                                             const int32_t* __restrict__ hit_rows, const int32_t* __restrict__ count,
                                             const volatile uint32_t* wait_flag = nullptr, uint32_t wait_value = 0) {
    const int row = blockIdx.x;
    __shared__ int hit;
    if (threadIdx.x == 0) {
        if (wait_flag != nullptr) {   // 143-amdk: the wait_flag_ge launch folded in
            while (*wait_flag < wait_value) strata_spin_pause();
            __threadfence_system();
        }
        int h = 0;
        const int c = *count;
        for (int i = 0; i < c; ++i) h |= hit_rows[i] == row;
        hit = h;
    }
    __syncthreads();
    float4* d = dst + (int64_t) row * row4;
    if (hit) {
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        const volatile float4* sr = src + (int64_t) row * row4;
        for (int64_t i = threadIdx.x; i < row4; i += blockDim.x) d[i] = const_cast<const float4*>(sr)[i];
    }
}
__global__ void add_hits_kernel(float* __restrict__ parts, const float* __restrict__ hit_out,
                                const int32_t* __restrict__ dst, const int32_t* __restrict__ count, int n_embd) {
    const int h = blockIdx.y;
    if (h >= *count) return;
    const size_t row = (size_t) dst[h] * (size_t) n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        parts[row + i] += hit_out[row + i];
}
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    while (*flag < value) strata_spin_pause();
    __threadfence_system();
}
