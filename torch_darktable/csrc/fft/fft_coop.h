#pragma once

#include "common.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<int N>
__device__ __forceinline__ Complex shfl(cg::thread_block_tile<N> warp, Complex value, int partner) {
    return Complex(warp.shfl(value.re, partner), warp.shfl(value.im, partner));
}

// Core FFT implementation
template<int N>
__device__ __forceinline__ Complex warp_fft(Complex my_data, const Complex* twiddles) {
    constexpr int stages = log2_constexpr(N);
    
    auto warp = cg::tiled_partition<N>(cg::this_thread_block());
    int tid = warp.thread_rank();
    
    // Bit-reverse permutation
    int rev_tid = __brev(tid) >> (32 - stages);
    Complex temp = my_data;
    my_data = shfl<N>(warp, temp, rev_tid);
    
    // Radix-2 butterflies
    constexpr int twiddle_size = N / 2;
    #pragma unroll
    for (int stage = 0; stage < stages; stage++) {
        int step = 1 << stage;
        int partner = tid ^ step;
        
        // Exchange data with butterfly partner
        Complex partner_data = shfl<N>(warp, my_data, partner);
               
        if ((tid & step) == 0) {
            int twiddle_idx = (tid & (step - 1)) * (twiddle_size >> stage);
            my_data = my_data + partner_data * twiddles[twiddle_idx];
        } else {
            int twiddle_idx = (partner & (step - 1)) * (twiddle_size >> stage);
            my_data = partner_data - my_data * twiddles[twiddle_idx];
        }
    }
    
    return my_data;
}

// Forward FFT
template<int N>
__device__ __forceinline__ Complex fft(Complex value) {
    return warp_fft<N>(value, get_fft_twiddles<N>());
}

// Inverse FFT
template<int N>
__device__ __forceinline__ Complex ifft(Complex value) {
    Complex result = warp_fft<N>(value, get_ifft_twiddles<N>());
    return result * (1.0f / N);
}

// Block transpose for 2D FFT
template<int K>
__device__ __forceinline__ Complex transpose_block(Complex value) {
    auto block = cg::this_thread_block();
    dim3 idx = block.thread_index();

    __shared__ Complex shared_tile[K*K];
    block.sync();

    shared_tile[idx.y * K + idx.x] = value;
    block.sync();

    return shared_tile[idx.x * K + idx.y];
}

// 2D Forward FFT  
template<int N>
__device__ __forceinline__ Complex fft_2d(Complex value) {
    value = fft<N>(value);
    value = transpose_block<N>(value);
    return fft<N>(value);
}

// 2D Inverse FFT
template<int N>
__device__ __forceinline__ Complex ifft_2d(Complex value) {
    value = ifft<N>(value);
    value = transpose_block<N>(value);
    return ifft<N>(value);
}
