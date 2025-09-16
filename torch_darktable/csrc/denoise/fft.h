#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <array>

namespace cg = cooperative_groups;

#include <cutlass/numeric_types.h>
#include <cutlass/complex.h>



// Twiddle factors in half precision
__device__ constexpr float fft_twiddles_16[8][2] = {
    {1.0f, 0.0f},                    // k=0
    {0.9238795325f, -0.3826834324f}, // k=1
    {0.7071067812f, -0.7071067812f}, // k=2
    {0.3826834324f, -0.9238795325f}, // k=3
    {0.0f, -1.0f},                   // k=4
    {-0.3826834324f, -0.9238795325f}, // k=5
    {-0.7071067812f, -0.7071067812f}, // k=6
    {-0.9238795325f, -0.3826834324f}  // k=7
};

__device__ constexpr float ifft_twiddles_16[8][2] = {
    {1.0f, 0.0f},                    // k=0
    {0.9238795325f, 0.3826834324f},  // k=1
    {0.7071067812f, 0.7071067812f},  // k=2
    {0.3826834324f, 0.9238795325f},  // k=3
    {0.0f, 1.0f},                    // k=4
    {-0.3826834324f, 0.9238795325f}, // k=5
    {-0.7071067812f, 0.7071067812f}, // k=6
    {-0.9238795325f, 0.3826834324f}  // k=7
};

__device__ constexpr float fft_twiddles_32[16][2] = {
    {1.0f, 0.0f},                    // k=0
    {0.9807852804f, -0.1950903220f}, // k=1  
    {0.9238795325f, -0.3826834324f}, // k=2
    {0.8314696123f, -0.5555702330f}, // k=3
    {0.7071067812f, -0.7071067812f}, // k=4
    {0.5555702330f, -0.8314696123f}, // k=5
    {0.3826834324f, -0.9238795325f}, // k=6
    {0.1950903220f, -0.9807852804f}, // k=7
    {0.0f, -1.0f},                   // k=8
    {-0.1950903220f, -0.9807852804f}, // k=9
    {-0.3826834324f, -0.9238795325f}, // k=10
    {-0.5555702330f, -0.8314696123f}, // k=11
    {-0.7071067812f, -0.7071067812f}, // k=12
    {-0.8314696123f, -0.5555702330f}, // k=13
    {-0.9238795325f, -0.3826834324f}, // k=14
    {-0.9807852804f, -0.1950903220f}  // k=15
};

__device__ constexpr float ifft_twiddles_32[16][2] = {
    {1.0f, 0.0f},                    // k=0
    {0.9807852804f, 0.1950903220f},  // k=1
    {0.9238795325f, 0.3826834324f},  // k=2
    {0.8314696123f, 0.5555702330f},  // k=3
    {0.7071067812f, 0.7071067812f},  // k=4
    {0.5555702330f, 0.8314696123f},  // k=5
    {0.3826834324f, 0.9238795325f},  // k=6
    {0.1950903220f, 0.9807852804f},  // k=7
    {0.0f, 1.0f},                    // k=8
    {-0.1950903220f, 0.9807852804f}, // k=9
    {-0.3826834324f, 0.9238795325f}, // k=10
    {-0.5555702330f, 0.8314696123f}, // k=11
    {-0.7071067812f, 0.7071067812f}, // k=12
    {-0.8314696123f, 0.5555702330f}, // k=13
    {-0.9238795325f, 0.3826834324f}, // k=14
    {-0.9807852804f, 0.1950903220f}  // k=15
};

template<int K, bool inverse>
struct Twiddles;

template<>
struct Twiddles<16, false> {
    static __device__ __forceinline__ constexpr Complex get(int idx) {
        return Complex(half_t(fft_twiddles_16[idx][0]), half_t(fft_twiddles_16[idx][1]));
    }
};

template<>
struct Twiddles<32, false> {
    static __device__ __forceinline__ constexpr Complex get(int idx) {
        return Complex(half_t(fft_twiddles_32[idx][0]), half_t(fft_twiddles_32[idx][1]));
    }
};

template<>
struct Twiddles<16, true> {
    static __device__ __forceinline__ constexpr Complex get(int idx) {
        return Complex(half_t(ifft_twiddles_16[idx][0]), half_t(ifft_twiddles_16[idx][1]));
    }
};

template<>
struct Twiddles<32, true> {
    static __device__ __forceinline__ constexpr Complex get(int idx) {
        return Complex(half_t(ifft_twiddles_32[idx][0]), half_t(ifft_twiddles_32[idx][1]));
    }
};

// Compile-time log2 calculation
constexpr int log2_constexpr(int n) {
    return (n <= 1) ? 0 : 1 + log2_constexpr(n / 2);
}


template<int N>
__device__ __forceinline__ Complex shfl(cg::thread_block_tile<N> warp, Complex value, int partner) {
    return Complex(warp.shfl(value.real(), partner), warp.shfl(value.imag(), partner));
}


// Generic cooperative groups FFT template
template<int N, bool inverse>
__device__ __forceinline__ Complex coop_warp_fft(Complex my_data, half_t norm_factor) {
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

            auto twiddle = Twiddles<N, inverse>::get(twiddle_idx);
            my_data = my_data + partner_data * twiddle;
        } else {
            int twiddle_idx = (partner & (step - 1)) * (twiddle_size >> stage);
            auto twiddle = Twiddles<N, inverse>::get(twiddle_idx);

            my_data = partner_data - my_data * twiddle;
        }
    }
    
    return my_data * norm_factor;
}

// Templated FFT class
template<int K>
struct FFT;

template<int K>
__device__ __forceinline__ Complex transpose_block(Complex value) {
    auto block = cg::this_thread_block();
    dim3 idx = block.thread_index();

    __shared__ Array<Complex, K*K> shared_tile;
    block.sync();

    shared_tile[idx.y * K + idx.x] = value;
    block.sync();

    return shared_tile[idx.x * K + idx.y];
}


template<>
struct FFT<16> {
    __device__ __forceinline__ static void fft_1d(Complex& data) {
        data = coop_warp_fft<16, false>(data, half_t(1.0f));
    }
    
    __device__ __forceinline__ static void ifft_1d(Complex& data) {
        data = coop_warp_fft<16, true>(data, half_t(1.0f / 16.0f));
    }
    
    __device__ __forceinline__ static Complex fft_2d(Complex value) {
        value = coop_warp_fft<16, false>(value, half_t(1.0f));
        value = transpose_block<16>(value);
        return coop_warp_fft<16, false>(value, half_t(1.0f));
    }
    
    __device__ __forceinline__ static Complex ifft_2d(Complex value) {                
        value = coop_warp_fft<16, true>(value, half_t(1.0f / 16.0f));
        value = transpose_block<16>(value);
        return coop_warp_fft<16, true>(value, half_t(1.0f / 16.0f));
    }
};

template<>
struct FFT<32> {
    __device__ __forceinline__ static void fft_1d(Complex& data) {
        data = coop_warp_fft<32, false>(data, half_t(1.0f));
    }
    
    __device__ __forceinline__ static void ifft_1d(Complex& data) {
        data = coop_warp_fft<32, true>(data, half_t(1.0f / 32.0f));
    }
    
    __device__ __forceinline__ static Complex fft_2d(Complex value) {
        value = coop_warp_fft<32, false>(value, half_t(1.0f));
        value = transpose_block<32>(value);
        return coop_warp_fft<32, false>(value, half_t(1.0f));
    }
    
    __device__ __forceinline__ static Complex ifft_2d(Complex value) {
        value = coop_warp_fft<32, true>(value, half_t(1.0f / 32.0f));
        value = transpose_block<32>(value);
        return coop_warp_fft<32, true>(value, half_t(1.0f / 32.0f));
    }
};


