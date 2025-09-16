#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <array>

namespace cg = cooperative_groups;

struct Complex {
    float re, im;
    
    __device__ __host__ constexpr Complex() : re(0.0f), im(0.0f) {}
    __device__ __host__ constexpr Complex(float real, float imag) : re(real), im(imag) {}
    
    __device__ __host__ constexpr Complex operator+(const Complex& other) const {
        return Complex(re + other.re, im + other.im);
    }
    
    __device__ __host__ constexpr Complex operator-(const Complex& other) const {
        return Complex(re - other.re, im - other.im);
    }
    
    __device__ __host__ constexpr Complex operator*(const Complex& other) const {
        return Complex(re * other.re - im * other.im, re * other.im + im * other.re);
    }
    
    __device__ __host__ constexpr Complex operator*(float scalar) const {
        return Complex(re * scalar, im * scalar);
    }
    
    __device__ __host__ constexpr Complex operator+(float scalar) const {
        return Complex(re + scalar, im);
    }
    
    __device__ __host__ constexpr Complex operator-(float scalar) const {
        return Complex(re - scalar, im);
    }
    
    __device__ __host__ constexpr float magnitude_squared() const {
        return re * re + im * im;
    }
};

// Scalar operations from the left
__device__ __host__ constexpr Complex operator*(float scalar, const Complex& c) {
    return Complex(scalar * c.re, scalar * c.im);
}

__device__ __host__ constexpr Complex operator+(float scalar, const Complex& c) {
    return Complex(scalar + c.re, c.im);
}

__device__ __host__ constexpr Complex operator-(float scalar, const Complex& c) {
    return Complex(scalar - c.re, -c.im);
}

// Precomputed twiddle factors for 16-point FFT
__device__ constexpr Complex fft_twiddles_16[8] = {
    {1.0f, 0.0f},                    // k=0
    {0.9238795325f, -0.3826834324f}, // k=1
    {0.7071067812f, -0.7071067812f}, // k=2
    {0.3826834324f, -0.9238795325f}, // k=3
    {0.0f, -1.0f},                   // k=4
    {-0.3826834324f, -0.9238795325f}, // k=5
    {-0.7071067812f, -0.7071067812f}, // k=6
    {-0.9238795325f, -0.3826834324f}  // k=7
};

__device__ constexpr Complex ifft_twiddles_16[8] = {
    {1.0f, 0.0f},                    // k=0
    {0.9238795325f, 0.3826834324f},  // k=1
    {0.7071067812f, 0.7071067812f},  // k=2
    {0.3826834324f, 0.9238795325f},  // k=3
    {0.0f, 1.0f},                    // k=4
    {-0.3826834324f, 0.9238795325f}, // k=5
    {-0.7071067812f, 0.7071067812f}, // k=6
    {-0.9238795325f, 0.3826834324f}  // k=7
};

// Precomputed twiddle factors for 32-point FFT
__device__ constexpr Complex fft_twiddles_32[16] = {
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

__device__ constexpr Complex ifft_twiddles_32[16] = {
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

// Compile-time log2 calculation
constexpr int log2_constexpr(int n) {
    return (n <= 1) ? 0 : 1 + log2_constexpr(n / 2);
}


template<int N>
__device__ __forceinline__ Complex shfl(cg::thread_block_tile<N> warp, Complex value, int partner) {
    return Complex(warp.shfl(value.re, partner), warp.shfl(value.im, partner));
}


// Generic cooperative groups FFT template
template<int N, bool inverse>
__device__ __forceinline__ Complex coop_warp_fft(Complex my_data, float norm_factor, const Complex* twiddles) {
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
    
    return my_data * norm_factor;
}

// Templated FFT class
template<int K>
struct FFT;

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


template<>
struct FFT<16> {
    __device__ __forceinline__ static void fft_1d(Complex& data) {
        data = coop_warp_fft<16, false>(data, 1.0f, fft_twiddles_16);
    }
    
    __device__ __forceinline__ static void ifft_1d(Complex& data) {
        data = coop_warp_fft<16, true>(data, 1.0f / 16.0f, ifft_twiddles_16);
    }
    
    __device__ __forceinline__ static Complex fft_2d(Complex value) {
        value = coop_warp_fft<16, false>(value, 1.0f, fft_twiddles_16);
        value = transpose_block<16>(value);
        return coop_warp_fft<16, false>(value, 1.0f, fft_twiddles_16);
    }
    
    __device__ __forceinline__ static Complex ifft_2d(Complex value) {                
        value = coop_warp_fft<16, true>(value, 1.0f / 16.0f, ifft_twiddles_16);
        value = transpose_block<16>(value);
        return coop_warp_fft<16, true>(value, 1.0f / 16.0f, ifft_twiddles_16);
    }
};

template<>
struct FFT<32> {
    __device__ __forceinline__ static void fft_1d(Complex& data) {
        data = coop_warp_fft<32, false>(data, 1.0f, fft_twiddles_32);
    }
    
    __device__ __forceinline__ static void ifft_1d(Complex& data) {
        data = coop_warp_fft<32, true>(data, 1.0f / 32.0f, ifft_twiddles_32);
    }
    
    __device__ __forceinline__ static Complex fft_2d(Complex value) {
        value = coop_warp_fft<32, false>(value, 1.0f, fft_twiddles_32);
        value = transpose_block<32>(value);
        return coop_warp_fft<32, false>(value, 1.0f, fft_twiddles_32);
    }
    
    __device__ __forceinline__ static Complex ifft_2d(Complex value) {
        value = coop_warp_fft<32, true>(value, 1.0f / 32.0f, ifft_twiddles_32);
        value = transpose_block<32>(value);
        return coop_warp_fft<32, true>(value, 1.0f / 32.0f, ifft_twiddles_32);
    }
};


