#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <array>

namespace cg = cooperative_groups;

struct Complex {
    float x, y;
    
    __device__ __host__ constexpr Complex() : x(0.0f), y(0.0f) {}
    __device__ __host__ constexpr Complex(float real, float imag) : x(real), y(imag) {}
    
    __device__ __host__ constexpr Complex operator+(const Complex& other) const {
        return Complex(x + other.x, y + other.y);
    }
    
    __device__ __host__ constexpr Complex operator-(const Complex& other) const {
        return Complex(x - other.x, y - other.y);
    }
    
    __device__ __host__ constexpr Complex operator*(const Complex& other) const {
        return Complex(x * other.x - y * other.y, x * other.y + y * other.x);
    }
    
    __device__ __host__ constexpr Complex operator*(float scalar) const {
        return Complex(x * scalar, y * scalar);
    }
    
    __device__ __host__ constexpr Complex operator+(float scalar) const {
        return Complex(x + scalar, y);
    }
    
    __device__ __host__ constexpr Complex operator-(float scalar) const {
        return Complex(x - scalar, y);
    }
    
    __device__ __host__ constexpr float magnitude_squared() const {
        return x * x + y * y;
    }
};

// Scalar operations from the left
__device__ __host__ constexpr Complex operator*(float scalar, const Complex& c) {
    return Complex(scalar * c.x, scalar * c.y);
}

__device__ __host__ constexpr Complex operator+(float scalar, const Complex& c) {
    return Complex(scalar + c.x, c.y);
}

__device__ __host__ constexpr Complex operator-(float scalar, const Complex& c) {
    return Complex(scalar - c.x, -c.y);
}

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




// Cooperative groups version - more explicit about warp cooperation  
template<bool inverse>
__device__ void coop_warp_fft_32(Complex& my_data) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int tid = warp.thread_rank();  // 0-31
    
    // Bit-reverse permutation
    int rev_tid = __brev(tid) >> (32 - 5);
    Complex temp = my_data;
    my_data.x = warp.shfl(temp.x, rev_tid);
    my_data.y = warp.shfl(temp.y, rev_tid);
    
    // 5 stages of radix-2 butterflies
    for (int stage = 0; stage < 5; stage++) {
        int step = 1 << stage;
        int partner = tid ^ step;
        
        // Exchange data with butterfly partner
        Complex partner_data;
        partner_data.x = warp.shfl(my_data.x, partner);
        partner_data.y = warp.shfl(my_data.y, partner);
        
        bool is_upper = (tid & step) == 0;
        
        if (is_upper) {
            int twiddle_idx = (tid & (step - 1)) * (16 >> stage);
            Complex w = inverse ? ifft_twiddles_32[twiddle_idx] : fft_twiddles_32[twiddle_idx];
            Complex bw = partner_data * w;
            my_data = my_data + bw;
        } else {
            int twiddle_idx = (partner & (step - 1)) * (16 >> stage);
            Complex w = inverse ? ifft_twiddles_32[twiddle_idx] : fft_twiddles_32[twiddle_idx];
            Complex aw = my_data * w;
            my_data = partner_data - aw;
        }
    }
    
    if (inverse) {
        my_data = my_data * (1.0f / 32.0f);
    }
}



// Simple wrapper functions
__device__ void fft(Complex& data) {
    coop_warp_fft_32<false>(data);
}

__device__ void ifft(Complex& data) {
    coop_warp_fft_32<true>(data);
}

// 2D FFT using shared memory for transpose between row/column stages
__device__ void fft_2d_32x32(Complex shared_tile[32*32]) {
    auto block = cg::this_thread_block();
    dim3 idx = block.thread_index();
    int tile_idx = idx.y * 32 + idx.x;
    
    // Phase 1: Row FFTs
    // Each thread loads one element from its row
    Complex my_data = shared_tile[tile_idx];
    
    // Do 1D FFT on this row using warp cooperation
    fft(my_data);
    
    // Store back to shared memory
    shared_tile[tile_idx] = my_data;
    block.sync();
    
    // Phase 2: Column FFTs  
    // Load element from column (transpose access)
    int transpose_idx = idx.x * 32 + idx.y;
    my_data = shared_tile[transpose_idx];
    
    // Do 1D FFT on this column
    fft(my_data);
    
    // Store back to shared memory
    shared_tile[transpose_idx] = my_data;
    block.sync();
}

__device__ void ifft_2d_32x32(Complex shared_tile[32*32]) {
    auto block = cg::this_thread_block();
    dim3 idx = block.thread_index();
    int tile_idx = idx.y * 32 + idx.x;
    
    // Phase 1: Row IFFTs
    // Each thread loads one element from its row
    Complex my_data = shared_tile[tile_idx];
    
    // Do 1D IFFT on this row using warp cooperation
    ifft(my_data);
    
    // Store back to shared memory
    shared_tile[tile_idx] = my_data;
    block.sync();
    
    // Phase 2: Column IFFTs  
    // Load element from column (transpose access)
    int transpose_idx = idx.x * 32 + idx.y;
    my_data = shared_tile[transpose_idx];
    
    // Do 1D IFFT on this column
    ifft(my_data);
    
    // Store back to shared memory
    shared_tile[transpose_idx] = my_data;
    block.sync();
}

// Wrapper functions for 2D FFT
__device__ void fft_2d(Complex shared_tile[32*32]) {
    fft_2d_32x32(shared_tile);
}

__device__ void ifft_2d(Complex shared_tile[32*32]) {
    ifft_2d_32x32(shared_tile);
}